"""
Shared runtime and bound request handle for UMA.

UMARuntime is the canonical internal execution surface for request-scoped
retrieval. UMAMemory remains the public-primary API and delegates to this
runtime internally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ...types import RuntimeContext
from ..retrieval.rlm.request import RetrievalRequest
from ..working_memory.core import session_scope_from_runtime_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UMARequestHandle:
    """Immutable request-bound runtime handle.

    This handle carries request-scoped context only. All execution delegates to
    the shared runtime instance.
    """

    runtime: "UMARuntime"
    context: RuntimeContext

    @property
    def tenant_id(self) -> str:
        return self.context.tenant_id

    @property
    def agent_id(self) -> str:
        return self.context.agent_id

    @property
    def request_id(self) -> str:
        return self.context.request_id

    @property
    def user_id(self) -> Optional[str]:
        return self.context.user_id

    @property
    def workspace_id(self) -> Optional[str]:
        return self.context.workspace_id

    @property
    def session_id(self) -> Optional[str]:
        return self.context.session_id

    async def retrieve_structured_context(self, query_text: str) -> Dict[str, list]:
        """Retrieve structured UMA context for this bound request."""
        return await self.runtime.retrieve_structured_context(
            self.context,
            query_text=query_text,
        )

    async def retrieve_rendered_context(self, query_text: str) -> str:
        """Retrieve rendered UMA context for this bound request."""
        return await self.runtime.retrieve_rendered_context(
            self.context,
            query_text=query_text,
        )

    async def get_context_messages(
        self,
        query_text: str,
        *,
        render_mode: str = "openclaw_v1",
    ) -> Dict[str, Any]:
        """Retrieve UMA context rendered as prompt messages for this request."""
        return await self.runtime.get_context_messages(
            self.context,
            query_text=query_text,
            render_mode=render_mode,
        )


@dataclass(frozen=True)
class UMABoundMemory:
    """Public ergonomic facade bound to one explicit request context.

    This keeps UMAMemory as the only public entrypoint while hiding
    UMARuntime and RuntimeContext from common usage.

    Example:
        memory = UMAMemory.from_yaml(path).for_context(
            user_id="user:1",
            agent_id="agent-default",
            tenant_id="default",
            request_id="chat:user:1",
        )
        snippet = await memory.retrieve_rendered_context(query_text="hello")
    """

    memory: Any
    context: RuntimeContext

    def __getattr__(self, name: str) -> Any:
        return getattr(self.memory, name)

    async def retrieve_structured_context(self, *, query_text: str) -> Dict[str, list]:
        """Retrieve structured context using the bound request context."""
        return await self.memory.runtime.retrieve_structured_context(
            self.context,
            query_text=query_text,
        )

    async def retrieve_rendered_context(self, *, query_text: str) -> str:
        """Retrieve rendered context using the bound request context."""
        return await self.memory.runtime.retrieve_rendered_context(
            self.context,
            query_text=query_text,
        )

    async def get_context_messages(
        self,
        *,
        query_text: str,
        render_mode: str = "openclaw_v1",
    ) -> Dict[str, Any]:
        """Retrieve context messages using the bound request context."""
        return await self.memory.runtime.get_context_messages(
            self.context,
            query_text=query_text,
            render_mode=render_mode,
        )

    async def get_structured_context(self, *, query_text: str) -> Dict[str, list]:
        """Compatibility alias for retrieve_structured_context."""
        return await self.retrieve_structured_context(query_text=query_text)

    async def get_rendered_context(self, *, query_text: str) -> str:
        """Compatibility alias for retrieve_rendered_context."""
        return await self.retrieve_rendered_context(query_text=query_text)


class UMARuntime:
    """Shared UMA infrastructure container.

    Request scope must never be stored on this object. Shared runtime state is
    refreshed from the attached UMAMemory facade when needed, while retrieval
    execution remains canonical here.
    """

    def __init__(
        self,
        *,
        config: Any = None,
        raw_config: Any = None,
        stores: Optional[Mapping[str, Any]] = None,
        llm: Any = None,
        agent_llm: Any = None,
        embedder: Any = None,
        document_store: Any = None,
        graph_service: Any = None,
        ranking_service: Any = None,
        feature_registry: Optional[Mapping[str, Any]] = None,
        memory_bridge: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = config
        self.raw_config = raw_config
        self.stores: Dict[str, Any] = dict(stores or {})
        self.llm = llm
        self.agent_llm = agent_llm
        self.embedder = embedder
        self.document_store = document_store
        self.graph_service = graph_service
        self.ranking_service = ranking_service
        self.feature_registry: Dict[str, Any] = dict(feature_registry or {})
        self.memory_bridge = memory_bridge
        self.metadata: Dict[str, Any] = dict(metadata or {})

    @classmethod
    def from_memory(cls, memory: Any) -> "UMARuntime":
        """Build a runtime view over an existing UMAMemory instance."""
        graph_service = getattr(memory, "graph_core", None)
        ranking_service = getattr(getattr(memory, "_rlm_controller", None), "ranker", None)
        return cls(
            config=getattr(memory, "cfg", None),
            raw_config=getattr(memory, "raw_config", None),
            stores=getattr(memory, "_stores", None) or {},
            llm=getattr(memory, "llm", None),
            agent_llm=getattr(memory, "agent_llm", None),
            embedder=getattr(memory, "embedder", None),
            document_store=getattr(memory, "document_store", None),
            graph_service=graph_service,
            ranking_service=ranking_service,
            feature_registry=getattr(memory, "features", None) or {},
            memory_bridge=memory,
            metadata={"source": "UMAMemory"},
        )

    def bind(self, context: RuntimeContext) -> UMARequestHandle:
        """Return an immutable request handle bound to ``context``."""
        if not isinstance(context, RuntimeContext):
            raise TypeError("UMARuntime.bind requires a RuntimeContext instance")
        return UMARequestHandle(runtime=self, context=context)

    def _require_memory_bridge(self) -> Any:
        memory = self.memory_bridge
        if memory is None:
            raise RuntimeError("UMARuntime operation requires a memory_bridge.")
        return memory

    def refresh_from_memory(self) -> None:
        """Refresh runtime-owned shared service references from UMAMemory."""
        memory = self._require_memory_bridge()
        self.config = getattr(memory, "cfg", None)
        self.raw_config = getattr(memory, "raw_config", None)
        self.stores = dict(getattr(memory, "_stores", None) or {})
        self.llm = getattr(memory, "llm", None)
        self.agent_llm = getattr(memory, "agent_llm", None)
        self.embedder = getattr(memory, "embedder", None)
        self.document_store = getattr(memory, "document_store", None)
        self.graph_service = getattr(memory, "graph_core", None)
        self.ranking_service = getattr(getattr(memory, "_rlm_controller", None), "ranker", None)
        self.feature_registry = dict(getattr(memory, "features", None) or {})

    def ensure_retrieval_ready(self) -> None:
        """Ensure retrieval-only dependencies are initialized."""
        memory = self._require_memory_bridge()
        memory._ensure_retrieval_ready()
        self.refresh_from_memory()

    @staticmethod
    def _build_retrieval_request(context: RuntimeContext) -> RetrievalRequest:
        """Convert a RuntimeContext into a RetrievalRequest."""
        return RetrievalRequest.from_runtime_context(
            context,
            trace_id=context.request_id,
        )

    @staticmethod
    def _working_memory_scope_for_context(context: RuntimeContext) -> Optional[Any]:
        """Build the working-memory scope for a runtime context."""
        return session_scope_from_runtime_context(context)

    async def retrieve_structured_context(
        self,
        context: RuntimeContext,
        *,
        query_text: str,
    ) -> Dict[str, list]:
        """Retrieve structured UMA context for one explicit request scope."""
        from ...adapters.observability.metrics import increment, timed
        from ..retrieval.rlm.coverage import compute_confidence

        if not isinstance(context, RuntimeContext):
            raise TypeError("UMARuntime retrieval requires a RuntimeContext instance.")
        if not context.user_id:
            raise ValueError("UMARuntime retrieval requires RuntimeContext.user_id.")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(
                "UMARuntime.retrieve_structured_context: query_text must be a non-empty string."
            )

        self.ensure_retrieval_ready()
        memory = self._require_memory_bridge()

        with timed("uma.get_structured_context.latency"):
            try:
                wm_scope = self._working_memory_scope_for_context(context)
                wm_core = getattr(memory, "working_memory", None)
                wm_stored = (
                    wm_core.get_context(wm_scope)
                    if wm_core is not None and wm_scope is not None
                    else []
                )
            except Exception:
                logger.exception(
                    "UMARuntime.retrieve_structured_context: failed to load WM "
                    "tenant=%s agent=%s session=%s",
                    context.tenant_id,
                    context.agent_id,
                    context.session_id,
                )
                wm_stored = []

            controller = getattr(memory, "_rlm_controller", None)
            if controller is None:
                raise RuntimeError(
                    "UMARuntime.retrieve_structured_context: RLM controller not initialized."
                )

            pack = await controller.retrieve_context(
                request=self._build_retrieval_request(context),
                query_text=query_text.strip(),
            )
            increment("uma.get_structured_context.calls", tags={"path": "rlm"})
            coverage = getattr(pack, "coverage", None)

            return {
                "working_memory": wm_stored,
                "episodic": pack.episodes,
                "facts": pack.facts or [],
                "chunks": getattr(pack, "chunks", []),
                "skills": pack.skills,
                "graph": pack.graph,
                "trace": pack.steps,
                "confidence": compute_confidence(coverage) if coverage is not None else {},
            }

    async def retrieve_rendered_context(
        self,
        context: RuntimeContext,
        *,
        query_text: str,
    ) -> str:
        """Retrieve rendered UMA context for one explicit request scope."""
        from ..utils.context_pack_builder import ContextPackBuilder

        structured = await self.retrieve_structured_context(
            context,
            query_text=query_text,
        )
        pack = ContextPackBuilder.build(query_text, structured)
        ctx_cfg = getattr(getattr(self.config, "retrieval", None), "context", None)

        if getattr(ctx_cfg, "snippet_refiner_enabled", False):
            return await ContextPackBuilder.render_snippet_async(
                pack,
                ctx_cfg,
                llm=self.llm,
            )
        return ContextPackBuilder.render_snippet(pack, ctx_cfg)

    async def get_context_messages(
        self,
        context: RuntimeContext,
        *,
        query_text: str,
        render_mode: str = "openclaw_v1",
    ) -> Dict[str, Any]:
        """Retrieve context formatted as prompt messages."""
        if not isinstance(context, RuntimeContext):
            raise TypeError("UMARuntime retrieval requires a RuntimeContext instance.")
        if not context.user_id:
            raise ValueError("UMARuntime.get_context_messages: RuntimeContext.user_id is required.")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("UMARuntime.get_context_messages: query_text must be a non-empty string.")
        if not isinstance(render_mode, str) or not render_mode.strip():
            raise ValueError("UMARuntime.get_context_messages: render_mode must be a non-empty string.")

        normalized_render_mode = render_mode.strip()
        if normalized_render_mode not in {"openclaw_v1", "raw_rendered"}:
            raise ValueError(
                f"UMARuntime.get_context_messages: unsupported render_mode={normalized_render_mode!r}. "
                "Supported modes: 'openclaw_v1', 'raw_rendered'."
            )

        rendered = await self.retrieve_rendered_context(
            context,
            query_text=query_text,
        )
        rendered = (rendered or "").strip()

        messages: list[dict[str, str]] = []
        if rendered:
            if normalized_render_mode == "openclaw_v1":
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Relevant memory context from UMA follows. "
                            "Use it only as supporting context; prefer direct task instructions "
                            "and the current conversation when they conflict.\n\n"
                            f"{rendered}"
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "system",
                        "content": rendered,
                    }
                )

        return {
            "messages": messages,
            "meta": {
                "provider": "uma",
                "format": "message_list",
                "render_mode": normalized_render_mode,
                "message_count": len(messages),
            },
        }