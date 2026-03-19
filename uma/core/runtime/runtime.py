"""
Shared runtime and bound request handle skeleton for UMA.

This module establishes the architecture boundary introduced in PR 2:
- UMARuntime owns shared infrastructure references only
- UMARequestHandle carries immutable runtime context for one request/session

Behavioral cutover is intentionally out of scope. Existing production flows
continue to use UMAMemory directly for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ...types import RuntimeContext


@dataclass(frozen=True)
class UMARequestHandle:
    """
    Immutable bound handle for one runtime context.

    This handle is intentionally thin in PR 2. It exists to establish the
    separation between shared runtime services and request-scoped context
    without migrating any execution behavior yet.
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

    def _require_memory_bridge(self) -> Any:
        memory = getattr(self.runtime, "memory_bridge", None)
        if memory is None:
            raise RuntimeError(
                "UMARequestHandle retrieval requires a runtime with a memory_bridge."
            )
        return memory

    async def retrieve_structured_context(self, query_text: str) -> Dict[str, list]:
        memory = self._require_memory_bridge()
        return await memory._retrieve_structured_context_for_context(
            self.context,
            query_text=query_text,
        )

    async def retrieve_rendered_context(self, query_text: str) -> str:
        memory = self._require_memory_bridge()
        return await memory._retrieve_rendered_context_for_context(
            self.context,
            query_text=query_text,
        )

    async def get_context_messages(
        self,
        query_text: str,
        *,
        render_mode: str = "openclaw_v1",
    ) -> Dict[str, Any]:
        memory = self._require_memory_bridge()
        return await memory._get_context_messages_for_context(
            self.context,
            query_text=query_text,
            render_mode=render_mode,
        )


class UMARuntime:
    """
    Shared service container for UMA infrastructure.

    Request scope must never be stored on this object. It may be shared safely
    across many bound handles that each carry distinct immutable contexts.
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
        # Passive coexistence bridge during the runtime redesign. This is not
        # request scope and must not become a hidden execution shortcut.
        self.memory_bridge = memory_bridge
        self.metadata: Dict[str, Any] = dict(metadata or {})

    @classmethod
    def from_memory(cls, memory: Any) -> "UMARuntime":
        """
        Build a shared runtime view over an existing UMAMemory instance.

        This bridge keeps UMAMemory as the primary public execution path while
        allowing new bound handles to coexist without behavioral cutover.
        """

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
        """
        Return an immutable request handle bound to the supplied runtime context.
        """

        if not isinstance(context, RuntimeContext):
            raise TypeError("UMARuntime.bind requires a RuntimeContext instance")
        return UMARequestHandle(runtime=self, context=context)
