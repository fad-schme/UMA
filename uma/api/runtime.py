"""
Shared runtime and bound request handle for UMA.

UMARuntime is the canonical internal execution surface for request-scoped
retrieval. UMAMemory remains the public-primary API and delegates to this
runtime internally.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from uma.common.types import RuntimeContext
from uma.common.storage_metadata import (
    EPISODIC_LANE,
    KB_LANES,
    PROCEDURAL_LANE,
    PROFILE_LANE,
    RAW_LANE,
    SEMANTIC_LANE,
    shared_metadata_view,
)
from uma.retrieve.planner import build_retrieval_plan
from uma.retrieve.rlm.request import RetrievalRequest
from uma.memory.working_memory.core import session_scope_from_runtime_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UMARequestHandle:
    """Immutable request-bound runtime handle.

    The handle carries one explicit `RuntimeContext` and delegates execution to
    the shared `UMARuntime` instance without mutating runtime-global state.
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

    async def retrieve_context(
        self,
        query_text: str,
        *,
        lane_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return await self.runtime.retrieve_context(
            self.context,
            query_text=query_text,
            lane_filter=lane_filter,
        )

    async def retrieve_memory(
        self,
        query_text: str,
        *,
        memory_intent: str = "continuity",
    ) -> Dict[str, Any]:
        return await self.runtime.retrieve_memory(
            self.context,
            query_text=query_text,
            memory_intent=memory_intent,
        )

    async def get_context_messages(
        self,
        query_text: str,
        *,
        render_mode: str = "animus_v1",
    ) -> Dict[str, Any]:
        return await self.runtime.get_context_messages(
            self.context,
            query_text=query_text,
            render_mode=render_mode,
        )

@dataclass(frozen=True)
class _AnimusProfileCacheEntry:
    """One cached markdown profile file.

    The provider keeps the source path and refreshes the cached text lazily when
    the TTL expires. This is intentionally simple because OpenClaw uses one
    current user profile and one current agent profile.
    """

    path: str
    text: str
    loaded_at: float
    expires_at: float


class AnimusProfileProvider:
    """Small in-memory provider for USER.md and SOUL.md overlays.

    Phase 1 design:
    - no DB persistence
    - no partial field editing
    - load from markdown files provided by the integration
    - refresh cached content when the TTL expires
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        if ttl_seconds <= 0:
            raise ValueError("AnimusProfileProvider.ttl_seconds must be a positive integer.")
        self.ttl_seconds = ttl_seconds
        self._user_profile: Optional[_AnimusProfileCacheEntry] = None
        self._agent_profile: Optional[_AnimusProfileCacheEntry] = None

    def load_user_profile(self, path: str) -> None:
        """Load and cache the current USER.md profile."""
        self._user_profile = self._load_profile(path)
        logger.info("AnimusProfileProvider: loaded user profile from %s", self._user_profile.path)

    def load_agent_profile(self, path: str) -> None:
        """Load and cache the current SOUL.md profile."""
        self._agent_profile = self._load_profile(path)
        logger.info("AnimusProfileProvider: loaded agent profile from %s", self._agent_profile.path)

    def get_user_profile_text(self) -> str:
        """Return cached USER.md content, refreshing it after TTL expiry."""
        self._user_profile = self._refresh_if_needed(self._user_profile)
        return self._user_profile.text if self._user_profile is not None else ""

    def get_agent_profile_text(self) -> str:
        """Return cached SOUL.md content, refreshing it after TTL expiry."""
        self._agent_profile = self._refresh_if_needed(self._agent_profile)
        return self._agent_profile.text if self._agent_profile is not None else ""

    def _refresh_if_needed(
        self,
        entry: Optional[_AnimusProfileCacheEntry],
    ) -> Optional[_AnimusProfileCacheEntry]:
        if entry is None:
            return None
        if time.time() < entry.expires_at:
            return entry
        refreshed = self._load_profile(entry.path)
        logger.info("AnimusProfileProvider: refreshed profile from %s after TTL expiry", refreshed.path)
        return refreshed

    def _load_profile(self, path: str) -> _AnimusProfileCacheEntry:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("AnimusProfileProvider path must be a non-empty string.")

        normalized_path = path.strip()
        try:
            with open(normalized_path, "r", encoding="utf-8") as handle:
                text = handle.read().strip()
        except FileNotFoundError as exc:
            logger.exception("AnimusProfileProvider: profile file not found: %s", normalized_path)
            raise RuntimeError(f"Profile file not found: {normalized_path}") from exc
        except Exception as exc:
            logger.exception("AnimusProfileProvider: failed to read profile file: %s", normalized_path)
            raise RuntimeError(f"Failed to read profile file: {normalized_path}") from exc

        now = time.time()
        return _AnimusProfileCacheEntry(
            path=normalized_path,
            text=text,
            loaded_at=now,
            expires_at=now + self.ttl_seconds,
        )
    

class UMARuntime:
    """Shared UMA infrastructure container.

    Request scope must never be stored on this object. UMAMemory owns
    initialized services; this runtime reads them through the attached memory
    bridge when present while keeping request-scoped execution canonical here.
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
        self._config = config
        self._stores: Dict[str, Any] = dict(stores or {})
        self._llm = llm
        self.memory_bridge = memory_bridge
        self.metadata: Dict[str, Any] = dict(metadata or {})
        del raw_config
        del agent_llm
        del embedder
        del document_store
        del graph_service
        del ranking_service
        del feature_registry

    @classmethod
    def from_memory(cls, memory: Any) -> "UMARuntime":
        """Build a runtime view over an existing UMAMemory instance."""
        return cls(
            memory_bridge=memory,
            metadata={"source": "UMAMemory"},
        )


    def _require_memory_bridge(self) -> Any:
        memory = self.memory_bridge
        if memory is None:
            raise RuntimeError("UMARuntime operation requires a memory_bridge.")
        return memory

    @property
    def config(self) -> Any:
        memory = self.memory_bridge
        if memory is not None:
            return getattr(memory, "cfg", None)
        return self._config

    @property
    def stores(self) -> Dict[str, Any]:
        memory = self.memory_bridge
        if memory is not None:
            return dict(getattr(memory, "_stores", None) or {})
        return self._stores

    @property
    def llm(self) -> Any:
        memory = self.memory_bridge
        if memory is not None:
            return getattr(memory, "llm", None)
        return self._llm

    @property
    def agent_id(self) -> Optional[str]:
        """Return the bridge runtime agent identity, if one is known."""
        memory = self.memory_bridge
        if memory is None:
            return None
        agent_id = getattr(memory, "agent_id", None)
        if isinstance(agent_id, str):
            normalized = agent_id.strip()
            return normalized or None
        return None

    def ensure_retrieval_ready(self) -> None:
        """Ensure retrieval-only dependencies are initialized."""
        memory = self._require_memory_bridge()
        memory._ensure_retrieval_ready()

    def bind(self, context: RuntimeContext) -> UMARequestHandle:
        """Bind one explicit request context without mutating shared runtime state."""
        if not isinstance(context, RuntimeContext):
            raise TypeError("UMARuntime.bind requires a RuntimeContext instance.")
        return UMARequestHandle(runtime=self, context=context)

    def _available_retrieval_lanes(self) -> List[str]:
        memory = self.memory_bridge
        stores = self.stores or {}
        lanes: List[str] = []
        if getattr(memory, "chunk_core", None) is not None or stores.get("chunk") is not None:
            lanes.append(RAW_LANE)
        if getattr(memory, "semantic_core", None) is not None or stores.get("semantic") is not None:
            lanes.extend([SEMANTIC_LANE, PROFILE_LANE])
        if getattr(memory, "episodic_core", None) is not None or stores.get("episodic") is not None:
            lanes.append(EPISODIC_LANE)
        if getattr(memory, "procedural_core", None) is not None or stores.get("procedural") is not None:
            lanes.append(PROCEDURAL_LANE)
        seen: set[str] = set()
        return [lane for lane in lanes if not (lane in seen or seen.add(lane))]

    @staticmethod
    def _build_retrieval_request(
        context: RuntimeContext,
        *,
        plan: Optional[Any] = None,
    ) -> RetrievalRequest:
        """Convert a RuntimeContext into a RetrievalRequest."""
        return RetrievalRequest.from_runtime_context(
            context,
            trace_id=context.request_id,
            plan=plan,
        )

    @staticmethod
    def _working_memory_scope_set_context(context: RuntimeContext) -> Optional[Any]:
        """Build the working-memory scope for a runtime context."""
        return session_scope_from_runtime_context(context)

    @staticmethod
    def _normalize_lane_filter(lane_filter: Optional[List[str]]) -> List[str]:
        if lane_filter is None:
            return []
        normalized: List[str] = []
        seen: set[str] = set()
        for lane in lane_filter:
            candidate = str(lane or "").strip().lower()
            if not candidate:
                continue
            if candidate not in KB_LANES:
                raise ValueError(f"UMARuntime.retrieve_context: invalid lane_filter value {lane!r}")
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    @staticmethod
    def _item_lane(item: Any) -> Optional[str]:
        meta = getattr(item, "meta", None)
        if isinstance(item, dict):
            meta = item.get("meta", meta)
        if not isinstance(meta, dict):
            return None
        raw = meta.get("kb_lane")
        if raw is None:
            return None
        lane = str(raw).strip().lower()
        return lane or None

    @classmethod
    def _filter_items_by_lanes(cls, items: List[Any], lane_filter: List[str]) -> List[Any]:
        if not lane_filter:
            return list(items or [])
        allowed = set(lane_filter)
        return [item for item in (items or []) if cls._item_lane(item) in allowed]

    def _load_working_memory_for_context(
        self,
        runtime_context: RuntimeContext,
    ) -> List[Any]:
        memory = self._require_memory_bridge()
        try:
            wm_scope = self._working_memory_scope_set_context(runtime_context)
            wm_core = getattr(memory, "working_memory", None)
            return (
                wm_core.get_context(wm_scope)
                if wm_core is not None and wm_scope is not None
                else []
            )
        except Exception:
            logger.exception(
                "UMARuntime.retrieve_context: failed to load WM "
                "tenant=%s agent=%s session=%s",
                runtime_context.tenant_id,
                runtime_context.agent_id,
                runtime_context.session_id,
            )
            return []

    def _assemble_public_context_result(
        self,
        *,
        query_text: str,
        lane_filter: List[str],
        plan: Any,
        working_memory: List[Any],
        pack: Any,
    ) -> Dict[str, Any]:
        from uma.retrieve.rlm.coverage import compute_confidence

        coverage = getattr(pack, "coverage", None)
        active_lanes = list(plan.participating_lanes)
        episodic = self._filter_items_by_lanes(pack.episodes, active_lanes)
        facts = self._filter_items_by_lanes(pack.facts or [], active_lanes)
        chunks = self._filter_items_by_lanes(getattr(pack, "chunks", []), active_lanes)
        skills = self._filter_items_by_lanes(pack.skills, active_lanes)

        return {
            "product": "context",
            "query": query_text,
            "lane_filter": list(lane_filter),
            "active_lanes": active_lanes,
            "working_memory": working_memory,
            "episodic": episodic,
            "facts": facts,
            "chunks": chunks,
            "documents": self._group_memory_artifacts(chunks),
            "skills": skills,
            "graph": pack.graph,
            "trace": list(getattr(pack, "steps", []) or []),
            "confidence": compute_confidence(coverage) if coverage is not None else {},
        }

    
    async def retrieve_context(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
        lane_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve curated evidence-oriented UMA context for one explicit request scope.

        Contract:
        - context retrieval is the canonical RAG/context path
        - a small lane planner decides which persisted lanes participate before RLM runs
        - chunks/documents remain the primary evidence product
        - wiki/compiled memory state is not required by default
        - `lane_filter` applies only to persisted retrieval lanes, not working memory
        """
        from uma.adapters.observability.metrics import increment, timed

        if not isinstance(runtime_context, RuntimeContext):
            raise TypeError("UMARuntime retrieval requires a RuntimeContext instance.")
        if not runtime_context.user_id:
            raise ValueError("UMARuntime retrieval requires RuntimeContext.user_id.")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(
                "UMARuntime.retrieve_context: query_text must be a non-empty string."
            )
        normalized_query_text = query_text.strip()
        self.ensure_retrieval_ready()
        normalized_lane_filter = self._normalize_lane_filter(lane_filter)
        plan = build_retrieval_plan(
            product="context",
            query_text=normalized_query_text,
            available_lanes=self._available_retrieval_lanes(),
            lane_filter=normalized_lane_filter,
        )

        with timed("uma.get_structured_context.latency"):
            working_memory = self._load_working_memory_for_context(runtime_context)
            memory = self._require_memory_bridge()
            controller = getattr(memory, "_rlm_controller", None)
            if controller is None:
                raise RuntimeError(
                    "UMARuntime.retrieve_context: RLM controller not initialized."
                )
            pack = await controller.retrieve_context(
                request=self._build_retrieval_request(runtime_context, plan=plan),
                query_text=normalized_query_text,
            )
            increment("uma.get_structured_context.calls", tags={"path": "rlm"})
            return self._assemble_public_context_result(
                query_text=normalized_query_text,
                lane_filter=normalized_lane_filter,
                plan=plan,
                working_memory=working_memory,
                pack=pack,
            )

    @staticmethod
    def _group_memory_artifacts(chunks: List[Any]) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks or []:
            chunk_id = getattr(chunk, "id", None)
            doc_id = str(getattr(chunk, "doc_id", None) or chunk_id or "")
            if not doc_id:
                continue
            meta = dict(getattr(chunk, "meta", None) or {})
            artifact = grouped.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "kind": meta.get("kind"),
                    "kb_lane": meta.get("kb_lane"),
                    "owner_type": getattr(chunk, "owner_type", None),
                    "owner_id": getattr(chunk, "owner_id", None),
                    "source_path": getattr(chunk, "source_path", None),
                    "metadata": shared_metadata_view(
                        meta=meta,
                        owner_type=str(getattr(chunk, "owner_type", None) or ""),
                        owner_id=str(getattr(chunk, "owner_id", None) or ""),
                        created_at=getattr(chunk, "created_at", None),
                        updated_at=getattr(chunk, "updated_at", None),
                    ),
                    "page_ranges": [],
                    "chunk_ids": [],
                    "evidence": [],
                },
            )
            page_range = getattr(chunk, "page_range", None)
            if page_range and page_range not in artifact["page_ranges"]:
                artifact["page_ranges"].append(page_range)
            if chunk_id and chunk_id not in artifact["chunk_ids"]:
                artifact["chunk_ids"].append(chunk_id)
            artifact["evidence"].append(chunk)
        return list(grouped.values())

    async def retrieve_memory(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
        memory_intent: str = "continuity",
    ) -> Dict[str, Any]:
        """Retrieve compiled, evidence-backed memory results for one request scope.

        This path uses its own lane plan, may reuse context-path candidate
        gathering beneath that plan, and must not silently collapse into
        context retrieval. If compiled memory artifacts are unavailable, the
        result surfaces an explicit evidence-only fallback.
        """
        if not isinstance(memory_intent, str) or not memory_intent.strip():
            raise ValueError("UMARuntime.retrieve_memory: memory_intent must be a non-empty string.")
        self.ensure_retrieval_ready()
        plan = build_retrieval_plan(
            product="memory",
            query_text=query_text.strip(),
            available_lanes=self._available_retrieval_lanes(),
            memory_intent=memory_intent.strip(),
        )
        context = await self.retrieve_context(
            runtime_context,
            query_text=query_text,
            lane_filter=list(plan.participating_lanes),
        )
        chunks = list(context.get("chunks") or [])
        memories: List[Dict[str, Any]] = []
        fallback_used = not memories
        fallback_reason = "no_compiled_memory_available" if fallback_used else None
        if fallback_used:
            logger.info(
                "UMARuntime.retrieve_memory: explicit evidence-only fallback tenant=%s agent=%s user=%s intent=%s chunks=%d",
                runtime_context.tenant_id,
                runtime_context.agent_id,
                runtime_context.user_id,
                memory_intent,
                len(chunks),
            )
        return {
            "query": query_text.strip(),
            "product": "memory",
            "memory_intent": memory_intent.strip(),
            "memories": memories,
            "evidence": chunks,
            "supporting_facts": list(context.get("facts") or []),
            "supporting_skills": list(context.get("skills") or []),
            "conflicts": [],
            "fallback": {
                "used": fallback_used,
                "mode": "evidence_only" if fallback_used else "none",
                "reason": fallback_reason,
                "message": (
                    "No compiled memory artifacts were available; returning evidence-only fallback."
                    if fallback_used
                    else ""
                ),
            },
            "memory_sources": self._group_memory_artifacts(chunks),
            "confidence": dict(context.get("confidence") or {}),
            "trace": list(context.get("trace") or [])
            + [
                {
                    "event": "memory_plan",
                    "product": "memory",
                    "memory_intent": memory_intent.strip(),
                    "requested_lanes": list(plan.requested_lanes),
                    "participating_lanes": list(plan.participating_lanes),
                    "excluded_lanes": [dict(item) for item in plan.excluded_lanes],
                    "fallback_used": fallback_used,
                }
            ],
        }

    def _render_profile_overlay(self) -> str:
        """Render cached USER.md and SOUL.md overlays for every response.

        The integration loads the files explicitly through UMAMemory. Retrieval
        remains deterministic here: if a profile is loaded, it is prepended to
        the rendered context on every request.
        """
        memory = self._require_memory_bridge()
        provider = getattr(memory, "animus_profile_provider", None)
        if provider is None:
            return ""

        try:
            agent_profile = provider.get_agent_profile_text().strip()
            user_profile = provider.get_user_profile_text().strip()
        except Exception:
            logger.exception("UMARuntime: failed to load cached Animus profiles.")
            return ""

        sections: list[str] = []
        if agent_profile:
            sections.append("## Agent Profile\n" + agent_profile)
        if user_profile:
            sections.append("## User Profile\n" + user_profile)
        return "\n\n".join(sections).strip()


    async def render_context(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
    ) -> str:
        """Render retrieved context for presentation after canonical context retrieval."""
        from uma.retrieve.context_pack_builder import ContextPackBuilder

        structured = await self.retrieve_context(
            runtime_context,
            query_text=query_text,
        )
        pack = ContextPackBuilder.build(query_text, structured)
        ctx_cfg = getattr(getattr(self.config, "retrieval", None), "context", None)

        if getattr(ctx_cfg, "snippet_refiner_enabled", False):
            rendered_memory = await ContextPackBuilder.render_snippet_async(
                pack,
                ctx_cfg,
                llm=self.llm,
            )
        else:
            rendered_memory = ContextPackBuilder.render_snippet(pack, ctx_cfg)

        profile_overlay = self._render_profile_overlay()
        parts = [part.strip() for part in (profile_overlay, rendered_memory) if part and part.strip()]
        return "\n\n".join(parts).strip()

    async def get_context_messages(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
        render_mode: str = "animus_v1",
    ) -> Dict[str, Any]:
        """Retrieve context formatted as prompt messages."""
        if not isinstance(runtime_context, RuntimeContext):
            raise TypeError("UMARuntime retrieval requires a RuntimeContext instance.")
        if not runtime_context.user_id:
            raise ValueError("UMARuntime.get_context_messages: RuntimeContext.user_id is required.")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("UMARuntime.get_context_messages: query_text must be a non-empty string.")
        if not isinstance(render_mode, str) or not render_mode.strip():
            raise ValueError("UMARuntime.get_context_messages: render_mode must be a non-empty string.")

        normalized_render_mode = render_mode.strip()
        if normalized_render_mode not in {"animus_v1", "raw_rendered"}:
            raise ValueError(
                f"UMARuntime.get_context_messages: unsupported render_mode={normalized_render_mode!r}. "
                "Supported modes: 'animus_v1', 'raw_rendered'."
            )

        rendered = await self.render_context(
            runtime_context,
            query_text=query_text,
        )
        rendered = (rendered or "").strip()

        messages: list[dict[str, str]] = []
        if rendered:
            if normalized_render_mode == "animus_v1":
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
    
