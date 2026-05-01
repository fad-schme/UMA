r"""
  _   _ __  __   _      ___ _    __  __
 | | | |  \/  | /_\ ___| _ \ |  |  \/  |
 | |_| | |\/| |/ _ \___|   / |__| |\/| |
  \___/|_|  |_/_/ \_\  |_|_\____|_|  |_|

    UMAMemory — Core UMA Memory Runtime
    =====================================

UMA is a **memory SDK**, not an autonomous agent.

This class owns and orchestrates all UMA memory subsystems:

    • Working Memory (short-term conversational state)
    • Episodic Memory (indexed event history)
    • Fact Memory (facts, preferences, domain knowledge)
    • Skill Memory (skills, routines)
    • Temporal Graph (optional knowledge graph)

UMA does not generate assistant replies and does not perform agent reasoning.
Developers bring their own LLM or agent loop and use UMA strictly for memory
management.

Typical developer workflow
--------------------------

1. Initialize UMA:

    memory = UMAMemory.from_yaml("uma.yaml")

2. Bind request scope ergonomically for repeated retrieval:

    memory = memory.set_context(
        user_id=user_id,
        agent_id="agent-default",
        tenant_id="default",
        request_id="req-1",
        session_id="session-1",
    )

3. Retrieve context:

    context = await memory.retrieve_context(
        query_text="the user's current question or task"
    )

4. Ingest conversation turns or documents through the public APIs on UMAMemory.

Design Philosophy
-----------------
- UMA provides *memory operations*, not agent behaviors.
- All reasoning, tool use, and final reply generation happen outside UMA.
- UMA focuses on correctness, retrieval quality, summarization, and
  long-term memory structure.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Dict, Optional

from uma.common.config import UMAConfig
from uma.common.config_types import RuntimeConfig, parse_plugin_spec
from uma.common.hooks import UMAHooks
from uma.common.identity import normalize_user_id
from uma.retrieve.user_query_helper import _build_bootstrap_skip_result, _build_memory_bootstrap_signature, _extract_memory_bootstrap_lines, _is_bootstrap_manifest_duplicate, _persist_bootstrap_manifest, _read_bootstrap_file, _validate_bootstrap_file_path, build_fact_embedding_text
from uma.memory.working_memory.core import WorkingMemoryCore
from uma.memory.episodic.core import EpisodicCore
from uma.memory.episodic.indexer import EpisodeIndexer
from uma.memory.episodic.policies import EpisodicRetentionPolicy
from uma.memory.semantic.core import SemanticCore
from uma.memory.procedural.core import ProceduralCore
from uma.memory.chunk.core import ChunkCore
from uma.memory.graph import TemporalGraphCore
from uma.common.initializers.runtime import (
    init_retrieval_ready,
    init_ingestion_ready,
    schedule_ingestion_warmup,
)
from uma.common.registry import FeatureLoader, FeaturePolicy, default_feature_registry
from .runtime import AnimusProfileProvider, UMARuntime
from uma.common.storage_metadata import normalize_document_metadata, normalize_fact_metadata
from uma.common.types import Fact, RuntimeContext, TargetOwner
from ..stores.document_sql import DocumentRecord

logger = logging.getLogger(__name__)


class UMAMemory:
    """UMA Memory Runtime Container.

    This class:
        - initializes all core subsystems lazily
        - provides public APIs for retrieval, ingestion, and maintenance
        - loads optional features via direct attachment


    UMARuntime remains internal and canonical for retrieval execution.
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "UMAMemory":
        """Create a UMAMemory instance from YAML config."""
        cfg = UMAConfig.load_yaml(path)
        mem = cls(cfg, config_path=path)

        # Predictable startup cost: make retrieval instant thereafter.
        init_retrieval_ready(mem)
        mem._retrieval_ready = True
        mem.initialized = True

        # Background warmup: ingestion subsystems and optional memory features.
        schedule_ingestion_warmup(mem)

        return mem

    def __init__(self, config: UMAConfig, config_path: Optional[str] = None) -> None:
        """Initialize the UMA memory container from validated config."""
        self.raw_config = config
        self._config_path = config_path or getattr(config, "_source_path", None)
        self._config_dir = getattr(config, "_source_dir", None)
        self.initialized: bool = False
        self._features_initialized: bool = False
        self._retrieval_ready: bool = False
        self._ingestion_ready: bool = False
        self._warmup_scheduled: bool = False
        self._lifecycle_lock = threading.RLock()
        self._init_condition = threading.Condition(self._lifecycle_lock)
        self._init_inflight: set[str] = set()

        # RLM components (initialized later)
        self.memory_env = None
        self._rlm_controller = None

        # Unified runtime config
        self.cfg = RuntimeConfig.from_uma_config(config)
        self.llm_cfg = self.cfg.llm
        self.agent_llm_cfg = self.cfg.agent_llm
        self.embedding_cfg = self.cfg.embedding
        self.working_memory_cfg = self.cfg.working_memory
        self.retrieval_cfg = self.cfg.retrieval
        self.features_cfg = self.cfg.features
        self.consolidation_cfg = self.cfg.consolidation
        self.pipeline_cfg = self.cfg.pipeline
        self.semantic_salience_threshold = self.cfg.semantic_salience_threshold
        self._agent_id: Optional[str] = None
        self._runtime: Optional[UMARuntime] = None
        self._bound_runtime_context: Optional[RuntimeContext] = None
        self.animus_profile_provider = AnimusProfileProvider()

        # Internal store registry (core-only access; no direct store usage outside cores)
        self._stores: Dict[str, Any] = {}

        # Hooks + Feature attachment registry
        self.hooks = UMAHooks()
        self.features: Dict[str, Any] = {}
        self._feature_policy = FeaturePolicy()

        # Optional promotion policy (set by features or left None)
        self.promotion_policy: Optional[Any] = None

        # Core runtime components (initialized later)
        self.llm: Any = None
        self.agent_llm: Any = None
        self.embedder: Any = None
        self.pipeline: Optional[Any] = None
        self.document_store: Optional[Any] = None

        self.working_memory: Optional[WorkingMemoryCore] = None
        self.semantic_core: Optional[SemanticCore] = None
        self.episodic_core: Optional[EpisodicCore] = None
        self.graph_core: Optional[TemporalGraphCore] = None
        self.procedural_core: Optional[ProceduralCore] = None
        self.chunk_core: Optional[ChunkCore] = None

        logger.debug("UMAMemory instance created with unified storage config.")

    # ----------------------------------------------------------------------
    # Internal lazy initialization
    # ----------------------------------------------------------------------

    @property
    def runtime(self) -> UMARuntime:
        """Return the shared internal runtime, refreshing lazy-owned services."""
        if self._runtime is None:
            self._runtime = UMARuntime.from_memory(self)
        else:
            self._runtime.refresh_from_memory()
        return self._runtime

    @property
    def agent_id(self) -> Optional[str]:
        """Return the current runtime agent identity, if one is known.

        This remains request/runtime state, not durable configuration. A value
        may come from a bound runtime context or from internal bootstrap code
        that seeds `_agent_id` before request-scoped flows execute.
        """
        agent_id = self._agent_id
        if isinstance(agent_id, str):
            normalized = agent_id.strip()
            return normalized or None

        runtime_context = self._bound_runtime_context
        if runtime_context is not None and runtime_context.agent_id:
            return runtime_context.agent_id
        return None

    def _require_bound_runtime_context(self) -> RuntimeContext:
        """Return the currently bound runtime context for public retrieval APIs."""
        runtime_context = self._bound_runtime_context
        if runtime_context is None:
            raise RuntimeError(
                "UMAMemory requires a bound runtime_context. Call set_context(...) before retrieval."
            )
        return runtime_context

    @staticmethod
    def _extract_daily_diary_entries(raw_text: str) -> list[str]:
        """Extract one diary entry per markdown bullet."""
        entries: list[str] = []
        for line in raw_text.splitlines():
            stripped = line.lstrip()
            if not stripped.startswith("- "):
                continue
            entry = stripped[2:].strip()
            if entry:
                entries.append(entry)
        return entries

    @staticmethod
    def _build_diary_bootstrap_signature(*, raw_text: str) -> dict:
        """Build a stable manifest signature for diary bootstrap idempotency."""
        return {
            "pipeline_version": "daily_diary_bootstrap_v1",
            "content_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        }

    def _ensure_base_ready(self) -> None:
        """Ensure the minimum baseline runtime is ready."""
        if not getattr(self, "_retrieval_ready", False):
            init_retrieval_ready(self)
        self.initialized = True

    def _ensure_retrieval_ready(self) -> None:
        """Ensure retrieval-only dependencies are ready."""
        if not getattr(self, "_retrieval_ready", False):
            init_retrieval_ready(self)
        self.initialized = True

    def _ensure_ingestion_ready(self) -> None:
        """Ensure ingestion dependencies are ready."""
        if not getattr(self, "_ingestion_ready", False):
            init_ingestion_ready(self)
        self.initialized = True

    def _build_runtime_context(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: str = "default",
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> RuntimeContext:
        """Build a validated runtime context for public retrieval APIs."""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("UMAMemory retrieval requires a non-empty user_id.")

        resolved_agent_id = (agent_id or self.agent_id or "agent-default").strip()
        if not resolved_agent_id:
            raise ValueError("UMAMemory retrieval requires a non-empty agent_id.")

        resolved_tenant_id = (tenant_id or "default").strip()
        if not resolved_tenant_id:
            raise ValueError("UMAMemory retrieval requires a non-empty tenant_id.")

        resolved_request_id = (request_id or f"request:{user_id.strip()}").strip()
        if not resolved_request_id:
            raise ValueError("UMAMemory retrieval requires a non-empty request_id.")

        return RuntimeContext(
            tenant_id=resolved_tenant_id,
            agent_id=resolved_agent_id,
            request_id=resolved_request_id,
            user_id=user_id.strip(),
            workspace_id=workspace_id,
            session_id=session_id,
        )

    # ----------------------------------------------------------------------
    # -------------------- Core Public APIs -------------------------------
    # ----------------------------------------------------------------------
    def set_context(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: str = "default",
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "UMAMemory":
        runtime_context = self._build_runtime_context(
            user_id=user_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            request_id=request_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        self._bound_runtime_context = runtime_context
        self._agent_id = runtime_context.agent_id
        logger.debug(
            "UMAMemory.for_context: bound tenant=%s agent=%s user=%s request=%s session=%s",
            runtime_context.tenant_id,
            runtime_context.agent_id,
            runtime_context.user_id,
            runtime_context.request_id,
            runtime_context.session_id,
        )
        return self
    
    # ----------------------------------------------------------------------
    # Core API: Retrieval
    # ----------------------------------------------------------------------
    async def retrieve_context(
        self,
        *,
        query_text: str,
    ) -> Dict[str, list]:
        """Return curated LLM context for the bound runtime scope.

        Contract:
        - intended for LLM context assembly, not durable memory projection
        - returns the canonical multi-lane retrieval bundle
        - persisted artifacts expose canonical UMA metadata through `meta`
        - lane contents remain source-traceable through facts, chunks, skills, and graph items
        """
        runtime_context = self._require_bound_runtime_context()
        return await self.runtime.retrieve_context(
            runtime_context,
            query_text=query_text,
        )

    async def retrieve_memory(
        self,
        *,
        query_text: str,
    ) -> Dict[str, Any]:
        """Return durable, evidence-backed memory artifacts for the bound runtime scope.

        Contract:
        - `facts`: memory facts relevant to the query
        - `skills`: procedural memory artifacts relevant to the query
        - `evidence`: supporting chunk objects backing the memory result
        - `artifacts`: document-level groupings of supporting evidence for continuity-oriented use
        - `confidence`: retrieval confidence metadata carried from the underlying retrieval pass
        - evidence and grouped artifacts expose canonical UMA metadata instead of inferred lanes
        """
        runtime_context = self._require_bound_runtime_context()
        return await self.runtime.retrieve_memory(
            runtime_context,
            query_text=query_text,
        )

    async def _retrieve_context_for_context(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
    ) -> Dict[str, list]:
        return await self.runtime.retrieve_context(
            runtime_context,
            query_text=query_text,
        )

    async def _retrieve_memory_for_context(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
    ) -> Dict[str, Any]:
        return await self.runtime.retrieve_memory(
            runtime_context,
            query_text=query_text,
        )

    async def _render_context_for_context(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
    ) -> str:
        return await self.runtime.render_context(
            runtime_context,
            query_text=query_text,
        )

    async def _retrieve_structured_context_for_context(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
    ) -> Dict[str, list]:
        """Compatibility shim for pre-cleanup context callers."""
        return await self._retrieve_context_for_context(
            runtime_context,
            query_text=query_text,
        )

    async def _retrieve_rendered_context_for_context(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
    ) -> str:
        """Compatibility shim for pre-cleanup rendered callers."""
        return await self._render_context_for_context(
            runtime_context,
            query_text=query_text,
        )

    async def _get_context_messages_for_context(
        self,
        runtime_context: RuntimeContext,
        *,
        query_text: str,
        render_mode: str = "animus_v1",
    ) -> Dict[str, Any]:
        return await self.runtime.get_context_messages(
            runtime_context,
            query_text=query_text,
            render_mode=render_mode,
        )

    def _build_retrieval_request(self, runtime_context: RuntimeContext) -> Any:
        """Compatibility bridge to the canonical runtime request builder."""
        return self.runtime._build_retrieval_request(runtime_context)
    
    
    # ----------------------------------------------------------------------
    # Core API: Data Ingestion
    # ----------------------------------------------------------------------
    async def process_turn(
        self,
        *,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Public turn-ingest entrypoint.

        This is the supported memory-surface wrapper over the canonical
        pipeline turn processor. It uses the current runtime `agent_id` and the
        explicit `extra_meta` scope fields consumed by `MemoryPipeline`.
        """
        self._ensure_ingestion_ready()

        if self.pipeline is None:
            from uma.ingest.pipeline import MemoryPipeline

            self.pipeline = MemoryPipeline(
                memory_client=self,
                hooks=self.hooks,
                promotion_policy=self.promotion_policy,
            )
            logger.debug("UMAMemory.process_turn: MemoryPipeline initialized lazily.")

        normalized_user_id = normalize_user_id(user_id)
        await self.pipeline.process_turn(
            user_id=normalized_user_id,
            user_msg=user_msg,
            assistant_reply=assistant_reply,
            extra_meta=extra_meta,
        )

    async def sync_memory(
        self,
        *,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Backward-compatible alias for `process_turn(...)`."""
        await self.process_turn(
            user_id=user_id,
            user_msg=user_msg,
            assistant_reply=assistant_reply,
            extra_meta=extra_meta,
        )

    async def ingest_document(
        self,
        file_path: str,
        *,
        target_owner: Optional[TargetOwner] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        config: Optional[Any] = None,
    ) -> Any:
        """Ingest an unstructured document into UMA memory."""
        self._ensure_ingestion_ready()

        from uma.ingest.ingest_service import ingest_document as _ingest
        return await _ingest(
            file_path,
            target_owner=target_owner,
            owner_type=owner_type,
            owner_id=owner_id,
            config=config,
            memory=self,
        )

    def load_userprofile(self, path: str) -> "UMAMemory":
        """Load USER.md into the in-memory Animus profile cache."""
        self.animus_profile_provider.load_user_profile(path)
        return self

    def load_agentprofile(self, path: str) -> "UMAMemory":
        """Load SOUL.md into the in-memory Animus profile cache."""
        self.animus_profile_provider.load_agent_profile(path)
        return self

    async def load_memory_bootstrap(
        self,
        file_path: str,
        *,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Bootstrap long-term memory facts from an optional MEMORY.md file.

        Phase-1 behavior stays intentionally simple:
        - missing file -> skip
        - empty file -> skip
        - no importable lines -> skip
        - otherwise create one durable fact per extracted line

        The file is imported once per exact content hash. After import, UMA is
        the source of truth and normal memory consolidation can manage future
        duplicates at the memory layer.
        """
        del config  # Reserved for future bootstrap tuning.

        runtime_context = self._require_bound_runtime_context()
        normalized_user_id = runtime_context.user_id
        normalized_tenant_id = runtime_context.tenant_id
        workspace_id = runtime_context.workspace_id

        if not normalized_user_id:
            raise ValueError("UMAMemory.load_memory_bootstrap: bound runtime_context.user_id is required")

        normalized_path = _validate_bootstrap_file_path(
            file_path,
            api_name="load_memory_bootstrap",
        )

        if not os.path.exists(normalized_path):
            logger.info(
                "UMAMemory.load_memory_bootstrap: skipping missing memory bootstrap file path=%s user_id=%s",
                normalized_path,
                normalized_user_id,
            )
            return _build_bootstrap_skip_result(
                reason="missing_file",
                path=normalized_path,
                user_id=normalized_user_id,
                tenant_id=normalized_tenant_id,
            )

        raw_text = _read_bootstrap_file(
            normalized_path,
            api_name="load_memory_bootstrap",
        )

        if not raw_text.strip():
            logger.info(
                "UMAMemory.load_memory_bootstrap: skipping empty memory bootstrap file path=%s user_id=%s",
                normalized_path,
                normalized_user_id,
            )
            return _build_bootstrap_skip_result(
                reason="empty_file",
                path=normalized_path,
                user_id=normalized_user_id,
                tenant_id=normalized_tenant_id,
            )

        entries = _extract_memory_bootstrap_lines(raw_text)
        if not entries:
            logger.info(
                "UMAMemory.load_memory_bootstrap: skipping memory bootstrap without importable lines path=%s user_id=%s",
                normalized_path,
                normalized_user_id,
            )
            return _build_bootstrap_skip_result(
                reason="no_entries",
                path=normalized_path,
                user_id=normalized_user_id,
                tenant_id=normalized_tenant_id,
            )

        ingest_signature = _build_memory_bootstrap_signature(raw_text=raw_text)
        source_hash = ingest_signature["content_hash"]

        if await _is_bootstrap_manifest_duplicate(
            document_store=self.document_store,
            owner_type="user",
            owner_id=normalized_user_id,
            source_hash=source_hash,
            ingest_signature=ingest_signature,
            api_name="load_memory_bootstrap",
        ):
            logger.info(
                "UMAMemory.load_memory_bootstrap: skipping idempotent re-import path=%s user_id=%s",
                normalized_path,
                normalized_user_id,
            )
            return _build_bootstrap_skip_result(
                reason="idempotent",
                path=normalized_path,
                user_id=normalized_user_id,
                tenant_id=normalized_tenant_id,
                extra={"entries_found": len(entries)},
            )

        self._ensure_ingestion_ready()

        semantic_store = self._stores.get("semantic")
        if semantic_store is None or not hasattr(semantic_store, "upsert_fact"):
            raise RuntimeError(
                "UMAMemory.load_memory_bootstrap: semantic store is not initialized or does not support upsert_fact"
            )
        if self.embedder is None:
            raise RuntimeError("UMAMemory.load_memory_bootstrap: embedder is not initialized")

        logger.info(
            "UMAMemory.load_memory_bootstrap: importing memory bootstrap path=%s tenant_id=%s user_id=%s entries=%d",
            normalized_path,
            normalized_tenant_id,
            normalized_user_id,
            len(entries),
        )

        now = datetime.now(timezone.utc)
        facts: list[Fact] = []
        for index, entry_text in enumerate(entries):
            fact_hash = hashlib.sha256(
                f"{normalized_tenant_id}|{normalized_user_id}|{entry_text}".encode("utf-8")
            ).hexdigest()[:24]
            fact = Fact(
                id=f"fact_mem_{fact_hash}",
                subject=normalized_user_id,
                predicate="remembers",
                object=entry_text,
                created_at=now,
                updated_at=now,
                source_ids=[f"memory_bootstrap:{source_hash}"],
                confidence=1.0,
                meta=normalize_fact_metadata(
                    {
                        "source_kind": "memory_bootstrap",
                        "source_file": normalized_path,
                        "import_mode": "bootstrap",
                        "line_index": index,
                        "source_type": "memory_bootstrap",
                    },
                    fact_id=f"fact_mem_{fact_hash}",
                    owner_type="user",
                    owner_id=normalized_user_id,
                    created_at=now,
                    updated_at=now,
                    source_ids=[f"memory_bootstrap:{source_hash}"],
                    session_id=runtime_context.session_id,
                ),
                owner_type="user",
                owner_id=normalized_user_id,
                tenant_id=normalized_tenant_id,
                workspace_id=workspace_id,
                session_id=runtime_context.session_id,
                origin_agent_id=runtime_context.agent_id,
                origin_user_id=runtime_context.user_id,
                origin_session_id=runtime_context.session_id,
                scope_model_version="v2",
                salience=1.0,
            )
            fact.validate()
            facts.append(fact)

        embed_texts = [build_fact_embedding_text(fact) for fact in facts]
        try:
            vectors = await self.embedder.embed(embed_texts)
        except Exception as exc:
            logger.exception("UMAMemory.load_memory_bootstrap: embedding failed path=%s", normalized_path)
            raise RuntimeError(
                f"UMAMemory.load_memory_bootstrap: embedding failed for file: {normalized_path}"
            ) from exc

        if not isinstance(vectors, list) or len(vectors) != len(facts):
            raise RuntimeError(
                "UMAMemory.load_memory_bootstrap: embedder returned an invalid number of vectors"
            )

        persisted_fact_ids: list[str] = []
        for fact, vector in zip(facts, vectors):
            try:
                await semantic_store.upsert_fact(fact, vector)
                persisted_fact_ids.append(fact.id)
            except Exception:
                logger.exception(
                    "UMAMemory.load_memory_bootstrap: failed to persist fact fact_id=%s path=%s",
                    fact.id,
                    normalized_path,
                )

        if not persisted_fact_ids:
            raise RuntimeError(
                f"UMAMemory.load_memory_bootstrap: failed to persist any facts for file: {normalized_path}"
            )

        await _persist_bootstrap_manifest(
            document_store=self.document_store,
            doc_id=f"memory-bootstrap:{source_hash}",
            source_path=normalized_path,
            source_hash=source_hash,
            runtime_context=runtime_context,
            meta=normalize_document_metadata(
                {
                    "source_kind": "memory_bootstrap",
                    "import_mode": "bootstrap",
                    "entries_found": len(entries),
                    "facts_created": len(persisted_fact_ids),
                    "ingest_signature": ingest_signature,
                    "source_type": "memory_bootstrap",
                },
                doc_id=f"memory-bootstrap:{source_hash}",
                owner_type="user",
                owner_id=normalized_user_id,
                ingested_at=now,
                source_path=normalized_path,
                source_hash=source_hash,
            ),
            api_name="load_memory_bootstrap",
        )

        return {
            "status": "ingested",
            "path": normalized_path,
            "tenant_id": normalized_tenant_id,
            "user_id": normalized_user_id,
            "workspace_id": workspace_id,
            "entries_found": len(entries),
            "facts_created": len(persisted_fact_ids),
            "fact_ids": persisted_fact_ids,
        }

    async def load_daily_diary_bootstrap(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Bootstrap a daily OpenClaw diary file into UMA episodic memory.

        Phase-1 behavior:
        - one-time import only
        - each markdown bullet becomes one episode
        - if the file does not exist, skip cleanly
        - if the file exists but has no bullet entries, skip cleanly

        After import, UMA becomes the source of truth.
        """
        runtime_context = self._require_bound_runtime_context()

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("UMAMemory.load_daily_diary_bootstrap: file_path must be a non-empty string")

        normalized_path = os.path.abspath(file_path.strip())
        normalized_user_id = runtime_context.user_id
        normalized_tenant_id = runtime_context.tenant_id
        workspace_id = runtime_context.workspace_id

        if not normalized_user_id:
            raise ValueError(
                "UMAMemory.load_daily_diary_bootstrap: bound runtime_context.user_id is required"
            )

        if not os.path.exists(normalized_path):
            logger.info(
                "UMAMemory.load_daily_diary_bootstrap: skipping missing diary file path=%s user_id=%s",
                normalized_path,
                normalized_user_id,
            )
            return {
                "status": "skipped",
                "reason": "missing_file",
                "path": normalized_path,
                "user_id": normalized_user_id,
                "tenant_id": normalized_tenant_id,
            }

        if not os.path.isfile(normalized_path):
            raise ValueError(
                f"UMAMemory.load_daily_diary_bootstrap: path is not a file: {normalized_path}"
            )

        try:
            with open(normalized_path, "r", encoding="utf-8") as handle:
                raw_text = handle.read()
        except Exception as exc:
            logger.exception(
                "UMAMemory.load_daily_diary_bootstrap: failed to read diary file path=%s",
                normalized_path,
            )
            raise RuntimeError(
                f"UMAMemory.load_daily_diary_bootstrap: failed to read file: {normalized_path}"
            ) from exc

        if not raw_text.strip():
            logger.info(
                "UMAMemory.load_daily_diary_bootstrap: skipping empty diary file path=%s user_id=%s",
                normalized_path,
                normalized_user_id,
            )
            return {
                "status": "skipped",
                "reason": "empty_file",
                "path": normalized_path,
                "user_id": normalized_user_id,
                "tenant_id": normalized_tenant_id,
            }

        entries = self._extract_daily_diary_entries(raw_text)

        if not entries:
            logger.info(
                "UMAMemory.load_daily_diary_bootstrap: skipping diary without bullet entries path=%s user_id=%s",
                normalized_path,
                normalized_user_id,
            )
            return {
                "status": "skipped",
                "reason": "no_entries",
                "path": normalized_path,
                "user_id": normalized_user_id,
                "tenant_id": normalized_tenant_id,
            }

        ingest_signature = self._build_diary_bootstrap_signature(raw_text=raw_text)
        source_hash = ingest_signature["content_hash"]

        existing_manifest = None
        try:
            if self.document_store is not None and hasattr(self.document_store, "get_by_owner_and_hash"):
                existing_manifest = await self.document_store.get_by_owner_and_hash(
                    owner_type="user",
                    owner_id=normalized_user_id,
                    source_hash=source_hash,
                )
        except Exception:
            existing_manifest = None
            logger.exception(
                "UMAMemory.load_daily_diary_bootstrap: manifest lookup failed; continuing with import"
            )

        if existing_manifest is not None:
            existing_sig = (getattr(existing_manifest, "meta", None) or {}).get("ingest_signature") or {}
            if existing_sig == ingest_signature:
                logger.info(
                    "UMAMemory.load_daily_diary_bootstrap: skipping idempotent re-import path=%s user_id=%s",
                    normalized_path,
                    normalized_user_id,
                )
                return {
                    "status": "skipped",
                    "reason": "idempotent",
                    "path": normalized_path,
                    "user_id": normalized_user_id,
                    "tenant_id": normalized_tenant_id,
                    "entries_found": len(entries),
                }

        diary_date = None
        try:
            diary_date = Path(normalized_path).stem
        except Exception:
            diary_date = None

        self._ensure_ingestion_ready()

        from uma.ingest.episodic_writer import write_daily_diary_episodes

        logger.info(
            "UMAMemory.load_daily_diary_bootstrap: importing diary path=%s tenant_id=%s user_id=%s entries=%d",
            normalized_path,
            normalized_tenant_id,
            normalized_user_id,
            len(entries),
        )

        episode_ids = await write_daily_diary_episodes(
            file_path=normalized_path,
            diary_date=diary_date,
            entries=entries,
            owner_type="user",
            owner_id=normalized_user_id,
            user_id=normalized_user_id,
            embedder=self.embedder,
            episodic_core=self.episodic_core,
        )

        if episode_ids and self.document_store is not None and hasattr(self.document_store, "upsert_document"):
            try:
                now = Path(normalized_path).stat().st_mtime if os.path.exists(normalized_path) else None
                if now is not None:
                    ingested_at = datetime.fromtimestamp(now, tz=timezone.utc)
                else:
                    ingested_at = datetime.now(timezone.utc)

                await self.document_store.upsert_document(
                    DocumentRecord(
                        doc_id=f"daily-diary:{source_hash}",
                        source_path=normalized_path,
                        source_hash=source_hash,
                        ingested_at=ingested_at,
                        tenant_id=normalized_tenant_id,
                        owner_type="user",
                        owner_id=normalized_user_id,
                        workspace_id=workspace_id,
                        meta=normalize_document_metadata(
                            {
                                "source_kind": "daily_diary",
                                "diary_date": diary_date,
                                "import_mode": "bootstrap",
                                "entries_found": len(entries),
                                "episodes_created": len(episode_ids),
                                "ingest_signature": ingest_signature,
                                "source_type": "daily_diary",
                            },
                            doc_id=f"daily-diary:{source_hash}",
                            owner_type="user",
                            owner_id=normalized_user_id,
                            ingested_at=ingested_at,
                            source_path=normalized_path,
                            source_hash=source_hash,
                        ),
                    )
                )
            except Exception:
                logger.exception(
                    "UMAMemory.load_daily_diary_bootstrap: failed to persist diary manifest path=%s",
                    normalized_path,
                )

        return {
            "status": "ingested",
            "path": normalized_path,
            "tenant_id": normalized_tenant_id,
            "user_id": normalized_user_id,
            "workspace_id": workspace_id,
            "diary_date": diary_date,
            "entries_found": len(entries),
            "episodes_created": len(episode_ids),
            "episode_ids": episode_ids,
        }

    # ----------------------------------------------------------------------
    # Core API: Health and maintenance
    # ----------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Run basic dependency readiness checks."""
        if not self.initialized:
            return {
                "status": "error",
                "checks": {
                    "memory": {
                        "name": "memory",
                        "status": "error",
                        "detail": "UMAMemory not initialized",
                        "latency_ms": None,
                    }
                },
            }

        from uma.common.health import run_health_checks

        return run_health_checks(self)

    async def rebuild_vector_indexes(
        self,
        *,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Rebuild vector indexes from SQL-backed data."""
        from uma.common.maintenance import rebuild_vector_indexes

        return await rebuild_vector_indexes(
            self,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            include_procedural=include_procedural,
            batch_size=batch_size,
        )

    async def rebuild_derived_indexes(
        self,
        *,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        include_graph: bool = True,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Rebuild derived vector and graph indexes from authoritative SQL-backed data."""
        from uma.common.maintenance import rebuild_derived_indexes

        return await rebuild_derived_indexes(
            self,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            include_procedural=include_procedural,
            include_graph=include_graph,
            batch_size=batch_size,
        )

    # ----------------------------------------------------------------------
    # Core API: Shutdown
    # ----------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean up backend resources."""
        if self.graph_core:
            try:
                self.graph_core.close()
            except Exception:
                logger.exception("Error shutting down GraphCore.")


    # ----------------------------------------------------------------------
    # Core subsystem initialization
    # ----------------------------------------------------------------------

    def _init_core_subsystems(self) -> None:
        """Initialize core UMA subsystems that depend on LLM/embedder/stores."""
        if self.llm is None or self.embedder is None:
            raise RuntimeError(
                "UMAMemory._init_core_subsystems: LLM and embedder must "
                "be initialized before core subsystems."
            )

        if not self._stores:
            raise RuntimeError(
                "UMAMemory._init_core_subsystems: stores must be initialized "
                "before core subsystems."
            )

        # ---------------------- Working Memory Core ----------------------
        try:
            self.working_memory = WorkingMemoryCore(
                llm=self.llm,
                memory_client=self,
            )
            logger.debug("WorkingMemoryCore initialized.")
        except Exception:
            logger.exception("UMAMemory: failed to initialize WorkingMemoryCore.")
            raise

        # ---------------------- Episodic Core ---------------------------
        try:
            indexer = EpisodeIndexer(
                llm=self.llm,
                embedder=self.embedder,
            )

            retention = EpisodicRetentionPolicy(
                max_episodes=self.consolidation_cfg.max_episodes_per_cycle
            )

            self.episodic_core = EpisodicCore(
                episodic_store=self._stores["episodic"],
                episode_indexer=indexer,
                retention_policy=retention,
            )
            logger.debug("EpisodicCore initialized.")
        except Exception:
            logger.exception("UMAMemory: failed to initialize EpisodicCore.")
            raise

        # ---------------------- Semantic Core ---------------------------
        try:
            salience = self.semantic_salience_threshold
            self.semantic_core = SemanticCore(
                llm=self.llm,
                embedder=self.embedder,
                semantic_store=self._stores["semantic"],
                salience_threshold=salience,
                memory=self,
            )
            logger.info(
                "SemanticCore initialized (salience_threshold=%.2f).",
                salience,
            )
        except Exception:
            logger.exception("UMAMemory: failed to initialize SemanticCore.")
            raise

        # ---------------------- Procedural Core -------------------------
        try:
            self.procedural_core = ProceduralCore(self._stores["procedural"])
            logger.debug("ProceduralCore initialized.")
        except Exception:
            logger.exception("UMAMemory: failed to initialize ProceduralCore.")
            raise

        # ---------------------- Chunk Core ------------------------------
        try:
            self.chunk_core = ChunkCore(self._stores["chunk"], memory=self)
            logger.debug("ChunkCore initialized.")
        except Exception:
            logger.exception("UMAMemory: failed to initialize ChunkCore.")
            raise

    # ----------------------------------------------------------------------
    # Graph subsystem initialization
    # ----------------------------------------------------------------------

    def _init_graph_core(self) -> None:
        """Initialize the graph subsystem using the unified storage config."""
        storage_cfg = self.raw_config.storage
        backend = storage_cfg.graph_backend

        # --------------------------------------------------------------
        # 1) Disabled backend → skip cleanly
        # --------------------------------------------------------------
        if backend == "disabled":
            self.graph_core = None
            logger.info("Graph subsystem disabled via config.storage.graph_backend.")
            return

        # --------------------------------------------------------------
        # 2) Load graph config block
        # --------------------------------------------------------------
        if "graph_config" not in storage_cfg:
            raise ValueError(
                "Missing required 'storage.graph_config' section in config when "
                f"graph_backend is '{backend}'."
            )

        graph_cfg = storage_cfg.get("graph_config") or {}

        # --------------------------------------------------------------
        # 3) Backend selection
        # --------------------------------------------------------------
        if backend in {"neo4j", "memgraph"}:
            raise ValueError(
                "Graph backends are now loaded via extensions. "
                "Set storage.graph_backend to a plugin spec 'module:callable'."
            )

        if ":" not in str(backend):
            raise ValueError(
                f"Unsupported storage.graph_backend={backend!r}. "
                "Expected: 'disabled' or plugin spec 'module:callable'."
            )

        if not isinstance(graph_cfg, dict):
            raise ValueError("'storage.graph_config' must be a mapping for plugin graph backends")

        try:
            adapter_factory = parse_plugin_spec(backend)
            if not callable(adapter_factory):
                raise TypeError("storage.graph_backend plugin must be a callable 'module:attr'")
            adapter = adapter_factory(**graph_cfg)
        except Exception as exc:
            logger.exception("Graph adapter initialization failed.")
            raise RuntimeError(
                "Failed to initialize graph adapter. "
                "Verify plugin path and configuration."
            ) from exc

        # --------------------------------------------------------------
        # 4) Connect adapter to TemporalGraphCore
        # --------------------------------------------------------------
        try:
            self.graph_core = TemporalGraphCore(adapter)
            logger.info(
                "TemporalGraphCore initialized (backend=%s, uri=%s).",
                backend,
                graph_cfg.get("uri"),
            )
        except Exception as exc:
            logger.exception("Failed to initialize TemporalGraphCore.")
            raise RuntimeError(
                "Graph core initialization failed. "
                "Verify graph adapter dependencies and configuration."
            ) from exc

    # ----------------------------------------------------------------------
    # Optional Features
    # ----------------------------------------------------------------------

    def _init_optional_features(self) -> None:
        """Attach optional UMA features from config."""
        policy_cfg = self.features_cfg.policy or {}
        self._feature_policy = FeaturePolicy(
            on_attach_error=str(policy_cfg.get("on_attach_error", "log_and_skip")),
            allow_method_override=bool(policy_cfg.get("allow_method_override", False)),
        )

        registry = default_feature_registry()
        registry.register_entry_points()
        loader = FeatureLoader(registry, self._feature_policy)

        services = {
            "procedural_core": self.procedural_core,
            "episodic_core": self.episodic_core,
            "semantic_core": self.semantic_core,
            "llm": self.llm,
            "embedder": self.embedder,
            "hooks": self.hooks,
            "graph_core": self.graph_core,
            "cluster_similarity": self.consolidation_cfg.cluster_similarity,
            "max_episodes_per_cycle": self.consolidation_cfg.max_episodes_per_cycle,
        }

        loader.load_from_config(
            memory_client=self,
            feature_cfgs=self.features_cfg.load,
            services=services,
        )

    def register_methods(
        self,
        feature_name: str,
        methods: Dict[str, Any],
        allow_override: Optional[bool] = None,
    ) -> None:
        """Attach feature methods to UMAMemory with collision checks."""
        allow_override = (
            self._feature_policy.allow_method_override
            if allow_override is None
            else allow_override
        )
        for name, func in methods.items():
            if not allow_override and hasattr(self, name):
                raise ValueError(
                    f"Feature '{feature_name}' attempted to override '{name}'"
                )
            setattr(self, name, func)

   
