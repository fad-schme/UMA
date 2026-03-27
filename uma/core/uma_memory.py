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

    bound = memory.for_context(
        user_id=user_id,
        agent_id="agent-default",
        tenant_id="default",
        request_id="req-1",
        session_id="session-1",
    )

3. Retrieve context:

    snippet = await bound.retrieve_rendered_context(
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

import logging
import threading
from typing import Any, Dict, Optional

from .memory_config import UMAConfig
from .utils.config_types import RuntimeConfig
from .utils.hooks import UMAHooks
from .utils.identity import normalize_user_id
from .utils.logging_setup import logger as uma_logger  # noqa: F401 (init side-effect)
from .working_memory.core import WorkingMemoryCore
from .episodic.core import EpisodicCore
from .episodic.indexer import EpisodeIndexer
from .episodic.policies import EpisodicRetentionPolicy
from .semantic.core import SemanticCore
from .procedural.core import ProceduralCore
from .chunk.core import ChunkCore
from .graph import TemporalGraphCore

# Stores
from .utils.config_types import parse_plugin_spec
from .initializers.runtime import init_retrieval_ready, init_ingestion_ready, schedule_ingestion_warmup

# Optional Features
from .utils.registry import FeatureLoader, FeaturePolicy, default_feature_registry
from ..types import RuntimeContext, TargetOwner
from .runtime import UMABoundMemory, UMARuntime

logger = logging.getLogger(__name__)


class UMAMemory:
    """UMA Memory Runtime Container.

    This class:
        - initializes all core subsystems lazily
        - provides public APIs for retrieval, ingestion, and maintenance
        - loads optional features via direct attachment

    Developers should primarily interact with:

        memory = UMAMemory.from_yaml("uma.yaml")

        bound = memory.for_context(
            user_id=user_id,
            agent_id="agent-default",
            tenant_id="default",
            request_id="req-1",
        )

        snippet = await bound.retrieve_rendered_context(query_text)

        await memory.process_turn(
            user_id=user_id,
            user_msg=user_msg,
            assistant_reply=assistant_reply,
        )

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

        # Background warmup: cores/features/pipeline (ingestion readiness).
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
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @property
    def runtime(self) -> UMARuntime:
        """Return the shared internal runtime, refreshing lazy-owned services."""
        if self._runtime is None:
            self._runtime = UMARuntime.from_memory(self)
        else:
            self._runtime.refresh_from_memory()
        return self._runtime


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

    # ----------------------------------------------------------------------
    # Public retrieval API
    # ----------------------------------------------------------------------

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

    def for_context(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: str = "default",
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> UMABoundMemory:
        """Return an ergonomic bound facade for repeated retrieval calls.

        This keeps UMAMemory as the only public entrypoint while hiding
        UMARuntime and RuntimeContext from common usage.
        """
        context = self._build_runtime_context(
            user_id=user_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            request_id=request_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        logger.debug(
            "UMAMemory.for_context: bound tenant=%s agent=%s user=%s request=%s session=%s",
            context.tenant_id,
            context.agent_id,
            context.user_id,
            context.request_id,
            context.session_id,
        )
        return UMABoundMemory(memory=self, context=context)

    async def retrieve_structured_context(
        self,
        *,
        query_text: str,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: str = "default",
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, list]:
        """Retrieve structured memory context for one explicit request scope."""
        context = self._build_runtime_context(
            user_id=user_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            request_id=request_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        return await self.runtime.retrieve_structured_context(
            context,
            query_text=query_text,
        )

    async def retrieve_rendered_context(
        self,
        *,
        query_text: str,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: str = "default",
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Retrieve rendered memory context for one explicit request scope."""
        context = self._build_runtime_context(
            user_id=user_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            request_id=request_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        return await self.runtime.retrieve_rendered_context(
            context,
            query_text=query_text,
        )

    async def get_context_messages(
        self,
        *,
        query_text: str,
        user_id: str,
        agent_id: Optional[str] = None,
        tenant_id: str = "default",
        request_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        render_mode: str = "openclaw_v1",
    ) -> Dict[str, Any]:
        """Retrieve context formatted as prompt messages for one request scope."""
        context = self._build_runtime_context(
            user_id=user_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            request_id=request_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        return await self.runtime.get_context_messages(
            context,
            query_text=query_text,
            render_mode=render_mode,
        )


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

    # ----------------------------------------------------------------------
    # Health and maintenance
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

        from .utils.health import run_health_checks

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
        from .utils.maintenance import rebuild_vector_indexes

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
        from .utils.maintenance import rebuild_derived_indexes

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
    # Ingestion
    # ----------------------------------------------------------------------

    async def process_turn(
        self,
        *,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ingest a full conversation turn into UMA memory."""
        self._ensure_ingestion_ready()

        if not getattr(self, "pipeline", None):
            raise RuntimeError("UMAMemory.process_turn: pipeline not initialized.")

        normalized_user_id = normalize_user_id(user_id)
        await self.pipeline.process_turn(
            user_id=normalized_user_id,
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

        from .ingest.ingest_service import ingest_document as _ingest
        return await _ingest(
            file_path,
            target_owner=target_owner,
            owner_type=owner_type,
            owner_id=owner_id,
            config=config,
            memory=self,
        )

    # ----------------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean up backend resources."""
        if self.graph_core:
            try:
                self.graph_core.close()
            except Exception:
                logger.exception("Error shutting down GraphCore.")