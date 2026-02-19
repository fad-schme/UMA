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
    • RetrievalService (developer-facing recall API)

UMA does **not** generate assistant replies and does **not** perform
agent reasoning. Developers bring their own LLM or agent loop and use
UMA strictly for memory management.

Typical developer workflow
--------------------------

1. Initialize UMA:

    memory = UMAMemory.from_yaml("uma.yaml")

2. After the developer's agent produces a reply, store the memory turn:

    await memory.process_turn(
        user_id=user_id,
        user_msg=user_message,
        assistant_reply=final_agent_reply,
    )

   This uses an internal ingestion pipeline for:
        - working memory update + compaction
        - episodic memory generation
        - semantic fact ingestion
        - graph updates
        - lifecycle hooks

3. When constructing a prompt for the agent, fetch UMA context:

    ctx = await memory.get_structured_context(
        user_id=user_id,
        query_text="the user's current question or task"
    )

   The context dictionary includes:

        {
            "working_memory": [...],   # always included
            "episodic": [...],         # retrieval-matched episodes
            "facts": [...],            # relevant long-term facts
            "skills": [...],           # relevant skills
            "graph": [...],            # relevant graph items
        }

4. Developers decide how to inject this memory into their agent prompts.
   UMA never performs reasoning or constructs prompts on its own.

Internal retrieval components
-----------------------------
    The following attributes are internal and initialized lazily on first use:

    • memory_env
        An instance of UMAMemoryEnvironment.
        This provides a safe, read-only retrieval environment
        used exclusively by the RLMController.

    • rlm_controller
        An optional RLMController instance.
        When enabled via configuration, it performs bounded, recursive
        memory retrieval using the configured LLM as a control model.
        When disabled or unavailable, UMA falls back to classic retrieval.

    These components:
    • Are not part of the public API
    • Never mutate memory
    • Never perform agent reasoning
    • Are used only by `get_structured_context()`

Design Philosophy
-----------------
- UMA provides *memory operations*, not agent behaviors.
- All reasoning, tool use, and final reply generation happen outside UMA.
- UMA focuses on correctness, retrieval quality, summarization, and
  long-term memory structure.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, Optional

from .memory_config import UMAConfig  # YAML loader + validation (dict-like)
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
from .retrieval.service import RetrievalService
from .graph import TemporalGraphCore

# Stores
from .utils.config_types import parse_plugin_spec
from .initializers.runtime import init_retrieval_ready, init_ingestion_ready, schedule_ingestion_warmup

# Optional Features
from .utils.registry import FeatureLoader, FeaturePolicy, default_feature_registry

logger = logging.getLogger(__name__)


class UMAMemory:
    """
    UMA Memory Runtime Container.

    This class:
        - Initializes all core subsystems lazily
        - Provides public APIs for pipeline and retrieval
        - Loads optional features via direct attachment

    Developers ONLY interact with:
        memory = UMAMemory.from_yaml("uma.yaml")

        # after their own agent produces a reply:
        await memory.process_turn(user_id=user_id, user_msg=user_msg, assistant_reply=assistant_reply)

        # when building prompts:
        ctx = await memory.get_structured_context(user_id, query_text)

    All other subsystems remain internal.

    The configuration is YAML-driven and MUST contain at top-level:
        storage           – {db_root, sql_backend, vector_backend, graph_backend}
        working_memory    – WM token window + thresholds
        embedding         – provider, model, dimension
        llm               – UMA internal LLM provider + model
        retrieval         – caps for episodes/facts/skills/graph items
        consolidation     – cluster_similarity, max_episodes_per_cycle,
                            prune_min_fact_salience
        features          – {load, policy}
        graph             – (only required when storage.graph_backend != "disabled")
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "UMAMemory":
        cfg = UMAConfig.load_yaml(path)
        mem = cls(cfg, config_path=path)

        # Predictable startup cost: make retrieval instant thereafter.
        init_retrieval_ready(mem)
        mem._retrieval_ready = True
        mem.initialized = True

        # Background warmup: cores/features/pipeline (ingestion readiness).
        schedule_ingestion_warmup(mem)
        # Ingestion readiness is best-effort in warmup; enforced on first ingestion API call.

        return mem

    def __init__(self, config: UMAConfig, config_path: Optional[str] = None) -> None:
        """
        Convert nested config dicts into typed dataclasses where appropriate.

        Notes
        -----
        - Episodic / semantic / procedural stores no longer have dedicated
          config sections; all DB paths and vector settings are derived from:
              storage.db_root
              storage.sql_backend
              storage.vector_backend
              embedding.dimension
        - This keeps the config surface minimal for developers.
        """
        self.raw_config = config  # keep original for debugging
        self._config_path = config_path or getattr(config, "_source_path", None)
        self._config_dir = getattr(config, "_source_dir", None)
        self.initialized: bool = False
        self._features_initialized: bool = False
        self._retrieval_ready: bool = False
        self._ingestion_ready: bool = False
        self._warmup_scheduled: bool = False

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
        self.retrieval_service: Optional[RetrievalService] = None
        self.graph_core: Optional[TemporalGraphCore] = None
        self.procedural_core: Optional[ProceduralCore] = None
        self.chunk_core: Optional[ChunkCore] = None

        logger.debug("UMAMemory instance created with unified storage config.")

    # ----------------------------------------------------------------------
    # Internal lazy initialization (developer should NOT call initialization)
    # ----------------------------------------------------------------------

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value: Optional[str]) -> None:
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("UMAMemory.agent_id must be a non-empty string or None.")
            value = value.strip()
        self._agent_id = value
        # Keep RLM environment in sync with the canonical agent_id.
        if getattr(self, "memory_env", None) is not None:
            self.memory_env._agent_id = value

    def _ensure_base_ready(self) -> None:
        """
        Minimal baseline readiness:
        - stores only

        This must be cheap and safe to call from any public API.
        """
        # In the new model, from_yaml() makes retrieval ready immediately.
        # Base readiness is therefore equivalent to ensuring retrieval is ready.
        if not getattr(self, "_retrieval_ready", False):
            init_retrieval_ready(self)
        self.initialized = True


    def _ensure_retrieval_ready(self) -> None:
        """
        Retrieval readiness (automatic lazy init).

        RLM is the default retrieval mode when enabled AND LLM is available.
        Classic retrieval is ALWAYS wired as fallback and used when LLM/RLM unavailable.

        Must remain lean:
        - stores + embedder + RetrievalService (+ graph if enabled)
        - MUST NOT initialize ingestion cores/pipeline/features
        """
        # Retrieval must be ready immediately after from_yaml().
        # Keep a defensive guard for non-standard construction paths.
        if not getattr(self, "_retrieval_ready", False):
            init_retrieval_ready(self)
        self.initialized = True

    def _ensure_ingestion_ready(self) -> None:
        """
        Ingestion readiness (automatic lazy init).

        Heavy path:
        - stores
        - LLM (required)
        - embedder
        - cores
        - optional features
        - pipeline
        """
        # Ingestion is warmed up in the background, but must self-heal on first ingestion call.
        if not getattr(self, "_ingestion_ready", False):
            init_ingestion_ready(self)
        self.initialized = True

    # ----------------------------------------------------------------------
    # Core Subsystems (WM, Episodic, Semantic, Procedural, Chunk — Retrieval wired lazily)
    # ----------------------------------------------------------------------

    def _init_core_subsystems(self) -> None:
        """
        Initialize core UMA subsystems that depend on:
            - LLM + embedder
            - SQL + vector stores

        Subsystems initialized here:
            - WorkingMemoryCore
            - EpisodicCore
            - SemanticCore
            - ProceduralCore
            - ChunkCore
        """
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
            sal = self.semantic_salience_threshold
            self.semantic_core = SemanticCore(
                llm=self.llm,
                embedder=self.embedder,
                semantic_store=self._stores["semantic"],
                salience_threshold=sal,
            )
            logger.info(
                "SemanticCore initialized (salience_threshold=%.2f).",
                sal,
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
    # Graph Subsystem Initialization (Unified Backend Selection)
    # ----------------------------------------------------------------------
    def _init_graph_core(self) -> None:
        """
        Initialize the graph subsystem using the unified storage config.

        Uses:
            storage.graph_backend: plugin spec "module:callable" | "disabled"
            storage.graph_config: connection details for plugin adapter

        Rules:
        -------
        • If graph_backend="disabled" → skip cleanly.
        • If plugin spec → load adapter factory and pass graph_config.
        • Never swallow connection failures silently.
        """

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

    def health_check(self) -> Dict[str, Any]:
        """
        Run basic dependency readiness checks.

        Returns a dict with overall status and per-dependency details.
        """
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
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Rebuild vector indexes from SQL-backed data.

        This is a recovery utility and should be used in maintenance jobs.
        """
        from .utils.maintenance import rebuild_vector_indexes

        return await rebuild_vector_indexes(
            self,
            owner_type=owner_type,
            owner_id=owner_id,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            include_procedural=include_procedural,
            batch_size=batch_size,
        )
    
    # ----------------------------------------------------------------------
    # PUBLIC DEVELOPER API — Unified User Context (WM + LT Retrieval)
    # ----------------------------------------------------------------------
    async def get_structured_context(self, user_id: str, query_text: str) -> Dict[str, list]:
        """
        Return a unified, developer-facing context pack for retrieval-augmented agents.

        UMA is a memory SDK:
        - No assistant reply generation
        - No prompt building
        - Retrieval only

        Retrieval path
        --------------
        - Always includes stored Working Memory (WM).
        - Long-term retrieval is performed using:
            1) RLMController (recursive retrieval) if enabled, else
            2) RetrievalService (classic retrieval).

        Returns
        -------
        Dict[str, list]
            {
                "working_memory": [...],
                "episodic": [...],
            "facts": [...],
            "skills": [...],
                "graph": [...],
            }
        """
        from ..adapters.observability.metrics import increment, timed

        self._ensure_retrieval_ready()
        
        if not user_id or not isinstance(user_id, str):
            raise ValueError("UMAMemory.get_structured_context: user_id must be a non-empty string.")
        if not query_text or not isinstance(query_text, str):
            raise ValueError("UMAMemory.get_structured_context: query_text must be a non-empty string.")
        normalized_user_id = normalize_user_id(user_id)

        with timed("uma.get_structured_context.latency"):
            # 1) Stored WM
            try:
                wm_stored = (
                    self.working_memory.get_context(normalized_user_id)
                    if self.working_memory
                    else []
                )
            except Exception:
                logger.exception(
                    "UMAMemory.get_structured_context: failed to load WM user=%s",
                    normalized_user_id,
                )
                wm_stored = []

            # 2) Prefer RLM retrieval
            if getattr(self, "_rlm_controller", None) is not None:
                try:
                    pack = await self._rlm_controller.retrieve_context(
                        user_id=normalized_user_id,
                        query_text=query_text,
                    )
                    increment("uma.get_structured_context.calls", tags={"path": "rlm"})
                    coverage = getattr(pack, "coverage", None)
                    from .retrieval.rlm.coverage import compute_confidence
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
                except Exception:
                    logger.exception(
                        "UMAMemory.get_structured_context: RLM failed",
                        extra={"user": normalized_user_id},
                    )
                    if bool(getattr(self, "retrieval_cfg", None) and self.retrieval_cfg.strict):
                        raise
                    logger.warning(
                        "UMAMemory.get_structured_context: falling back to classic user=%s",
                        normalized_user_id,
                    )

            # 3) Classic fallback
            try:
                retrieved = await self.retrieval_service.retrieve(
                    user_id=normalized_user_id,
                    memory_type="all",
                    query_text_or_embedding=query_text,
                    agent_id=getattr(self, "agent_id", None),
                )
            except Exception:
                logger.exception(
                    "UMAMemory.get_structured_context: classic retrieval failed user=%s",
                    normalized_user_id,
                )
                retrieved = {}

            increment("uma.get_structured_context.calls", tags={"path": "classic"})
        return {
            "working_memory": wm_stored,
            "episodic": retrieved.get("episodes", []) or [],
            "facts": retrieved.get("facts", []) or [],
            "chunks": retrieved.get("chunks", []) or [],
            "skills": retrieved.get("skills", []) or [],
            "graph": retrieved.get("graph", []) or [],
            "trace": retrieved.get("trace", []) or [],
            "confidence": {},
        }

    async def process_turn(
        self,
        *,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
    ) -> None:
        """
        Ingest a full conversation turn into UMA memory.

        This is the primary ingestion API and wraps the internal pipeline.
        """
        self._ensure_ingestion_ready()

        if not getattr(self, "pipeline", None):
            raise RuntimeError("UMAMemory.process_turn: pipeline not initialized.")

        normalized_user_id = normalize_user_id(user_id)
        await self.pipeline.process_turn(
            user_id=normalized_user_id,
            user_msg=user_msg,
            assistant_reply=assistant_reply,
        )

    async def ingest_document(
        self,
        file_path: str,
        *,
        owner_type: str,
        owner_id: str,
        config: Optional[Any] = None,
    ) -> Any:
        """
        Ingest an unstructured document into UMA memory.
        """
        self._ensure_ingestion_ready()

        from .ingest.ingest_service import ingest_document as _ingest
        return await _ingest(
            file_path,
            owner_type=owner_type,
            owner_id=owner_id,
            config=config,
            memory=self,
        )

    async def get_rendered_context(self, user_id: str, query_text: str) -> str:
        """
        Render a production-ready snippet directly from RLM retrieval.
        This path is shared by the app and gold tests to avoid divergence.
        """
        from .utils.context_pack_builder import get_rendered_context

        return await get_rendered_context(self, user_id=user_id, query_text=query_text)


    # ----------------------------------------------------------------------
    # OPTIONAL UTILITIES — Structured CoT Memory Builder
    # ----------------------------------------------------------------------

    async def build_cot_memory(self, user_id: str, query_text: str) -> dict:
        """
        Build a structured chain-of-thought (CoT) knowledge scaffold from UMA.

        This is NOT an LLM-generated chain-of-thought.
        Instead, it is a deterministic, structured template created from:
            - semantic facts
            - episodic summaries
            - procedural skills
            - graph nodes

        Example:
            cot = await memory.build_cot_memory(user, query)

        Returns
        -------
        dict
            {
                "reasoning_goals": [...],
                "known_facts": [...],
                "relevant_events": [...],
                "available_skills": [...],
                "graph_context": [...],
                "planning_scaffold": [...],
            }
        """
        from .utils.cot_memory_builder import build_cot_memory

        return await build_cot_memory(self, user_id=user_id, query_text=query_text)

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

        #
