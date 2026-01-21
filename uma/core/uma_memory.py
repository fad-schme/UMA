"""
UMAMemory — Core UMA Memory Runtime
=====================================

UMA is a **memory SDK**, not an autonomous agent.

This class owns and orchestrates all UMA memory subsystems:

    • Working Memory (short-term conversational state)
    • Episodic Memory (indexed event history)
    • Semantic Memory (facts, preferences, domain knowledge)
    • Procedural Memory (skills, routines)
    • Temporal Graph (optional knowledge graph)
    • RetrievalService (developer-facing recall API)

UMA does **not** generate assistant replies and does **not** perform
agent reasoning. Developers bring their own LLM or agent loop and use
UMA strictly for memory management.

Typical developer workflow
--------------------------

1. Initialize UMA:

    memory = UMAMemory.from_yaml("uma.yaml")
    memory.initialize()

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

    ctx = await memory.get_user_context(
        user_id=user_id,
        query_text="the user's current question or task"
    )

   The context dictionary includes:

        {
            "working_memory": [...],   # always included
            "episodic": [...],         # retrieval-matched episodes
            "semantic": [...],         # relevant long-term facts
            "procedural": [...],       # relevant skills
            "graph": [...],            # relevant graph items
        }

4. Developers decide how to inject this memory into their agent prompts.
   UMA never performs reasoning or constructs prompts on its own.

Internal retrieval components
-----------------------------
    The following attributes are internal and initialized during `initialize()`:

    • memory_env
        An instance of UMAMemoryEnvironment.
        This provides a safe, read-only abstraction over UMA memory stores
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
    • Are used only by `get_user_context()`

Design Philosophy
-----------------
- UMA provides *memory operations*, not agent behaviors.
- All reasoning, tool use, and final reply generation happen outside UMA.
- UMA focuses on correctness, retrieval quality, summarization, and
  long-term memory structure.
"""

from __future__ import annotations

import logging
import inspect
from typing import Any, Dict, Optional

from .memory_config import UMAConfig     # YAML loader + validation (dict-like)
from .utils.config_types import (
    LLMConfig,
    EmbeddingConfig,
    WorkingMemorySettings,
    RetrievalConfig,
    FeaturesConfig,
    ConsolidationConfig,
    parse_plugin_spec,
)

from .utils.hooks import UMAHooks
from .working_memory.core import WorkingMemoryCore
from .episodic.core import EpisodicCore
from .episodic.indexer import EpisodeIndexer
from .episodic.policies import EpisodicRetentionPolicy
from .semantic.core import SemanticCore
from .retrieval.service import RetrievalService
from .graph import TemporalGraphCore
from ..adapters.llm.base import EmbeddingInterface, LLMInterface
from ..adapters.llm.callable_adapter import CallableEmbedderAdapter, CallableLLMAdapter

# Stores
from ..stores.episodic_sql import EpisodicSQLStore
from ..stores.semantic_sql import SemanticSQLStore
from ..stores.procedural_sql import ProceduralSQLStore

# DB + Vector Adapters
from ..adapters.db.sqlite_adapter import SQLiteAdapter
from ..adapters.vector.faiss_adapter import FaissIndex
from ..adapters.graph.neo4j_adapter import Neo4jAdapter

# Optional Features
from .utils.registry import FeatureLoader, FeaturePolicy, default_feature_registry

logger = logging.getLogger(__name__)


class UMAMemory:
    """
    UMA Memory Runtime Container.

    This class:
        - Initializes all core subsystems
        - Provides public APIs for pipeline and retrieval
    - Loads optional features via direct attachment

    Developers ONLY interact with:
        memory = UMAMemory.from_yaml("uma.yaml")
        memory.initialize()

        # after their own agent produces a reply:
        await memory.process_turn(user_id=user_id, user_msg=user_msg, assistant_reply=assistant_reply)

        # when building prompts:
        ctx = await memory.get_user_context(user_id, query_text)

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
        """Load YAML config and construct UMAMemory."""
        cfg = UMAConfig.load_yaml(path)
        return cls(cfg)

    def __init__(self, config: UMAConfig) -> None:
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
        self.initialized: bool = False

        # RLM components (initialized later)
        self.memory_env = None
        self.rlm_controller = None

        # Typed config objects for subsystems that benefit from strong typing
        self.llm_cfg = LLMConfig.from_dict(config.llm)
        self.embedding_cfg = EmbeddingConfig.from_dict(config.embedding)
        self.working_memory_cfg = WorkingMemorySettings.from_dict(config.working_memory)
        self.retrieval_cfg = RetrievalConfig.from_dict(config.retrieval)
        self.features_cfg = FeaturesConfig.from_dict(config.features)
        self.consolidation_cfg = ConsolidationConfig.from_dict(config.consolidation)

        # Hooks + Feature attachment registry
        self.hooks = UMAHooks()
        self.features: Dict[str, Any] = {}
        self._feature_policy = FeaturePolicy()

        # Core runtime components (initialized later)
        self.llm: Any = None
        self.embedder: Any = None

        self.episodic_store: Optional[EpisodicSQLStore] = None
        self.semantic_store: Optional[SemanticSQLStore] = None
        self.procedural_store: Optional[ProceduralSQLStore] = None

        self.working_memory: Optional[WorkingMemoryCore] = None
        self.semantic_core: Optional[SemanticCore] = None
        self.episodic_core: Optional[EpisodicCore] = None
        self.retrieval_service: Optional[RetrievalService] = None
        self.graph_core: Optional[TemporalGraphCore] = None

        logger.info("UMAMemory instance created with unified storage config.")

    # ----------------------------------------------------------------------
    # Initialization Entry Point
    # ----------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize all REQUIRED UMA subsystems in a safe, idempotent way.

        RetrievalService is wired exactly once and never overwritten.
        """
        if self.initialized:
            logger.debug("UMAMemory.initialize(): already initialized — skipping.")
            return

        logger.info("Initializing UMA Memory Runtime...")

        # 1. Load LLM + embedder
        self._init_llm_and_embedder()

        # 2. Load stores (episodic + semantic + procedural)
        self._init_stores()

        # 3. Load core subsystems (WM, EpisodicCore, SemanticCore)
        self._init_core_subsystems()

        # 4. Initialize graph backend (optional)
        self._init_graph_core()

        # 5. Register optional features
        self._init_optional_features()

        # 6. Initialize pipeline for ingestion
        if getattr(self, "pipeline", None) is None:
            from .utils.pipeline import MemoryPipeline

            self.pipeline = MemoryPipeline(memory_client=self, hooks=self.hooks)

        # 7. Wire RetrievalService EXACTLY once
        if self.retrieval_service is None:
            try:
                self.retrieval_service = RetrievalService(
                    memory=self,
                    retr_cfg=self.retrieval_cfg,
                )
                logger.info("RetrievalService wired to UMAMemory.")
            except Exception:
                logger.exception("Failed to initialize RetrievalService.")
                raise
        else:
            logger.debug("RetrievalService already initialized — not overwriting.")

        # 8. Wire RLM Controller (optional, retrieval-side only)
        rlm_cfg = self.retrieval_cfg.rlm

        if rlm_cfg is not None and rlm_cfg.enabled:
            try:
                from .retrieval.rlm.environment import UMAMemoryEnvironment
                from .retrieval.rlm.controller import RLMController

                self.memory_env = UMAMemoryEnvironment(self)

                self.rlm_controller = RLMController(
                    llm=self.llm,  # SAME LLM, control role
                    env=self.memory_env,
                    max_steps=rlm_cfg.max_steps,
                    max_actions_per_step=rlm_cfg.max_actions_per_step,
                    max_items_per_type=rlm_cfg.max_items_per_type,
                    llm_max_tokens=rlm_cfg.llm_max_tokens,
                    timeout_s=rlm_cfg.timeout_s,
                    max_env_calls=rlm_cfg.max_env_calls,
                    max_return_chars=rlm_cfg.max_return_chars,
                )

                logger.info("RLMController enabled and wired.")

            except Exception:
                logger.exception(
                    "Failed to initialize RLMController; falling back to classic retrieval."
                )
                self.rlm_controller = None
        else:
            logger.info("RLMController disabled by config.")

        # Mark initialization complete
        self.initialized = True
        logger.info("UMA Memory Runtime initialized successfully.")

    # ----------------------------------------------------------------------
    # LLM + Embedder Initialization (Typed Configs)
    # ----------------------------------------------------------------------

    def _init_llm_and_embedder(self) -> None:
        """
        Initialize the LLM and embedding model based on typed dataclass configs:
            - self.llm_cfg   (LLMConfig)
            - self.embedding_cfg (EmbeddingConfig)

        This method replaces the old dynamic config loader and guarantees that
        UMAMemory always uses validated, structured configuration settings.
        """

        # ----------------------------- LLM -----------------------------
        if self.llm_cfg.provider == "openai":
            from uma.adapters.llm.openai_llm import OpenAILLM

            llm_kwargs = {**self.llm_cfg.config}
            llm_kwargs.pop("model", None)
            self.llm = OpenAILLM(
                model=self.llm_cfg.model,
                **llm_kwargs,
            )
            logger.info("Loaded OpenAI LLM (model=%s)", self.llm_cfg.model)

        elif self.llm_cfg.provider == "ollama":
            from uma.adapters.llm.ollama_llm import OllamaLLM

            model = self.llm_cfg.ollama_model or self.llm_cfg.model
            llm_kwargs = {**self.llm_cfg.config}
            llm_kwargs.pop("model", None)
            self.llm = OllamaLLM(model=model, **llm_kwargs)
            logger.info("Loaded Ollama LLM (model=%s)", model)

        else:
            llm_cls = parse_plugin_spec(self.llm_cfg.provider)
            llm_kwargs = {**self.llm_cfg.config}
            if self.llm_cfg.model and "model" not in llm_kwargs:
                llm_kwargs["model"] = self.llm_cfg.model
            if isinstance(llm_cls, LLMInterface):
                self.llm = llm_cls
            elif inspect.isclass(llm_cls):
                self.llm = llm_cls(**llm_kwargs)
            elif callable(llm_cls):
                self.llm = CallableLLMAdapter(
                    callable_fn=llm_cls,
                    name=self.llm_cfg.provider,
                    preflight=bool(llm_kwargs.pop("preflight", True)),
                    default_kwargs=llm_kwargs,
                )
            else:
                raise TypeError(f"Unsupported LLM provider type: {type(llm_cls)}")
            logger.info("Loaded custom LLM adapter (%s)", self.llm_cfg.provider)

        # --------------------------- EMBEDDER --------------------------
        if self.embedding_cfg.provider == "openai":
            from uma.adapters.llm.openai_embedding import OpenAIEmbedder

            embed_kwargs = {**self.embedding_cfg.config}
            embed_kwargs.pop("model", None)
            self.embedder = OpenAIEmbedder(
                model=self.embedding_cfg.model,
                dimension=self.embedding_cfg.dimension,
                **embed_kwargs,
            )
            logger.info("Loaded OpenAI embedder (model=%s)", self.embedding_cfg.model)

        elif self.embedding_cfg.provider == "ollama":
            from uma.adapters.llm.ollama_embedding import OllamaEmbedder

            if not (self.embedding_cfg.model and self.embedding_cfg.dimension):
                raise ValueError(
                    "For Ollama embedder: both 'model' and 'dimension' must be specified "
                    "in the embedding configuration."
                )

            embed_kwargs = {**self.embedding_cfg.config}
            embed_kwargs.pop("model", None)
            embed_kwargs.pop("dimension", None)
            if "mode" not in embed_kwargs:
                embed_kwargs["mode"] = "native"
            self.embedder = OllamaEmbedder(
                model=self.embedding_cfg.model,
                dimension=self.embedding_cfg.dimension,
                **embed_kwargs,
            )
            logger.info(
                "Loaded Ollama embedder (model=%s, dimension=%d)",
                self.embedding_cfg.model,
                self.embedding_cfg.dimension,
            )

        else:
            embed_cls = parse_plugin_spec(self.embedding_cfg.provider)
            embed_kwargs = {**self.embedding_cfg.config}
            if self.embedding_cfg.model and "model" not in embed_kwargs:
                embed_kwargs["model"] = self.embedding_cfg.model
            if "dimension" not in embed_kwargs:
                embed_kwargs["dimension"] = self.embedding_cfg.dimension
            if isinstance(embed_cls, EmbeddingInterface):
                self.embedder = embed_cls
            elif inspect.isclass(embed_cls):
                self.embedder = embed_cls(**embed_kwargs)
            elif callable(embed_cls):
                self.embedder = CallableEmbedderAdapter(
                    callable_fn=embed_cls,
                    dimension=self.embedding_cfg.dimension,
                    name=self.embedding_cfg.provider,
                    preflight=bool(embed_kwargs.pop("preflight", True)),
                    default_kwargs=embed_kwargs,
                )
            else:
                raise TypeError(f"Unsupported embedder provider type: {type(embed_cls)}")
            logger.info("Loaded custom embedder adapter (%s)", self.embedding_cfg.provider)

        logger.info("LLM + Embedder initialization successful.")

    # ----------------------------------------------------------------------
    # Stores Initialization (uses unified storage config)
    # ----------------------------------------------------------------------
    def _init_stores(self) -> None:
        """
        Initialize all SQL + vector stores according to the unified storage config.

        storage.db_root        – root folder where UMA creates all SQL DB files
        storage.sql_backend    – sqlite | postgres
        storage.vector_backend – faiss | pinecone | weaviate | inmemory

        The same embedding dimension (embedding.dimension) is used for:
        - episodic vector index
        - semantic vector index
        - procedural vector index
        """
        dim = self.embedding_cfg.dimension
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Invalid embedding.dimension={dim!r}; must be > 0 integer.")

        storage_cfg = self.raw_config.storage

        # --------------------------------------------------------------
        # Validate and compute DB paths
        # --------------------------------------------------------------
        db_root = storage_cfg.db_root.rstrip("/") + "/"

        episodic_db_path = db_root + "episodic.db"
        semantic_db_path = db_root + "semantic.db"
        procedural_db_path = db_root + "procedural.db"

        # --------------------------------------------------------------
        # SQL BACKEND SELECTION
        # --------------------------------------------------------------
        sql_backend = storage_cfg.sql_backend

        if sql_backend == "sqlite":
            sql_adapter_cls = SQLiteAdapter

        elif sql_backend == "postgres":
            # Optional future backend: PostgresAdapter
            try:
                from uma.adapters.db.postgres_adapter import PostgresAdapter
            except ImportError as exc:
                logger.exception("PostgresAdapter import failed.")
                raise RuntimeError(
                    "storage.sql_backend='postgres' but PostgresAdapter is not available. "
                    "Install the required dependency or switch to 'sqlite'."
                ) from exc
            sql_adapter_cls = PostgresAdapter

        else:
            raise ValueError(f"Unsupported storage.sql_backend={sql_backend!r}")

        # Instantiate DB adapters
        epi_db = sql_adapter_cls(episodic_db_path)
        sem_db = sql_adapter_cls(semantic_db_path)
        pro_db = sql_adapter_cls(procedural_db_path)

        # --------------------------------------------------------------
        # VECTOR BACKEND SELECTION
        # --------------------------------------------------------------
        vector_backend = storage_cfg.vector_backend
        vector_cfg = storage_cfg.get("vector_config", {}) if isinstance(storage_cfg, dict) else {}

        if vector_backend == "faiss":
            # Prefer FAISS; fall back to in-memory if unavailable
            from uma.adapters.vector.inmemory import InMemoryVectorIndex

            def vector_init(d: int):
                try:
                    return FaissIndex(d)
                except Exception:
                    logger.exception(
                        "Failed to initialize FaissIndex; falling back to InMemoryVectorIndex."
                    )
                    return InMemoryVectorIndex.fallback_if_faiss_unavailable(d)

        elif vector_backend == "inmemory":
            from uma.adapters.vector.inmemory import InMemoryVectorIndex

            vector_init = lambda d: InMemoryVectorIndex(d)

        elif vector_backend == "pinecone":
            from uma.adapters.vector.pinecone_adapter import PineconeIndex

            index_name = vector_cfg.get("index_name", "")
            if not index_name:
                raise ValueError("Pinecone vector backend requires storage.vector_config.index_name")
            vector_init = lambda d: PineconeIndex(index_name=index_name, dim=d)

        elif vector_backend == "weaviate":
            from uma.adapters.vector.weaviate_adapter import WeaviateIndex

            url = vector_cfg.get("url", "")
            api_key = vector_cfg.get("api_key", "")
            class_name = vector_cfg.get("class_name", "")
            if not (url and api_key and class_name):
                raise ValueError(
                    "Weaviate vector backend requires storage.vector_config.url, "
                    "storage.vector_config.api_key, and storage.vector_config.class_name"
                )
            vector_init = lambda d: WeaviateIndex(
                url=url,
                api_key=api_key,
                class_name=class_name,
                dim=d,
            )

        else:
            raise ValueError(f"Unsupported storage.vector_backend={vector_backend!r}")

        # Instantiate vector indexes using the *shared* embedding dimension
        epi_idx = vector_init(dim)
        sem_idx = vector_init(dim)
        pro_idx = vector_init(dim)

        # --------------------------------------------------------------
        # Create SQL + Vector stores
        # --------------------------------------------------------------
        try:
            self.episodic_store = EpisodicSQLStore(epi_db, epi_idx)
            self.semantic_store = SemanticSQLStore(sem_db, sem_idx)
            self.procedural_store = ProceduralSQLStore(pro_db, pro_idx)
        except Exception:
            logger.exception("Failed to initialize one or more SQL/vector stores.")
            raise

        logger.info(
            "Stores initialized (sql_backend=%s, vector_backend=%s, db_root=%s, dim=%d)",
            sql_backend,
            vector_backend,
            db_root,
            dim,
        )

    # ----------------------------------------------------------------------
    # Core Subsystems (WM, Episodic, Semantic ONLY — Retrieval wired in initialize)
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

        RetrievalService is *NOT* wired here anymore — it is initialized
        exactly once inside initialize() to preserve idempotency.
        """
        if self.llm is None or self.embedder is None:
            raise RuntimeError(
                "UMAMemory._init_core_subsystems: LLM and embedder must "
                "be initialized before core subsystems."
            )

        if self.episodic_store is None or self.semantic_store is None:
            raise RuntimeError(
                "UMAMemory._init_core_subsystems: episodic_store and "
                "semantic_store must be initialized before core subsystems."
            )

        # ---------------------- Working Memory Core ----------------------
        try:
            wm_cfg = self.working_memory_cfg
            self.working_memory = WorkingMemoryCore(
                llm=self.llm,
                max_tokens=wm_cfg.max_tokens,
                warning_ratio=wm_cfg.warning_ratio,
                hard_limit_ratio=wm_cfg.hard_limit_ratio,
                chunk_size=wm_cfg.chunk_size,
                keep_recent_messages=getattr(wm_cfg, "keep_recent_messages", 4),
                keep_recent_token_fraction=getattr(wm_cfg, "keep_recent_token_fraction", 0.1),
            )
            logger.info("WorkingMemoryCore chunk_size set to %d", wm_cfg.chunk_size)

            logger.info(
                "WorkingMemoryCore initialized (max_tokens=%d, warn=%.2f, hard=%.2f)",
                wm_cfg.max_tokens,
                wm_cfg.warning_ratio,
                wm_cfg.hard_limit_ratio,
            )
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
                episodic_store=self.episodic_store,
                episode_indexer=indexer,
                retention_policy=retention,
            )
            logger.info("EpisodicCore initialized.")
        except Exception:
            logger.exception("UMAMemory: failed to initialize EpisodicCore.")
            raise

        # ---------------------- Semantic Core ---------------------------
        try:
            sal = self.consolidation_cfg.prune_min_fact_salience
            self.semantic_core = SemanticCore(
                llm=self.llm,
                embedder=self.embedder,
                semantic_store=self.semantic_store,
                salience_threshold=sal,
            )
            logger.info(
                "SemanticCore initialized (salience_threshold=%.2f).",
                sal,
            )
        except Exception:
            logger.exception("UMAMemory: failed to initialize SemanticCore.")
            raise


       # ----------------------------------------------------------------------
    # Graph Subsystem Initialization (Unified Backend Selection)
    # ----------------------------------------------------------------------
    def _init_graph_core(self) -> None:
        """
        Initialize the graph subsystem using the unified storage config.

        Uses:
            storage.graph_backend: "neo4j" | "memgraph" | "disabled"
            graph: connection details for supported backends

        Rules:
        -------
        • If graph_backend="disabled" → skip cleanly.
        • If neo4j → require uri, user, password.
        • If memgraph → require uri; user/password optional.
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
        if "graph" not in self.raw_config:
            raise ValueError(
                "Missing required 'graph' section in config when graph_backend "
                f"is '{backend}'."
            )

        graph_cfg = self.raw_config.graph

        # --------------------------------------------------------------
        # 3) Backend selection
        # --------------------------------------------------------------
        if backend == "neo4j":
            uri = graph_cfg.get("uri")
            user = graph_cfg.get("user")
            password = graph_cfg.get("password")
            pool = graph_cfg.get("max_pool_size", 20)
            database = graph_cfg.get("database")

            if not uri or not user or not password:
                raise ValueError(
                    "Graph backend 'neo4j' requires 'graph.uri', "
                    "'graph.user', and 'graph.password'."
                )

            try:
                adapter = Neo4jAdapter(
                    uri=uri,
                    user=user,
                    password=password,
                    max_pool_size=pool,
                    database=database,
                )
            except Exception as exc:
                logger.exception("Neo4jAdapter initialization failed.")
                raise RuntimeError(
                    "Failed to connect to Neo4j graph backend. "
                    "Verify URI, credentials, and server availability."
                ) from exc

        elif backend == "memgraph":
            from uma.adapters.graph.memgraph_adapter import MemgraphAdapter

            uri = graph_cfg.get("uri")
            pool = graph_cfg.get("max_pool_size", 20)

            if not uri:
                raise ValueError(
                    "Graph backend 'memgraph' requires 'graph.uri' to be set."
                )

            try:
                adapter = MemgraphAdapter(uri=uri, max_pool_size=pool)
            except Exception as exc:
                logger.exception("MemgraphAdapter initialization failed.")
                raise RuntimeError(
                    "Failed to connect to Memgraph backend. "
                    "Verify URI and server availability."
                ) from exc

        else:
            raise ValueError(
                f"Unsupported storage.graph_backend={backend!r}. "
                "Expected: 'neo4j', 'memgraph', or 'disabled'."
            )

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
            "store": self.procedural_store,
            "episodic_store": self.episodic_store,
            "semantic_store": self.semantic_store,
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
        user_id: Optional[str] = None,
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
            user_id=user_id,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            include_procedural=include_procedural,
            batch_size=batch_size,
        )
    
    # ----------------------------------------------------------------------
    # PUBLIC DEVELOPER API — Unified User Context (WM + LT Retrieval)
    # ----------------------------------------------------------------------
    async def get_user_context(self, user_id: str, query_text: str) -> Dict[str, list]:
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
                "semantic": [...],
                "procedural": [...],
                "graph": [...],
            }
        """
        from ..adapters.observability.metrics import increment, timed

        if not user_id or not isinstance(user_id, str):
            raise ValueError("UMAMemory.get_user_context: user_id must be a non-empty string.")
        if not query_text or not isinstance(query_text, str):
            raise ValueError("UMAMemory.get_user_context: query_text must be a non-empty string.")

        with timed("uma.get_user_context.latency"):
            # 1) Stored WM
            try:
                wm_stored = self.working_memory.get_context(user_id) if self.working_memory else []
            except Exception:
                logger.exception("UMAMemory.get_user_context: failed to load WM user=%s", user_id)
                wm_stored = []

            # If retrieval isn't wired, return WM only
            if not getattr(self, "retrieval_service", None):
                increment("uma.get_user_context.calls", tags={"path": "wm_only"})
                logger.warning(
                    "UMAMemory.get_user_context: retrieval_service not initialized; WM-only user=%s",
                    user_id,
                )
                return {
                    "working_memory": wm_stored,
                    "episodic": [],
                    "semantic": [],
                    "procedural": [],
                    "graph": [],
                }

            # 2) Prefer RLM retrieval
            if getattr(self, "rlm_controller", None) is not None:
                try:
                    pack = await self.rlm_controller.retrieve_context(
                        user_id=user_id,
                        query_text=query_text,
                    )
                    increment("uma.get_user_context.calls", tags={"path": "rlm"})
                    return {
                        "working_memory": wm_stored,
                        "episodic": pack.episodes,
                        "semantic": pack.facts,
                        "procedural": pack.skills,
                        "graph": pack.graph,
                    }
                except Exception:
                    logger.exception(
                        "UMAMemory.get_user_context: RLM failed; falling back to classic user=%s",
                        user_id,
                    )

            # 3) Classic fallback
            try:
                retrieved = await self.retrieval_service.retrieve(
                    user_id=user_id,
                    memory_type="all",
                    query_text_or_embedding=query_text,
                )
            except Exception:
                logger.exception(
                    "UMAMemory.get_user_context: classic retrieval failed user=%s",
                    user_id,
                )
                retrieved = {}

            increment("uma.get_user_context.calls", tags={"path": "classic"})
        return {
            "working_memory": wm_stored,
            "episodic": retrieved.get("episodes", []) or [],
            "semantic": retrieved.get("facts", retrieved.get("semantic", [])) or [],
            "procedural": retrieved.get("skills", retrieved.get("procedural", [])) or [],
            "graph": retrieved.get("graph", []) or [],
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
        if not self.initialized:
            raise RuntimeError("UMAMemory.process_turn requires initialized memory.")
        if not getattr(self, "pipeline", None):
            raise RuntimeError("UMAMemory.process_turn: pipeline not initialized.")

        await self.pipeline.process_turn(
            user_id=user_id,
            user_msg=user_msg,
            assistant_reply=assistant_reply,
        )

    # ----------------------------------------------------------------------
    # OPTIONAL UTILITIES — RAG-Ready Context Pack Builder
    # ----------------------------------------------------------------------

    async def build_context_pack(self, user_id: str, query_text: str) -> dict:
        """
        Build a RAG-ready structured context pack using UMA memory.

        Convenience wrapper around:
            - UMAMemory.get_user_context()
            - ContextPackBuilder.build()
        """
        from .utils.context_pack_builder import ContextPackBuilder

        ctx = await self.get_user_context(user_id, query_text)
        return ContextPackBuilder.build(query_text, ctx)

    async def build_prompt_messages(
        self,
        *,
        user_id: str,
        query_text: str,
        system_prompt: str,
    ) -> list:
        """
        Build LLM messages with UMA-RLM context embedded.

        This wraps retrieval + context formatting so developers do not
        manually collect memory slices.
        """
        from .utils.context_pack_builder import ContextPackBuilder

        pack = await self.build_context_pack(user_id=user_id, query_text=query_text)
        snippet = ContextPackBuilder.render_snippet(pack)
        if snippet:
            user_content = f"{query_text}\n\nRelevant memory:\n{snippet}"
        else:
            user_content = query_text

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

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
        from .utils.cot_memory_builder import CoTMemoryBuilder

        ctx = await self.get_user_context(user_id, query_text)
        return CoTMemoryBuilder.build(ctx)
    
    
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
