"""
UMA3Memory — Core UMA-3 Memory Runtime
=====================================

UMA-3 is a **memory SDK**, not an autonomous agent.

This class owns and orchestrates all UMA-3 memory subsystems:

    • Working Memory (short-term conversational state)
    • Episodic Memory (indexed event history)
    • Semantic Memory (facts, preferences, domain knowledge)
    • Procedural Memory (skills, routines)
    • Temporal Graph (optional knowledge graph)
    • RetrievalService (developer-facing recall API)

UMA-3 does **not** generate assistant replies and does **not** perform
agent reasoning. Developers bring their own LLM or agent loop and use
UMA-3 strictly for memory management.

Typical developer workflow
--------------------------

1. Initialize UMA-3:

    memory = UMA3Memory.from_yaml("uma3.yaml")
    memory.initialize()

2. After the developer's agent produces a reply, store the memory turn:

    await pipeline.process_turn(
        user_id=user_id,
        user_msg=user_message,
        assistant_reply=final_agent_reply,
    )

   The MemoryPipeline handles:
        - working memory update + compaction
        - episodic memory generation
        - semantic fact ingestion
        - graph updates
        - lifecycle hooks

3. When constructing a prompt for the agent, fetch UMA-3 context:

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
   UMA-3 never performs reasoning or constructs prompts on its own.

Design Philosophy
-----------------
- UMA-3 provides *memory operations*, not agent behaviors.
- All reasoning, tool use, and final reply generation happen outside UMA-3.
- UMA-3 focuses on correctness, retrieval quality, summarization, and
  long-term memory structure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .memory_config import UMA3Config     # YAML loader + validation (dict-like)
from .config_types import (
    LLMConfig,
    EmbeddingConfig,
    WorkingMemorySettings,
    RetrievalConfig,
    FeaturesConfig,
    ConsolidationConfig,
)

from .hooks import UMA3Hooks
from .registry import FeatureRegistry
from .working_memory.core import WorkingMemoryCore
from .episodic.core import EpisodicCore
from .episodic.indexer import EpisodeIndexer
from .episodic.policies import EpisodicRetentionPolicy
from .semantic.core import SemanticCore
from .retrieval.service import RetrievalService
from .graph import TemporalGraphCore

# Stores
from ..stores.episodic_sql import EpisodicSQLStore
from ..stores.semantic_sql import SemanticSQLStore
from ..stores.procedural_sql import ProceduralSQLStore

# DB + Vector Adapters
from ..adapters.db.sqlite_adapter import SQLiteAdapter
from ..adapters.vector.faiss_adapter import FaissIndex
from ..adapters.graph.neo4j_adapter import Neo4jAdapter

# Optional Features
from ..features.procedural.feature import ProceduralFeature
from ..features.consolidation.feature import ConsolidationFeature

logger = logging.getLogger(__name__)


class UMA3Memory:
    """
    UMA-3 Memory Runtime Container.

    This class:
        - Initializes all core subsystems
        - Provides public APIs for pipeline and retrieval
        - Loads optional features via FeatureRegistry

    Developers ONLY interact with:
        memory = UMA3Memory.from_yaml("uma3.yaml")
        memory.initialize()

        # after their own agent produces a reply:
        await pipeline.process_turn(user_id, user_msg, assistant_reply)

        # when building prompts:
        ctx = await memory.get_user_context(user_id, query_text)

    All other subsystems remain internal.

    The configuration is YAML-driven and MUST contain at top-level:
        storage           – {db_root, sql_backend, vector_backend, graph_backend}
        working_memory    – WM token window + thresholds
        embedding         – provider, model, dimension
        llm               – UMA-3 internal LLM provider + model
        retrieval         – caps for episodes/facts/skills/graph items
        consolidation     – cluster_similarity, max_episodes_per_cycle,
                            prune_min_fact_salience
        features          – {procedural_enabled, consolidation_enabled}
        graph             – (only required when storage.graph_backend != "disabled")
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "UMA3Memory":
        """Load YAML config and construct UMA3Memory."""
        cfg = UMA3Config.load_yaml(path)
        return cls(cfg)

    def __init__(self, config: UMA3Config) -> None:
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

        # Typed config objects for subsystems that benefit from strong typing
        self.llm_cfg = LLMConfig.from_dict(config.llm)
        self.embedding_cfg = EmbeddingConfig.from_dict(config.embedding)
        self.working_memory_cfg = WorkingMemorySettings.from_dict(config.working_memory)
        self.retrieval_cfg = RetrievalConfig.from_dict(config.retrieval)
        self.features_cfg = FeaturesConfig.from_dict(config.features)
        self.consolidation_cfg = ConsolidationConfig.from_dict(config.consolidation)

        # Hooks + Feature Registry
        self.hooks = UMA3Hooks()
        self.feature_registry = FeatureRegistry()
        self.features: Dict[str, Any] = {}

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

        logger.info("UMA3Memory instance created with unified storage config.")

    # ----------------------------------------------------------------------
    # Initialization Entry Point
    # ----------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize all REQUIRED UMA-3 subsystems in a safe, idempotent way.

        RetrievalService is wired exactly once and never overwritten.
        """
        if self.initialized:
            logger.debug("UMA3Memory.initialize(): already initialized — skipping.")
            return

        logger.info("Initializing UMA-3 Memory Runtime...")

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

        # 6. Wire RetrievalService EXACTLY once
        if self.retrieval_service is None:
            try:
                self.retrieval_service = RetrievalService(
                    memory=self,
                    retr_cfg=self.retrieval_cfg,
                )
                logger.info("RetrievalService wired to UMA3Memory.")
            except Exception:
                logger.exception("Failed to initialize RetrievalService.")
                raise
        else:
            logger.debug("RetrievalService already initialized — not overwriting.")

        # Mark initialization complete
        self.initialized = True
        logger.info("UMA-3 Memory Runtime initialized successfully.")

    # ----------------------------------------------------------------------
    # LLM + Embedder Initialization (Typed Configs)
    # ----------------------------------------------------------------------

    def _init_llm_and_embedder(self) -> None:
        """
        Initialize the LLM and embedding model based on typed dataclass configs:
            - self.llm_cfg   (LLMConfig)
            - self.embedding_cfg (EmbeddingConfig)

        This method replaces the old dynamic config loader and guarantees that
        UMA3Memory always uses validated, structured configuration settings.
        """

        # ----------------------------- LLM -----------------------------
        if self.llm_cfg.provider == "openai":
            from uma3.adapters.llm.openai_llm import OpenAILLM

            self.llm = OpenAILLM(model=self.llm_cfg.model)
            logger.info("Loaded OpenAI LLM (model=%s)", self.llm_cfg.model)

        elif self.llm_cfg.provider == "ollama":
            from uma3.adapters.llm.ollama_llm import OllamaLLM

            model = self.llm_cfg.ollama_model or self.llm_cfg.model
            self.llm = OllamaLLM(model=model)
            logger.info("Loaded Ollama LLM (model=%s)", model)

        else:
            raise ValueError(f"Unsupported llm.provider={self.llm_cfg.provider!r}")

        # --------------------------- EMBEDDER --------------------------
        if self.embedding_cfg.provider == "openai":
            from uma3.adapters.llm.openai_embedding import OpenAIEmbedder

            self.embedder = OpenAIEmbedder()
            logger.info("Loaded OpenAI embedder (model=%s)", self.embedding_cfg.model)

        elif self.embedding_cfg.provider == "ollama":
            from uma3.adapters.llm.ollama_embedding import OllamaEmbedder

            if not (self.embedding_cfg.model and self.embedding_cfg.dimension):
                raise ValueError(
                    "For Ollama embedder: both 'model' and 'dimension' must be specified "
                    "in the embedding configuration."
                )

            self.embedder = OllamaEmbedder(
                model=self.embedding_cfg.model,
                mode="native",
                dimension=self.embedding_cfg.dimension,
            )
            logger.info(
                "Loaded Ollama embedder (model=%s, dimension=%d)",
                self.embedding_cfg.model,
                self.embedding_cfg.dimension,
            )

        else:
            raise ValueError(f"Unsupported embedding.provider={self.embedding_cfg.provider!r}")

        logger.info("LLM + Embedder initialization successful.")

    # ----------------------------------------------------------------------
    # Stores Initialization (uses unified storage config)
    # ----------------------------------------------------------------------
    def _init_stores(self) -> None:
        """
        Initialize all SQL + vector stores according to the unified storage config.

        storage.db_root        – root folder where UMA-3 creates all SQL DB files
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
                from uma3.adapters.db.postgres_adapter import PostgresAdapter
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

        if vector_backend == "faiss":
            # Prefer FAISS; fall back to in-memory if unavailable
            from uma3.adapters.vector.inmemory import InMemoryVectorIndex

            def vector_init(d: int):
                try:
                    return FaissIndex(d)
                except Exception:
                    logger.exception(
                        "Failed to initialize FaissIndex; falling back to InMemoryVectorIndex."
                    )
                    return InMemoryVectorIndex.fallback_if_faiss_unavailable(d)

        elif vector_backend == "inmemory":
            from uma3.adapters.vector.inmemory import InMemoryVectorIndex

            vector_init = lambda d: InMemoryVectorIndex(d)

        elif vector_backend == "pinecone":
            from uma3.adapters.vector.pinecone_adapter import PineconeIndex

            vector_init = lambda d: PineconeIndex(d)

        elif vector_backend == "weaviate":
            from uma3.adapters.vector.weaviate_adapter import WeaviateIndex

            vector_init = lambda d: WeaviateIndex(d)

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
        Initialize core UMA-3 subsystems that depend on:
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
                "UMA3Memory._init_core_subsystems: LLM and embedder must "
                "be initialized before core subsystems."
            )

        if self.episodic_store is None or self.semantic_store is None:
            raise RuntimeError(
                "UMA3Memory._init_core_subsystems: episodic_store and "
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
            logger.exception("UMA3Memory: failed to initialize WorkingMemoryCore.")
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
            logger.exception("UMA3Memory: failed to initialize EpisodicCore.")
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
            logger.exception("UMA3Memory: failed to initialize SemanticCore.")
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
                )
            except Exception as exc:
                logger.exception("Neo4jAdapter initialization failed.")
                raise RuntimeError(
                    "Failed to connect to Neo4j graph backend. "
                    "Verify URI, credentials, and server availability."
                ) from exc

        elif backend == "memgraph":
            from uma3.adapters.graph.memgraph_adapter import MemgraphAdapter

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
        """Attach optional UMA-3 features (procedural, consolidation)."""
        if self.features_cfg.procedural_enabled:
            try:
                ProceduralFeature(
                    store=self.procedural_store,
                    embedder=self.embedder,
                ).attach(self)
                logger.info("ProceduralFeature attached.")
            except Exception:
                logger.exception("Failed to attach ProceduralFeature.")

        if self.features_cfg.consolidation_enabled:
            try:
                ConsolidationFeature(
                    episodic_store=self.episodic_store,
                    semantic_store=self.semantic_store,
                    llm=self.llm,
                    embedder=self.embedder,
                ).attach(self)
                logger.info("ConsolidationFeature attached.")
            except Exception:
                logger.exception("Failed to attach ConsolidationFeature.")

    
    # ----------------------------------------------------------------------
    # PUBLIC DEVELOPER API — Unified User Context (WM + LT Retrieval)
    # ----------------------------------------------------------------------

    async def get_user_context(
        self,
        user_id: str,
        query_text: str,
    ) -> Dict[str, list]:
        """
        Unified, developer-facing memory context retrieval for UMA-3.

        This method provides the complete context an AI agent needs for
        prompt construction. It returns:

            - Real Working Memory (WM): short-term conversation history.
            - Long-Term Memory Summary Nodes (LT nodes): summarized episodic,
              semantic, procedural, and graph knowledge relevant to the query.
            - Episodic matches (vector-based)
            - Semantic facts (vector + structure)
            - Procedural skills (vector-based)
            - Graph items (temporal KG)

        IMPORTANT:
        ----------
        • Working Memory is NOT mutated here.
          Long-term memory is *summarized* into temporary WM-style messages
          and appended to the returned WM list, but NOT stored.

        • This preserves WM determinism and keeps the SDK clean.

        Parameters
        ----------
        user_id : str
            Logical user/session identifier.

        query_text : str
            Natural-language prompt or question. Used for retrieval and LT
            summarization.

        Returns
        -------
        Dict[str, list]
            {
                "working_memory": [... synthetic + real ...],
                "episodic": [...],
                "semantic": [...],
                "procedural": [...],
                "graph": [...],
            }
        """
        # ---------------------------------------------------------------
        # Validate input
        # ---------------------------------------------------------------
        if not query_text or not isinstance(query_text, str):
            raise ValueError(
                "UMA3Memory.get_user_context: query_text must be a non-empty string."
            )

        # ---------------------------------------------------------------
        # 1) Retrieve REAL Working Memory (stored WM)
        # ---------------------------------------------------------------
        try:
            if self.working_memory:
                wm_stored = self.working_memory.get_context(user_id)
            else:
                wm_stored = []
        except Exception:
            logger.exception(
                "UMA3Memory.get_user_context: failed to load stored working memory."
            )
            wm_stored = []

        # ---------------------------------------------------------------
        # 2) Compute Long-Term Memory summary nodes (non-mutating)
        # ---------------------------------------------------------------
        if not self.retrieval_service or not self.working_memory:
            logger.warning(
                "UMA3Memory.get_user_context: retrieval_service or working_memory "
                "not initialized; returning only real WM."
            )
            return {
                "working_memory": wm_stored,
                "episodic": [],
                "semantic": [],
                "procedural": [],
                "graph": [],
            }

        try:
            lt_nodes = await self.working_memory.retrieve_long_memory(
                user_id=user_id,
                query_text=query_text,
                retrieval_service=self.retrieval_service,
            )
        except Exception:
            logger.exception(
                "UMA3Memory.get_user_context: long-term WM retrieval failed "
                "for user_id=%s",
                user_id,
            )
            lt_nodes = []

        # ---------------------------------------------------------------
        # 3) Retrieve individual memory systems (episodic/semantic/etc.)
        # ---------------------------------------------------------------
        try:
            retrieved = await self.retrieval_service.retrieve(
                user_id=user_id,
                memory_type="all",
                query_text_or_embedding=query_text,
            )
        except Exception:
            logger.exception("UMA3Memory.get_user_context: retrieval error.")
            retrieved = {}

        epis = retrieved.get("episodes", [])
        sem = retrieved.get("semantic", [])
        pro = retrieved.get("procedural", [])
        grf = retrieved.get("graph", [])

        # ---------------------------------------------------------------
        # 4) Compose unified Working Memory view
        # ---------------------------------------------------------------
        # NOTE:
        #   wm_stored = “true WM”
        #   lt_nodes  = “synthetic WM” (summary of long-term memory)
        #   combined_wm = what the agent actually uses as context
        combined_wm = wm_stored + lt_nodes

        # ---------------------------------------------------------------
        # 5) Return unified context dict
        # ---------------------------------------------------------------
        return {
            "working_memory": combined_wm,
            "episodic": epis,
            "semantic": sem,
            "procedural": pro,
            "graph": grf,
        }


    # ----------------------------------------------------------------------
    # OPTIONAL UTILITIES — RAG-Ready Context Pack Builder
    # ----------------------------------------------------------------------

    async def build_context_pack(self, user_id: str, query_text: str) -> dict:
        """
        Build a RAG-ready structured context pack using UMA-3 memory.

        Convenience wrapper around:
            - UMA3Memory.get_user_context()
            - ContextPackBuilder.build()
        """
        from .utils.context_pack_builder import ContextPackBuilder

        ctx = await self.get_user_context(user_id, query_text)
        return ContextPackBuilder.build(query_text, ctx)

    # ----------------------------------------------------------------------
    # OPTIONAL UTILITIES — Structured CoT Memory Builder
    # ----------------------------------------------------------------------

    async def build_cot_memory(self, user_id: str, query_text: str) -> dict:
        """
        Build a structured chain-of-thought (CoT) knowledge scaffold from UMA-3.

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