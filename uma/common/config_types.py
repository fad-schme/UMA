"""
config_types.py
================

Typed, frozen dataclasses representing UMA configuration sections.

These dataclasses replace dynamic UMAConfig attribute wrappers inside
UMAMemory, ensuring type safety, autocomplete support, and predictable
configuration behavior.

UMAMemory should convert UMAConfig → these dataclasses at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Optional, Union


def parse_plugin_spec(spec: Union[str, dict[str, Any]]):
    """
    Parse a plugin specification from configuration.

    Accepted forms:
    - A string of the form "module.path:callable" which will be imported
      via importlib and the attribute returned (preferred and safe).

    This avoids executing arbitrary code by default and prefers an
    import-by-path pattern.
    """
    # Shortcut: import path
    if isinstance(spec, str):
        if ":" not in spec:
            raise ValueError("plugin spec string must be 'module:attr'")
        module_name, attr = spec.split(":", 1)
        mod = importlib.import_module(module_name)
        if not hasattr(mod, attr):
            raise ImportError(f"module {module_name!r} has no attribute {attr!r}")
        return getattr(mod, attr)

    # Dict form is no longer supported for security reasons.
    if isinstance(spec, dict):
        raise ValueError("plugin spec must be an import path string 'module:attr'")

    raise ValueError("unsupported plugin spec type")




# example problematic usage (do not use):
# fn = types.FunctionType(source_string, globals())
# Replace any direct FunctionType-from-string usage with:
# fn = _make_function_from_source(source_string, globals())


# ---------------------------------------------------------------------------
# LLM + Embedding configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: Optional[str]
    ollama_model: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=d["provider"],
            model=d.get("model"),
            ollama_model=d.get("ollama_model"),
            config=d.get("config") or {},
        )


@dataclass(frozen=True)
class LLMsConfig:
    agent: LLMConfig
    uma: LLMConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMsConfig":
        # If explicit llms section exists, use it.
        if "llms" in d and isinstance(d["llms"], dict):
            llms = d["llms"]
            uma_cfg = LLMConfig.from_dict(llms["uma"])
            return cls(
                agent=LLMConfig.from_dict(llms["agent"]) if isinstance(llms.get("agent"), dict) else uma_cfg,
                uma=uma_cfg,
            )
        # Fallback to single llm section for UMA.
        if "llm" in d and isinstance(d["llm"], dict):
            uma_cfg = LLMConfig.from_dict(d["llm"])
            return cls(agent=uma_cfg, uma=uma_cfg)
        raise ValueError("Missing LLM configuration; expected 'llm' or 'llms' section")


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: Optional[str]
    dimension: int
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmbeddingConfig":
        return cls(
            provider=d["provider"],
            model=d.get("model"),
            dimension=d["dimension"],
            config=d.get("config") or {},
        )


# ---------------------------------------------------------------------------
# Secrets provider config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SecretsProviderConfig:
    provider: str
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> Optional["SecretsProviderConfig"]:
        if d is None:
            return None
        return cls(
            provider=d["provider"],
            options=d.get("options") or {},
        )


# ---------------------------------------------------------------------------
# Storage config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageConfig:
    db_root: str
    sql_backend: str
    vector_backend: str
    vector_config: dict[str, Any] = field(default_factory=dict)
    graph_backend: str = "disabled"
    graph_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StorageConfig":
        return cls(
            db_root=d["db_root"],
            sql_backend=d["sql_backend"],
            vector_backend=d["vector_backend"],
            vector_config=d.get("vector_config") or {},
            graph_backend=d.get("graph_backend", "disabled"),
            graph_config=d.get("graph_config") or {},
        )


# ---------------------------------------------------------------------------
# Memory stores: episodic, semantic, procedural
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodicConfig:
    db_path: str
    vector_dim: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodicConfig":
        return cls(
            db_path=d["db_path"],
            vector_dim=d["vector_dim"],
        )


@dataclass(frozen=True)
class SemanticConfig:
    db_path: str
    vector_dim: int
    salience_threshold: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SemanticConfig":
        return cls(
            db_path=d["db_path"],
            vector_dim=d["vector_dim"],
            salience_threshold=d["salience_threshold"],
        )


@dataclass(frozen=True)
class ProceduralConfig:
    db_path: str
    vector_dim: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProceduralConfig":
        return cls(
            db_path=d["db_path"],
            vector_dim=d["vector_dim"],
        )


# ---------------------------------------------------------------------------
# Working memory + Retrieval configs
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemorySettings:
    max_tokens: int
    warning_ratio: float
    hard_limit_ratio: float
    chunk_size: int = 20
    # How many recent messages to preserve at minimum during compaction
    keep_recent_messages: int = 4
    # Fraction of `max_tokens` that recent messages should cover before
    # allowing older messages to be summarized. This implements a
    # token+message hybrid rule: we keep at least `keep_recent_messages`
    # but may keep more until recent messages cover this fraction.
    keep_recent_token_fraction: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkingMemorySettings":
        return cls(
            max_tokens=d["max_tokens"],
            warning_ratio=d["warning_ratio"],
            hard_limit_ratio=d["hard_limit_ratio"],
            chunk_size=d.get("chunk_size", 20),
            keep_recent_messages=d.get("keep_recent_messages", 4),
            keep_recent_token_fraction=d.get("keep_recent_token_fraction", 0.1),
        )


@dataclass
class HybridRetrievalConfig:
    """
    Hybrid retrieval configuration (dense + lexical) for provider-agnostic core recall.

    Notes
    -----
    - `enabled`: toggles lexical retrieval + fusion when `query_text` is present.
    - `top_k_dense`: if <= 0, uses the call-site `k`.
    - `top_k_sparse`: lexical candidate pool size (0 disables lexical).
    - `fusion_strategy`: `rrf` (default) or `overlap_boost`.
    """

    enabled: bool = True
    top_k_dense: int = 0
    top_k_sparse: int = 15
    fusion_strategy: str = "rrf"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "HybridRetrievalConfig":
        d = d or {}
        enabled = bool(d.get("enabled", True))
        top_k_dense = int(d.get("top_k_dense", 0))
        top_k_sparse = int(d.get("top_k_sparse", 15))
        if top_k_dense < 0:
            raise ValueError("'retrieval.hybrid.top_k_dense' must be >= 0")
        if top_k_sparse < 0:
            raise ValueError("'retrieval.hybrid.top_k_sparse' must be >= 0")
        strategy = str(d.get("fusion_strategy", "rrf") or "rrf").strip().lower()
        if strategy not in ("rrf", "overlap_boost"):
            raise ValueError("'retrieval.hybrid.fusion_strategy' must be one of: rrf, overlap_boost")
        return cls(
            enabled=enabled,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            fusion_strategy=strategy,
        )


@dataclass
class RetrievalConfig:
    max_episodes: int
    max_facts: int
    max_skills: int
    max_graph_items: int
    context: "RetrievalContextConfig"
    strict: bool = True
    debug_scores: bool = False
    hybrid: HybridRetrievalConfig = field(default_factory=HybridRetrievalConfig)
    max_evidence_chunks: int = 6
    neighbor_window: int = 1
    max_expanded_chunks: int = 24

    # NEW
    rlm: Optional[RLMConfig] = None
    chunk_shortlist_k: int = 12
    chunk_shortlist_max_per_doc: int = 3

    # MMR/diversity-aware chunk selection (retrieval-ranking-gap ticket 07).
    # Off by default: only verified against a 15-question LoCoMo sample on
    # one vector backend (ticket 06) so far, not at production scale or
    # across FAISS/InMemory. mmr_lambda=0.6 is ticket 07's chosen default --
    # see its "Default lambda" finding for the full-sample comparison.
    # mmr_pool_multiplier widens the candidate pool MMR selects from (vs.
    # the plain path's 2x dedup-safety margin) -- MMR needs real headroom to
    # diversify within; 6x matches ticket 06's validated pool/k ratio.
    mmr_enabled: bool = False
    mmr_lambda: float = 0.6
    mmr_pool_multiplier: int = 6
    trust_weight: float = 0.15
    # H3: filter out quarantine-survivors at retrieval. With the boundary-
    # scan tier rubric in effect, a medium-severity hit reduces trust to
    # ~0.45 (=0.9 × 0.5 for a turn_user fact, similar for documents).
    # min_trust_score=0.5 filters every medium-severity survivor across
    # every legitimate source kind, while preserving clean tool_output
    # (trust=0.5 passes the >= comparison) and every higher-trust source.
    # The single edge case sacrificed is low-severity tool_output
    # (0.5 × 0.8 = 0.40), which is the system's weakest signal and is
    # most often false positives anyway.
    min_trust_score: float = 0.5

    # Gap analysis (reporting only — never filters or reranks). A fact is
    # flagged stale when its freshest supporting chunk is older than this, and
    # weakly supported when it rests on a single chunk below the trust bar.
    # The trust bar sits above `min_trust_score` on purpose: chunks below that
    # floor are already filtered out of retrieval entirely.
    gap_max_support_age_days: int = 180
    gap_min_support_trust: float = 0.6

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, profile: str = "lite") -> "RetrievalConfig":
        strict_mode = bool(d.get("strict", True))
        debug_scores = bool(d.get("debug_scores", False))
        max_evidence_chunks = int(d.get("max_evidence_chunks", 6))
        if max_evidence_chunks < 0:
            raise ValueError("'retrieval.max_evidence_chunks' must be a non-negative integer")
        neighbor_window = int(d.get("neighbor_window", 1))
        if neighbor_window < 0:
            raise ValueError("'retrieval.neighbor_window' must be a non-negative integer")
        max_expanded_chunks = int(d.get("max_expanded_chunks", 24))
        if max_expanded_chunks < 0:
            raise ValueError("'retrieval.max_expanded_chunks' must be a non-negative integer")
        chunk_shortlist_k = int(d.get("chunk_shortlist_k", 12))
        if chunk_shortlist_k < 0:
            raise ValueError("'retrieval.chunk_shortlist_k' must be a non-negative integer")
        chunk_shortlist_max_per_doc = int(d.get("chunk_shortlist_max_per_doc", 5))
        mmr_enabled = bool(d.get("mmr_enabled", False))
        mmr_lambda = float(d.get("mmr_lambda", 0.6))
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("'retrieval.mmr_lambda' must be between 0 and 1")
        mmr_pool_multiplier = int(d.get("mmr_pool_multiplier", 6))
        if mmr_pool_multiplier < 1:
            raise ValueError("'retrieval.mmr_pool_multiplier' must be a positive integer")
        if chunk_shortlist_max_per_doc < 0:
            raise ValueError("'retrieval.chunk_shortlist_max_per_doc' must be a non-negative integer")
        hybrid_cfg = d.get("hybrid")
        hybrid_obj = HybridRetrievalConfig.from_dict(hybrid_cfg if isinstance(hybrid_cfg, dict) else None)
        rlm_cfg = d.get("rlm")
        rlm_obj: Optional[RLMConfig] = None
        if isinstance(rlm_cfg, dict):
            allowlist = rlm_cfg.get("predicate_allowlist")
            predicate_allowlist = allowlist if isinstance(allowlist, dict) else None
            rlm_obj = RLMConfig(
                enabled=True,
                test_mode=bool(rlm_cfg.get("test_mode", False)),
                max_steps=int(rlm_cfg.get("max_steps", 4)),
                max_actions_per_step=int(rlm_cfg.get("max_actions_per_step", 2)),
                max_items_per_type=int(rlm_cfg.get("max_items_per_type", 30)),
                timeout_s=float(rlm_cfg.get("timeout_s", 20.0)),
                max_env_calls=int(rlm_cfg.get("max_env_calls", 12)),
                semantic_first=bool(rlm_cfg.get("semantic_first", True)),
                clusters_first=bool(rlm_cfg.get("clusters_first", True)),
                salience_threshold=float(rlm_cfg.get("salience_threshold", 0.6)),
                min_semantic_facts=int(rlm_cfg.get("min_semantic_facts", 4)),
                min_high_salience_facts=int(rlm_cfg.get("min_high_salience_facts", 2)),
                min_cluster_summaries=int(rlm_cfg.get("min_cluster_summaries", 1)),
                cluster_k=int(rlm_cfg.get("cluster_k", 3)),
                graph_predicate_limit=int(rlm_cfg.get("graph_predicate_limit", 2)),
                novelty_window=int(rlm_cfg.get("novelty_window", 2)),
                min_recent_novelty=int(rlm_cfg.get("min_recent_novelty", 1)),
                predicate_weights=(
                    rlm_cfg.get("predicate_weights")
                    if isinstance(rlm_cfg.get("predicate_weights"), dict)
                    else None
                ),
                predicate_allowlist=predicate_allowlist,
                max_new_facts_per_step=int(rlm_cfg.get("max_new_facts_per_step", 12)),
                max_new_chunks_per_step=int(rlm_cfg.get("max_new_chunks_per_step", 8)),
                max_graph_expansions_per_step=int(rlm_cfg.get("max_graph_expansions_per_step", 1)),
                chunk_fallback_enabled=bool(rlm_cfg.get("chunk_fallback_enabled", True)),
                chunk_fallback_k_multiplier=int(rlm_cfg.get("chunk_fallback_k_multiplier", 2)),
                query_decomposition_enabled=bool(rlm_cfg.get("query_decomposition_enabled", True)),
                query_decomposition_max_sub_queries=int(
                    rlm_cfg.get("query_decomposition_max_sub_queries", 4)
                ),
            )
        else:
            rlm_obj = RLMConfig(enabled=True)

        trust_weight = max(0.0, min(1.0, float(d.get("trust_weight", 0.15))))
        min_trust_score = max(0.0, min(1.0, float(d.get("min_trust_score", 0.5))))
        gap_max_support_age_days = int(d.get("gap_max_support_age_days", 180))
        if gap_max_support_age_days < 0:
            raise ValueError("'retrieval.gap_max_support_age_days' must be a non-negative integer")
        gap_min_support_trust = float(d.get("gap_min_support_trust", 0.6))
        if not 0.0 <= gap_min_support_trust <= 1.0:
            raise ValueError("'retrieval.gap_min_support_trust' must be between 0 and 1")
        return cls(
            max_episodes=int(d["max_episodes"]),
            max_facts=int(d["max_facts"]),
            max_skills=int(d["max_skills"]),
            max_graph_items=int(d["max_graph_items"]),
            hybrid=hybrid_obj,
            max_evidence_chunks=max_evidence_chunks,
            neighbor_window=neighbor_window,
            max_expanded_chunks=max_expanded_chunks,
            chunk_shortlist_k=chunk_shortlist_k,
            chunk_shortlist_max_per_doc=chunk_shortlist_max_per_doc,
            mmr_enabled=mmr_enabled,
            mmr_lambda=mmr_lambda,
            mmr_pool_multiplier=mmr_pool_multiplier,
            context=RetrievalContextConfig.from_dict(d.get("context") or {}, profile=profile),
            strict=strict_mode,
            debug_scores=debug_scores,
            rlm=rlm_obj,
            trust_weight=trust_weight,
            min_trust_score=min_trust_score,
            gap_max_support_age_days=gap_max_support_age_days,
            gap_min_support_trust=gap_min_support_trust,
        )


@dataclass
class RLMConfig:
    enabled: bool = False
    test_mode: bool = False
    max_steps: int = 4
    max_actions_per_step: int = 2
    max_items_per_type: int = 30
    timeout_s: float = 20.0
    max_env_calls: int = 12
    semantic_first: bool = True
    clusters_first: bool = True
    salience_threshold: float = 0.6
    min_semantic_facts: int = 4
    min_high_salience_facts: int = 2
    min_cluster_summaries: int = 1
    cluster_k: int = 3
    graph_predicate_limit: int = 2
    predicate_weights: Optional[dict[str, float]] = None
    predicate_allowlist: Optional[dict[str, list[str]]] = None
    novelty_window: int = 2
    min_recent_novelty: int = 1

    max_new_facts_per_step: int = 12
    max_new_chunks_per_step: int = 8
    max_graph_expansions_per_step: int = 1
    chunk_fallback_enabled: bool = True
    chunk_fallback_k_multiplier: int = 2

    # Query decomposition (retrieval-ranking-gap ticket 03): for broad/list
    # queries, split into narrower sub-queries and search each so scattered
    # answer-bearing chunks a single query embedding ranks far outside the
    # baseline pool still get a chance to compete for a slot. Skipped for
    # PERSONAL-intent queries (classify_query_intent) — those are already
    # well served by direct recall, and decomposition adds an LLM round trip
    # per query.
    query_decomposition_enabled: bool = True
    query_decomposition_max_sub_queries: int = 4


@dataclass
class RetrievalContextConfig:
    max_working_messages: int = 4
    max_episodic: int = 3
    max_semantic: int = 5
    max_chunks: int = 5
    max_procedural: int = 3
    max_graph: int = 3
    include_working_memory: bool = True
    include_episodic: bool = True
    include_graph: bool = True
    include_procedural: bool = True
    snippet_max_chars: int = 240
    snippet_refiner_available: bool = False
    snippet_refiner_top_k: int = 8
    episodic_clustering_available: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, profile: str = "lite") -> "RetrievalContextConfig":
        is_enterprise = profile == "enterprise"
        return cls(
            max_working_messages=int(d.get("max_working_messages", 4)),
            max_episodic=int(d.get("max_episodic", 3)),
            max_semantic=int(d.get("max_semantic", 5)),
            max_chunks=int(d.get("max_chunks", 5)),
            max_procedural=int(d.get("max_procedural", 3)),
            max_graph=int(d.get("max_graph", 3)),
            include_working_memory=bool(d.get("include_working_memory", True)),
            include_episodic=bool(d.get("include_episodic", True)),
            include_graph=bool(d.get("include_graph", True)),
            include_procedural=bool(d.get("include_procedural", True)),
            snippet_max_chars=int(d.get("snippet_max_chars", 240)),
            snippet_refiner_available=is_enterprise,
            snippet_refiner_top_k=int(d.get("snippet_refiner_top_k", 8)),
            episodic_clustering_available=is_enterprise,
        )

# ---------------------------------------------------------------------------
# Security config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SecurityConfig:
    scan_enabled: bool = True
    scan_severity_threshold: str = "high"
    custom_patterns_path: Optional[str] = None
    quarantine_enabled: bool = True

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "SecurityConfig":
        d = d or {}
        return cls(
            scan_enabled=bool(d.get("scan_enabled", True)),
            scan_severity_threshold=str(d.get("scan_severity_threshold", "high")),
            custom_patterns_path=d.get("custom_patterns_path") or None,
            quarantine_enabled=bool(d.get("quarantine_enabled", True)),
        )


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeaturesConfig:
    load: list[dict[str, Any]]
    policy: dict[str, Any]
    procedural_enabled: bool
    consolidation_enabled: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        load_cfg = d.get("load")
        policy_cfg = d.get("policy") or {}
        procedural_enabled = d.get("procedural_enabled", True)
        consolidation_enabled = d.get("consolidation_enabled", True)

        if isinstance(load_cfg, list):
            return cls(
                load=load_cfg,
                policy=policy_cfg,
                procedural_enabled=procedural_enabled,
                consolidation_enabled=consolidation_enabled,
            )

        load: list[dict[str, Any]] = []
        if procedural_enabled:
            load.append(
                {
                    "name": "procedural",
                    "enabled": True,
                    "provider": "uma.memory.procedural.feature:ProceduralFeature",
                    "config": {},
                }
            )
        if consolidation_enabled:
            load.append(
                {
                    "name": "consolidation",
                    "enabled": True,
                    "provider": "uma.memory.consolidation.feature:ConsolidationFeature",
                    "config": {},
                }
            )

        return cls(
            load=load,
            policy=policy_cfg,
            procedural_enabled=procedural_enabled,
            consolidation_enabled=consolidation_enabled,
        )


# ---------------------------------------------------------------------------
# Consolidation Config
# ---------------------------------------------------------------------------
@dataclass
class ConsolidationConfig:
    enabled: bool
    cluster_similarity: float
    max_episodes_per_cycle: int
    prune_min_fact_salience: float    # ← REQUIRED FIELD

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "ConsolidationConfig":
        d = d or {}
        return cls(
            enabled=d.get("enabled", False),
            cluster_similarity=float(d.get("cluster_similarity", 0.75)),
            max_episodes_per_cycle=int(d.get("max_episodes_per_cycle", 200)),
            prune_min_fact_salience=float(d.get("prune_min_fact_salience", 0.2)),
        )


# ---------------------------------------------------------------------------
# Unified runtime config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeConfig:
    storage: StorageConfig
    llm: LLMConfig
    agent_llm: LLMConfig
    embedding: EmbeddingConfig
    secrets: Optional[SecretsProviderConfig]
    working_memory: WorkingMemorySettings
    retrieval: RetrievalConfig
    features: FeaturesConfig
    consolidation: ConsolidationConfig
    security: SecurityConfig
    semantic_salience_threshold: float
    semantic_salience_decay_days: float = 180.0
    profile: str = "lite"

    @classmethod
    def from_uma_config(cls, cfg: dict[str, Any]) -> "RuntimeConfig":
        profile = str(cfg.get("profile", "lite")) if isinstance(cfg, dict) else "lite"

        llms_cfg = LLMsConfig.from_dict(cfg) if isinstance(cfg, dict) else None
        llm_cfg = llms_cfg.uma if llms_cfg else LLMConfig.from_dict(cfg["llm"])
        agent_llm_cfg = llms_cfg.agent if llms_cfg else llm_cfg

        embedding_cfg = EmbeddingConfig.from_dict(cfg["embedding"])
        secrets_cfg = SecretsProviderConfig.from_dict(cfg.get("secrets") if isinstance(cfg, dict) else None)
        working_memory_cfg = WorkingMemorySettings.from_dict(cfg["working_memory"])
        retrieval_cfg = RetrievalConfig.from_dict(cfg["retrieval"], profile=profile)
        features_cfg = FeaturesConfig.from_dict(cfg.get("features") or {})
        consolidation_cfg = ConsolidationConfig.from_dict(cfg.get("consolidation"))
        storage_cfg = StorageConfig.from_dict(cfg["storage"])
        security_cfg = SecurityConfig.from_dict(cfg.get("security") if isinstance(cfg, dict) else None)

        semantic_section = cfg.get("semantic", {}) if isinstance(cfg, dict) else {}
        semantic_salience = semantic_section.get("salience_threshold")
        if semantic_salience is None:
            semantic_salience = consolidation_cfg.prune_min_fact_salience
        semantic_decay_days = float(semantic_section.get("salience_decay_days", 180.0))

        return cls(
            storage=storage_cfg,
            llm=llm_cfg,
            agent_llm=agent_llm_cfg,
            embedding=embedding_cfg,
            secrets=secrets_cfg,
            working_memory=working_memory_cfg,
            retrieval=retrieval_cfg,
            features=features_cfg,
            consolidation=consolidation_cfg,
            security=security_cfg,
            semantic_salience_threshold=float(semantic_salience),
            semantic_salience_decay_days=semantic_decay_days,
            profile=profile,
        )
