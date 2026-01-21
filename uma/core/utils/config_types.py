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
from typing import Any, Dict, List, Optional, Union


def parse_plugin_spec(spec: Union[str, Dict[str, Any]]):
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
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=d["provider"],
            model=d.get("model"),
            ollama_model=d.get("ollama_model"),
            config=d.get("config") or {},
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: Optional[str]
    dimension: int
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmbeddingConfig":
        return cls(
            provider=d["provider"],
            model=d.get("model"),
            dimension=d["dimension"],
            config=d.get("config") or {},
        )


# ---------------------------------------------------------------------------
# Memory stores: episodic, semantic, procedural
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodicConfig:
    db_path: str
    vector_dim: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodicConfig":
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
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticConfig":
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
    def from_dict(cls, d: Dict[str, Any]) -> "ProceduralConfig":
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
    def from_dict(cls, d: Dict[str, Any]) -> "WorkingMemorySettings":
        return cls(
            max_tokens=d["max_tokens"],
            warning_ratio=d["warning_ratio"],
            hard_limit_ratio=d["hard_limit_ratio"],
            chunk_size=d.get("chunk_size", 20),
            keep_recent_messages=d.get("keep_recent_messages", 4),
            keep_recent_token_fraction=d.get("keep_recent_token_fraction", 0.1),
        )


@dataclass
class RetrievalConfig:
    max_episodes: int
    max_facts: int
    max_skills: int
    max_graph_items: int

    # NEW
    rlm: Optional[RLMConfig] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievalConfig":
        rlm_cfg = d.get("rlm")
        rlm_obj: Optional[RLMConfig] = None
        if isinstance(rlm_cfg, dict):
            rlm_obj = RLMConfig(
                enabled=bool(rlm_cfg.get("enabled", False)),
                max_steps=int(rlm_cfg.get("max_steps", 4)),
                max_actions_per_step=int(rlm_cfg.get("max_actions_per_step", 2)),
                max_items_per_type=int(rlm_cfg.get("max_items_per_type", 30)),
                llm_max_tokens=int(rlm_cfg.get("llm_max_tokens", 300)),
                timeout_s=float(rlm_cfg.get("timeout_s", 20.0)),
                max_env_calls=int(rlm_cfg.get("max_env_calls", 12)),
                max_return_chars=int(rlm_cfg.get("max_return_chars", 1200)),
            )

        return cls(
            max_episodes=int(d["max_episodes"]),
            max_facts=int(d["max_facts"]),
            max_skills=int(d["max_skills"]),
            max_graph_items=int(d["max_graph_items"]),
            rlm=rlm_obj,
        )

    
@dataclass
class RLMConfig:
    enabled: bool = False
    max_steps: int = 4
    max_actions_per_step: int = 2
    max_items_per_type: int = 30
    llm_max_tokens: int = 300
    timeout_s: float = 20.0
    max_env_calls: int = 12
    max_return_chars: int = 1200

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeaturesConfig:
    load: List[Dict[str, Any]]
    policy: Dict[str, Any]
    procedural_enabled: bool
    consolidation_enabled: bool

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
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

        load: List[Dict[str, Any]] = []
        if procedural_enabled:
            load.append(
                {
                    "name": "procedural",
                    "enabled": True,
                    "provider": "uma.features.procedural.feature:ProceduralFeature",
                    "config": {},
                }
            )
        if consolidation_enabled:
            load.append(
                {
                    "name": "consolidation",
                    "enabled": True,
                    "provider": "uma.features.consolidation.feature:ConsolidationFeature",
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
    def from_dict(cls, d: Dict[str, Any]) -> "ConsolidationConfig":
        return cls(
            enabled=d.get("enabled", False),
            cluster_similarity=d["cluster_similarity"],
            max_episodes_per_cycle=d["max_episodes_per_cycle"],
            prune_min_fact_salience=d["prune_min_fact_salience"],   # ← REQUIRED FIELD
        )
