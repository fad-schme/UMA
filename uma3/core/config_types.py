"""
config_types.py
================

Typed, frozen dataclasses representing UMA-3 configuration sections.

These dataclasses replace dynamic UMA3Config attribute wrappers inside
UMA3Memory, ensuring type safety, autocomplete support, and predictable
configuration behavior.

UMA3Memory should convert UMA3Config → these dataclasses at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import importlib
import types
from typing import Optional, Dict, Any, Union


def _make_function_from_source(src: str, glob: Optional[Dict[str, Any]] = None):
    """
    Safely compile a function source string and return the first defined function object.
    Use instead of types.FunctionType(source_string, ...).
    """
    ns: Dict[str, Any] = {}
    compiled = compile(src, "<config-fn>", "exec")
    exec(compiled, glob or {}, ns)
    # return first callable defined in the namespace
    for v in ns.values():
        if inspect.isfunction(v):
            return v
    raise ValueError("no function defined in source")


def parse_plugin_spec(spec: Union[str, Dict[str, Any]], allow_code: bool = False):
    """
    Parse a plugin specification from configuration.

    Accepted forms:
    - A string of the form "module.path:callable" which will be imported
      via importlib and the attribute returned (preferred and safe).
    - A dict with key "source" containing Python source code defining a
      callable. This is executed only when `allow_code` is True.

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

    # Dict form: allow 'source' only when explicitly enabled
    if isinstance(spec, dict):
        src = spec.get("source")
        if src is None:
            raise ValueError("plugin dict spec must contain 'source' key")
        if not allow_code:
            raise PermissionError("Execution of code in config is disabled; set security.allow_config_code to true to enable")
        # Fallback to old behavior for advanced users (still unsafe)
        return _make_function_from_source(src, glob=spec.get("globals"))

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
    model: str
    ollama_model: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=d["provider"],
            model=d.get("model"),
            ollama_model=d.get("ollama_model"),
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimension: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmbeddingConfig":
        return cls(
            provider=d["provider"],
            model=d["model"],
            dimension=d["dimension"],
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


@dataclass(frozen=True)
class RetrievalConfig:
    max_episodes: int
    max_facts: int
    max_skills: int
    max_graph_items: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            max_episodes=d["max_episodes"],
            max_facts=d["max_facts"],
            max_skills=d["max_skills"],
            max_graph_items=d["max_graph_items"],
        )


# ---------------------------------------------------------------------------
# Graph config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphConfig:
    backend: str
    uri: str
    user: str
    password: str
    max_pool_size: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            backend=d["backend"],
            uri=d.get("uri", ""),
            user=d.get("user", ""),
            password=d.get("password", ""),
            max_pool_size=d.get("max_pool_size", 10),
        )


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeaturesConfig:
    procedural_enabled: bool
    consolidation_enabled: bool

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            procedural_enabled=d.get("procedural_enabled", True),
            consolidation_enabled=d.get("consolidation_enabled", True),
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