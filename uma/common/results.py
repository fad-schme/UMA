"""Typed response models for UMA's public retrieval API.

Every model is strict (`extra="forbid"`): unknown keys raise. UMA is
v0.1.0 — there is no legacy dict compatibility. Callers use attribute
access; there is no `.to_dict()` shim.

Design principles
-----------------
- Attribute access, not `result["key"]`.
- Domain types are reused (`Fact`, `Episode`, `Skill`, `Chunk`) — the
  retrieval product is the same shape as the storage type.
- Observability lives in `debug` (opt-in surface for `explain_result()`
  and audit logs), not on the primary product.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from uma.common.health import HealthCheck
from uma.common.types import Chunk, Episode, Fact, Skill


class Confidence(BaseModel):
    """Aggregate confidence scoring derived from a coverage report.

    Present on a `ContextBundle` only when the retrieval pipeline
    produced a coverage report; otherwise the bundle's `confidence`
    field is `None`.
    """

    model_config = ConfigDict(extra="forbid")

    score: float
    semantic_enough: float
    clusters_present: float
    graph_present: float
    graph_entity_support: float
    graph_predicate_support: float
    novelty_recent: float
    contradictions: float


class Provenance(BaseModel):
    """Provenance for a retrieval-time context bundle.

    Fields mirror `uma.common.provenance.build_provenance`; the schema is
    a hard contract. Extra keys are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    source_chunk_ids: List[str]
    source_document_ids: List[str]
    derived_at: Optional[str] = None
    derivation_type: str
    retrieval_path: List[dict]
    parent_artifact_ids: List[str]
    support_density: Optional[float] = None
    confidence: Optional[float] = None
    conflicts: List[dict]
    evidence_scopes: List[dict]
    manual: bool
    valid: bool
    invalid_reasons: List[str]


class DebugInfo(BaseModel):
    """Observability surface for a retrieval call.

    Not part of the primary product. Consumers on the hot path SHOULD
    NOT read these fields for application logic; they exist for
    `memory.explain_result(...)`, the retrieval-audit writer, and tests.

    Fields
    ------
    lane_filter
        The `lane_filter` argument echoed from the caller (empty list
        means "all lanes were candidates").
    active_lanes
        The lanes the planner selected for this call after narrowing
        `lane_filter`.
    trace
        Ordered pipeline steps (planner decisions, RLM actions,
        selection events). Consumed by `explain_result`.
    pruned_via_llm
        True when the LLM-driven fact pruner ran during this call.
    """

    model_config = ConfigDict(extra="forbid")

    lane_filter: List[str]
    active_lanes: List[str]
    trace: List[dict]
    pruned_via_llm: bool


class ContextBundle(BaseModel):
    """Return type of `UMAMemory.retrieve_context`.

    Product surface (12 fields) is fully typed; observability lives in
    `debug`. The bundle is strict — unknown keys at construction time
    raise `ValidationError`.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    product: Literal["context"] = "context"
    query: str

    working_memory: List[Any]
    episodic: List[Episode]
    facts: List[Fact]
    chunks: List[Chunk]
    documents: List[dict]
    skills: List[Skill]
    graph: List[Any]

    confidence: Optional[Confidence] = None
    provenance: Provenance
    query_scan_severity: Literal["none", "low", "medium", "high"]

    debug: DebugInfo


class HealthStatus(BaseModel):
    """Return type of `UMAMemory.health_check`.

    Reuses the existing `HealthCheck` dataclass directly for per-check
    entries — no view-shape duplication. `status` is one of three
    literal values, mirroring the aggregation logic in
    `uma.common.health.run_health_checks`.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: Literal["ok", "degraded", "error"]
    checks: Dict[str, HealthCheck]


class CompiledMemory(BaseModel):
    """The compiled-answer summary attached to a `MemoryResult`.

    Present when the memory retrieval produced a compiled answer artifact
    (evidence was available); `None` on the evidence-only fallback path.
    """

    model_config = ConfigDict(extra="forbid")

    memory_intent: Optional[str]
    provenance_valid: bool


class MemoryResult(BaseModel):
    """Return type of `UMAMemory.retrieve_memory`.

    This is a narrower projection of the retrieval product than
    `ContextBundle`: it exposes an LLM-facing compiled answer, the
    serialized supporting facts and evidence, and provenance validity.
    Full retrieval detail is available only when `include_debug=True`.

    Notes on shapes
    ---------------
    - `facts` and `evidence` are serialized dict projections of the
      domain types (deliberately narrower than `Fact` / `Chunk`) — kept
      as `List[dict]` for D2. If a future fix wants typed sub-models,
      that is a separate design pass.
    - `debug` is opt-in observability: `None` unless the caller passes
      `include_debug=True`. Its internal structure carries every field
      of the runtime's detailed_result; deliberately untyped so future
      changes to that internal shape do not break the public model.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    product: Literal["memory"] = "memory"
    query: str
    compiled_memory: Optional[CompiledMemory]
    facts: List[dict]
    evidence: List[dict]
    provenance_valid: bool
    provenance_error: Optional[str] = None
    debug: Optional[dict] = None


__all__ = [
    "CompiledMemory",
    "Confidence",
    "ContextBundle",
    "DebugInfo",
    "HealthStatus",
    "MemoryResult",
    "Provenance",
]
