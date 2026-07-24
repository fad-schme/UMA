"""Factory for `ContextBundle` instances in tests.

`ContextBundle` is strict (`extra="forbid"`), so tests that construct a
bundle must supply every required field. This factory carries sensible
empty defaults — tests override only the fields under assertion.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from uma.common.results import ContextBundle, Confidence, DebugInfo, Provenance


def make_context_bundle(
    *,
    query: str = "",
    working_memory: Optional[List[Any]] = None,
    episodic: Optional[List[Any]] = None,
    facts: Optional[List[Any]] = None,
    chunks: Optional[List[Any]] = None,
    documents: Optional[List[dict]] = None,
    skills: Optional[List[Any]] = None,
    graph: Optional[List[Any]] = None,
    confidence: Optional[Confidence] = None,
    query_scan_severity: str = "none",
    lane_filter: Optional[Iterable[str]] = None,
    active_lanes: Optional[Iterable[str]] = None,
    trace: Optional[List[dict]] = None,
    pruned_via_llm: bool = False,
    provenance: Optional[Provenance] = None,
) -> ContextBundle:
    """Return a validator-passing `ContextBundle` with test-friendly defaults."""
    if provenance is None:
        provenance = Provenance(
            source_chunk_ids=[],
            source_document_ids=[],
            derivation_type="context_retrieval",
            retrieval_path=[],
            parent_artifact_ids=[],
            support_density=0.0,
            confidence=None,
            conflicts=[],
            evidence_scopes=[],
            manual=False,
            valid=True,
            invalid_reasons=[],
        )
    return ContextBundle(
        query=query,
        working_memory=list(working_memory or []),
        episodic=list(episodic or []),
        facts=list(facts or []),
        chunks=list(chunks or []),
        documents=list(documents or []),
        skills=list(skills or []),
        graph=list(graph or []),
        confidence=confidence,
        provenance=provenance,
        query_scan_severity=query_scan_severity,  # type: ignore[arg-type]
        debug=DebugInfo(
            lane_filter=list(lane_filter or []),
            active_lanes=list(active_lanes or []),
            trace=list(trace or []),
            pruned_via_llm=pruned_via_llm,
        ),
    )
