from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

from uma.common.storage_metadata import (
    EPISODIC_LANE,
    KB_LANES,
    PROCEDURAL_LANE,
    PROFILE_LANE,
    RAW_LANE,
    SEMANTIC_LANE,
    TRACE_LANE,
    WIKI_LANE,
)
from uma.retrieve.rlm.intent import QueryIntent, classify_query_intent

RetrievalProduct = Literal["context", "memory"]

_LANE_TO_DOMAIN = {
    RAW_LANE: "kb_doc",
    WIKI_LANE: "kb_doc",
    SEMANTIC_LANE: "kb_doc",
    PROFILE_LANE: "user_profile",
    PROCEDURAL_LANE: "procedural",
}

_HISTORY_MARKERS = (
    "decision",
    "decide",
    "history",
    "earlier",
    "previous",
    "last time",
    "what happened",
    "yesterday",
    "last week",
    "last month",
    "last session",
    "ago",
    "when did",
    "used to",
    "recall",
    "told me",
    "in the past",
)


@dataclass(frozen=True)
class RetrievalPlan:
    """Small canonical lane-selection plan for one retrieval product call."""

    product: RetrievalProduct
    query_text: str
    query_intent: str
    memory_intent: Optional[str]
    requested_lanes: Tuple[str, ...]
    participating_lanes: Tuple[str, ...]
    excluded_lanes: Tuple[Dict[str, str], ...]
    active_domains: Tuple[str, ...]
    available_lanes: Tuple[str, ...]
    requires_compiled_memory: bool
    evidence_expansion: bool

    def to_trace(self) -> Dict[str, Any]:
        return {
            "event": "lane_plan",
            "product": self.product,
            "query_intent": self.query_intent,
            "memory_intent": self.memory_intent,
            "requested_lanes": list(self.requested_lanes),
            "participating_lanes": list(self.participating_lanes),
            "excluded_lanes": [dict(item) for item in self.excluded_lanes],
            "active_domains": list(self.active_domains),
            "available_lanes": list(self.available_lanes),
            "requires_compiled_memory": self.requires_compiled_memory,
            "evidence_expansion": self.evidence_expansion,
        }


def build_retrieval_plan(
    *,
    product: RetrievalProduct,
    query_text: str,
    available_lanes: Sequence[str],
    lane_filter: Optional[Sequence[str]] = None,
    memory_intent: Optional[str] = None,
) -> RetrievalPlan:
    """Build the single canonical lane plan for one retrieval call.

    `wiki` is planned as a first-class compiled-memory lane. Whether the current
    runtime serves that lane from a dedicated compiled-memory source or from the
    existing retrievable document/evidence stack is a runtime capability detail,
    not a second planner path.
    """
    query = str(query_text or "").strip()
    if not query:
        raise ValueError("build_retrieval_plan requires a non-empty query_text")

    query_intent = classify_query_intent(query).value
    supported = _normalize_lanes(available_lanes)
    if product == "context":
        requested = _context_requested_lanes(
            query_text=query,
            query_intent=query_intent,
            lane_filter=lane_filter,
        )
    elif product == "memory":
        requested = _memory_requested_lanes(
            query_text=query,
            query_intent=query_intent,
            memory_intent=memory_intent,
        )
    else:
        raise ValueError(f"Unsupported retrieval product: {product!r}")

    participating = tuple(lane for lane in requested if lane in supported)
    active_domains = _domains_for_lanes(participating)
    excluded = _build_excluded_lanes(
        product=product,
        query_intent=query_intent,
        requested=requested,
        participating=participating,
        supported=supported,
        lane_filter=lane_filter,
    )
    return RetrievalPlan(
        product=product,
        query_text=query,
        query_intent=query_intent,
        memory_intent=str(memory_intent or "").strip() or None,
        requested_lanes=requested,
        participating_lanes=participating,
        excluded_lanes=excluded,
        active_domains=active_domains,
        available_lanes=supported,
        requires_compiled_memory=(product == "memory"),
        evidence_expansion=(product == "memory" and RAW_LANE in participating),
    )


def _context_requested_lanes(
    *,
    query_text: str,
    query_intent: str,
    lane_filter: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    explicit = _normalize_lanes(lane_filter or ())
    if explicit:
        return explicit
    if query_intent == QueryIntent.PERSONAL.value:
        return (PROFILE_LANE, PROCEDURAL_LANE, SEMANTIC_LANE, EPISODIC_LANE)
    if query_intent == QueryIntent.MIXED.value:
        return (RAW_LANE, SEMANTIC_LANE, PROFILE_LANE, PROCEDURAL_LANE)
    if _is_history_query(query_text):
        return (RAW_LANE, EPISODIC_LANE, SEMANTIC_LANE)
    return (RAW_LANE, SEMANTIC_LANE)


def _memory_requested_lanes(
    *,
    query_text: str,
    query_intent: str,
    memory_intent: Optional[str],
) -> Tuple[str, ...]:
    requested = [WIKI_LANE, RAW_LANE, SEMANTIC_LANE]
    normalized_memory_intent = str(memory_intent or "").strip().lower()
    if (
        (query_intent == QueryIntent.PERSONAL.value and not _is_history_query(query_text))
        or "profile" in normalized_memory_intent
    ):
        requested.append(PROFILE_LANE)
    if (
        normalized_memory_intent in {"continuity", "history", "decision"}
        or _is_history_query(query_text)
    ):
        requested.append(EPISODIC_LANE)
    if "procedure" in normalized_memory_intent or "skill" in normalized_memory_intent:
        requested.append(PROCEDURAL_LANE)
    return _normalize_lanes(requested)


def _build_excluded_lanes(
    *,
    product: RetrievalProduct,
    query_intent: str,
    requested: Sequence[str],
    participating: Sequence[str],
    supported: Sequence[str],
    lane_filter: Optional[Sequence[str]],
) -> Tuple[Dict[str, str], ...]:
    explicit_filter = bool(_normalize_lanes(lane_filter or ()))
    requested_set = set(requested)
    participating_set = set(participating)
    supported_set = set(supported)
    excluded = []
    for lane in KB_LANES:
        if lane in participating_set:
            continue
        if lane in requested_set and lane not in supported_set:
            reason = "lane_not_available_in_runtime"
        elif explicit_filter:
            reason = "excluded_by_explicit_lane_filter"
        elif lane == WIKI_LANE and product == "context":
            reason = "wiki_not_enabled_by_default_for_context"
        elif lane == PROFILE_LANE and query_intent == QueryIntent.TOPICAL.value:
            reason = "profile_not_selected_for_topical_query"
        elif lane == TRACE_LANE:
            reason = "trace_is_debug_metadata_not_a_retrieval_lane"
        else:
            reason = f"not_selected_by_{product}_lane_policy"
        excluded.append({"lane": lane, "reason": reason})
    return tuple(excluded)


def _domains_for_lanes(lanes: Sequence[str]) -> Tuple[str, ...]:
    domains = []
    seen = set()
    for lane in lanes:
        domain = _LANE_TO_DOMAIN.get(lane)
        if not domain or domain in seen:
            continue
        seen.add(domain)
        domains.append(domain)
    return tuple(domains)


def _normalize_lanes(lanes: Sequence[str]) -> Tuple[str, ...]:
    normalized = []
    seen = set()
    for lane in lanes or ():
        candidate = str(lane or "").strip().lower()
        if not candidate or candidate in seen or candidate not in KB_LANES:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return tuple(normalized)


def _is_history_query(query_text: str) -> bool:
    lowered = str(query_text or "").strip().lower()
    return any(marker in lowered for marker in _HISTORY_MARKERS)
