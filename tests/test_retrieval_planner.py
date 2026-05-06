from __future__ import annotations

from uma.common.storage_metadata import (
    EPISODIC_LANE,
    PROCEDURAL_LANE,
    PROFILE_LANE,
    RAW_LANE,
    SEMANTIC_LANE,
    WIKI_LANE,
)
from uma.retrieve.planner import build_retrieval_plan


def _excluded_reason(plan, lane: str) -> str:
    for item in plan.excluded_lanes:
        if item["lane"] == lane:
            return item["reason"]
    raise AssertionError(f"lane {lane!r} not found in excluded_lanes")


def test_context_plan_defaults_to_evidence_lanes_for_topical_queries() -> None:
    plan = build_retrieval_plan(
        product="context",
        query_text="How should a zero-trust rollout be structured?",
        available_lanes=[RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
    )

    assert plan.participating_lanes == (RAW_LANE, SEMANTIC_LANE)
    assert plan.active_domains == ("kb_doc",)
    assert _excluded_reason(plan, PROFILE_LANE) == "profile_not_selected_for_topical_query"
    assert _excluded_reason(plan, WIKI_LANE) == "wiki_not_enabled_by_default_for_context"


def test_context_plan_uses_profile_lane_without_confusing_it_with_user_kb() -> None:
    plan = build_retrieval_plan(
        product="context",
        query_text="What do I like?",
        available_lanes=[RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
    )

    assert plan.participating_lanes == (PROFILE_LANE, PROCEDURAL_LANE)
    assert plan.active_domains == ("user_profile", "procedural")


def test_memory_plan_prefers_wiki_but_surfaces_runtime_unavailability_explicitly() -> None:
    plan = build_retrieval_plan(
        product="memory",
        query_text="What decisions did I make earlier about the rollout?",
        available_lanes=[RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
        memory_intent="continuity",
    )

    assert plan.requested_lanes[:4] == (WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE)
    assert plan.participating_lanes == (RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE)
    assert plan.requires_compiled_memory is True
    assert plan.evidence_expansion is True
    assert _excluded_reason(plan, WIKI_LANE) == "lane_not_available_in_runtime"


def test_memory_plan_uses_wiki_when_runtime_supports_compiled_memory_lane() -> None:
    plan = build_retrieval_plan(
        product="memory",
        query_text="What decisions did I make earlier about the rollout?",
        available_lanes=[WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
        memory_intent="continuity",
    )

    assert plan.participating_lanes[:4] == (WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE)
    assert plan.requires_compiled_memory is True
    assert plan.evidence_expansion is True


def test_memory_plan_uses_profile_lane_only_for_profile_style_requests() -> None:
    plan = build_retrieval_plan(
        product="memory",
        query_text="What do I like?",
        available_lanes=[WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
        memory_intent="profile",
    )

    assert plan.participating_lanes == (
        WIKI_LANE,
        RAW_LANE,
        SEMANTIC_LANE,
        PROFILE_LANE,
    )
    assert plan.active_domains == ("kb_doc", "user_profile")


def test_planner_trace_stays_backend_agnostic() -> None:
    plan = build_retrieval_plan(
        product="memory",
        query_text="What decisions did I make earlier about the rollout?",
        available_lanes=[WIKI_LANE, RAW_LANE, SEMANTIC_LANE, EPISODIC_LANE, PROCEDURAL_LANE, PROFILE_LANE],
        memory_intent="continuity",
    )

    trace = plan.to_trace()

    assert "participating_lanes" in trace
    assert "active_domains" in trace
    assert "fusion_strategy" not in trace
    assert "top_k_dense" not in trace
    assert "top_k_sparse" not in trace
    assert "backend" not in trace
