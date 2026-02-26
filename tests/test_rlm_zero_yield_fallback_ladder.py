from __future__ import annotations

from uma.core.retrieval.rlm.decisions import deterministic_decision


class _Coverage:
    needs_semantic = True
    needs_clusters = False


def test_fallback_ladder_fetch_more_facts_zero_yield_prefers_chunks_once() -> None:
    class _Pack:
        facts = ["placeholder"]  # avoid semantic branch using empty-facts shortcut
        chunks = []
        episodes = []
        graph = []
        steps = [
            {
                "event": "action_result",
                "action": "fetch_more_facts",
                "store": "facts",
                "returned": 0,
                "novelty": 0,
            }
        ]
        owner_type = "agent"
        owner_id = "agent:test"
        user_id = "user:123"
        query_text = "cloud security architecture"
        intent = "topical"
        active_domains = ["kb_doc"]
        chunk_fallback_used = False

        def get_predicate_offset(self, _p: str) -> int:
            return 0

        def bump_predicate_offset(self, _p: str, _d: int) -> int:
            return 0

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={"max_items_per_type": 10, "chunk_fallback_k_multiplier": 2, "chunk_fallback_enabled": True},
    )
    assert decision is not None
    assert decision.actions
    assert decision.actions[0].action == "search_chunks"
    assert decision.actions[0].k and decision.actions[0].k > 10


def test_fallback_ladder_fetch_more_facts_zero_yield_then_broaden_semantic() -> None:
    class _Pack:
        facts = ["placeholder"]
        chunks = []
        episodes = []
        graph = []
        steps = [
            {
                "event": "action_result",
                "action": "search_chunks",
                "store": "chunks",
                "returned": 0,
                "novelty": 0,
            },
            {
                "event": "action_result",
                "action": "fetch_more_facts",
                "store": "facts",
                "returned": 0,
                "novelty": 0,
            },
        ]
        owner_type = "agent"
        owner_id = "agent:test"
        user_id = "user:123"
        query_text = "cloud security architecture"
        intent = "topical"
        active_domains = ["kb_doc"]
        chunk_fallback_used = False

        def get_predicate_offset(self, _p: str) -> int:
            return 0

        def bump_predicate_offset(self, _p: str, _d: int) -> int:
            return 0

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={"max_items_per_type": 10, "chunk_fallback_k_multiplier": 2, "chunk_fallback_enabled": True},
    )
    assert decision is not None
    assert decision.actions
    assert decision.actions[0].action == "search_semantic"
