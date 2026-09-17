"""RLM step-loop action scoping: which owner_type each _decide_* action carries.

`_scopes_for_action` (controller.py) narrows execution to the action's own
`owner_type` when it is set, and to every scope on the request when it is
unset. The step-loop `_decide_*` functions in `decisions.py` historically
pinned `owner_type=pack.owner_type` on every action they built -
`pack.owner_type` being `scopes[0].owner_type`, an ordering accident rather
than a statement about where the relevant data lives (see the graph fix in
this same series). This file covers the sibling functions: semantic,
chunk fallback, episodic clusters, and the zero-yield fallback.

Baseline retrieval already iterates every scope on the request for facts and
chunks (`RLMController._baseline_retrieval`), so leaving these step-loop
actions unscoped widens nothing new - it makes the loop consistent with what
baseline already does, and each per-scope call remains independently
ownership-filtered at the store layer.
"""
from __future__ import annotations
from datetime import datetime, timezone
from uma.common.types.types_fact import Fact
from uma.retrieve.rlm.decisions import (
    _decide_chunk_fallback,
    _decide_episodic_clusters,
    _decide_semantic,
    _decide_zero_yield_fallback,
)


def _fact(predicate: str = "SEGMENTED_INTO", text: str = "iam vpc kms") -> Fact:
    return Fact(
        id=f"fact_{predicate}",
        subject="doc:1",
        predicate=predicate,
        object="network segmentation",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        source_ids=["chunk_1"],
        meta={"domain": "kb_doc", "fact_text": text, "source_path": "kb/doc.md"},
        owner_type="agent",
        owner_id="agent:test",
        salience=0.0,
        confidence=0.7,
    )


class _CoverageNeedsSemantic:
    needs_semantic = True
    needs_clusters = False


class _CoverageNeedsClusters:
    needs_semantic = False
    needs_clusters = True


def _make_pack(**overrides: object) -> object:
    class _Pack:
        graph = []
        facts = []
        chunks = []
        episodes = []
        steps = []
        query_text = "query"
        intent = "topical"
        # scopes[0] happened to be the agent scope - the ordinary case for a
        # request carrying both an agent and a user scope.
        owner_type = "agent"
        owner_id = "agent:test"
        user_id = "user:123"
        active_domains = ["kb_doc"]
        trace_id = "trace-1"
        chunk_fallback_used = False

        def get_predicate_offset(self, predicate: str) -> int:
            return 0

        def bump_predicate_offset(self, predicate: str, amount: int) -> None:
            return None

    pack = _Pack()
    for key, value in overrides.items():
        setattr(pack, key, value)
    return pack


def test_decide_semantic_with_facts_does_not_pin_owner_type() -> None:
    pack = _make_pack(facts=[_fact()])
    actions = _decide_semantic(pack, _CoverageNeedsSemantic(), cfg={"max_items_per_type": 30})
    assert actions, "expected a fetch_more_facts or search_semantic action"
    assert all(a.owner_type is None for a in actions), (
        "semantic action pinned owner_type to scopes[0]; user-owned facts in "
        "a multi-scope request become unreachable in the step loop"
    )


def test_decide_semantic_with_no_facts_does_not_pin_owner_type() -> None:
    pack = _make_pack(facts=[])
    actions = _decide_semantic(pack, _CoverageNeedsSemantic(), cfg={"max_items_per_type": 30})
    assert actions, "expected a search_semantic fallback action"
    assert all(a.owner_type is None for a in actions)


def test_decide_chunk_fallback_does_not_pin_owner_type() -> None:
    pack = _make_pack(chunks=[])
    actions = _decide_chunk_fallback(pack, cfg={"chunk_fallback_enabled": True})
    assert actions, "expected a search_chunks fallback action"
    assert all(a.owner_type is None for a in actions)


def test_decide_episodic_clusters_lite_path_does_not_pin_owner_type() -> None:
    pack = _make_pack()
    actions = _decide_episodic_clusters(
        pack, _CoverageNeedsClusters(), cfg={"max_items_per_type": 30, "episodic_clustering_available": False}
    )
    assert actions, "expected a search_episodic action"
    assert all(a.owner_type is None for a in actions)


def test_decide_episodic_clusters_enterprise_path_does_not_pin_owner_type() -> None:
    pack = _make_pack(steps=[{"step": 1}, {"step": 2}])
    actions = _decide_episodic_clusters(
        pack, _CoverageNeedsClusters(), cfg={"max_items_per_type": 30, "episodic_clustering_available": True}
    )
    assert actions, "expected fetch_episode_clusters plus search_episodic"
    assert all(a.owner_type is None for a in actions)


def test_decide_zero_yield_fallback_facts_path_does_not_pin_owner_type() -> None:
    pack = _make_pack(
        steps=[{"event": "action_result", "action": "fetch_more_facts", "store": "facts", "novelty": 0}],
    )
    decision = _decide_zero_yield_fallback(
        pack, cfg={"max_items_per_type": 30, "chunk_fallback_enabled": True}
    )
    assert decision is not None
    assert all(a.owner_type is None for a in decision.actions)


def test_decide_zero_yield_fallback_chunks_path_does_not_pin_owner_type() -> None:
    pack = _make_pack(
        active_domains=["kb_doc"],
        steps=[{"event": "action_result", "action": "expand_graph", "store": "graph", "novelty": 0}],
    )
    decision = _decide_zero_yield_fallback(
        pack, cfg={"max_items_per_type": 30, "chunk_fallback_enabled": True}
    )
    assert decision is not None
    assert all(a.owner_type is None for a in decision.actions)
