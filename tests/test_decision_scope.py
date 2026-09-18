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
from typing import Any, Literal
from uma.common.types import RuntimeContext
from uma.retrieve.rlm.context_pack import ContextPack
from uma.retrieve.rlm.controller import RLMController
from uma.retrieve.rlm.request import RetrievalRequest
from uma.retrieve.rlm.decisions import (
    SearchSemanticAction,
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


def test_kept_items_excludes_ids_dropped_by_merge_truncation() -> None:
    """`_kept_items` must exclude ids that `_merge_unique` truncated away.

    Marking a dropped id "seen" makes it permanently unreachable: a later
    fetch that would legitimately re-surface it reports novelty=0 even
    though the pack never actually received it.
    """
    from uma.retrieve.rlm.controller import _kept_items, _merge_unique

    class _Item:
        def __init__(self, item_id: str) -> None:
            self.id = item_id

    existing = [_Item("A"), _Item("B")]
    research = [_Item("C"), _Item("D")]  # only "C" fits under limit=3
    merged = _merge_unique(existing, research, limit=3)
    assert [it.id for it in merged] == ["A", "B", "C"]

    kept = _kept_items(research, merged)
    assert [it.id for it in kept] == ["C"], (
        "dropped id 'D' must not be passed to apply_novelty - marking it "
        "seen would blacklist it from ever surfacing, despite never "
        "entering the pack"
    )


# ---------------------------------------------------------------------------
# Scope fanout: what the unpinned actions above actually do at execution time.
#
# The tests above assert only that an action carries `owner_type=None`, which
# is what makes `_scopes_for_action` return every scope on the request. This
# module's docstring then claims the safety property that makes that widening
# sound - "each per-scope call remains independently ownership-filtered at the
# store layer" - and nothing exercised it. These tests do: they drive an
# unpinned action through `_scopes_for_action` and `_execute_action` with a
# recording environment, and assert on the calls actually made.
# ---------------------------------------------------------------------------


class _RecordingEnv:
    """Environment stub that records the owner pair of every execute_action call.

    Returns per-scope-distinguishable items so cross-scope leakage is visible
    in the merged pack, not just in the call log.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def execute_action(self, **kwargs: Any) -> list[Fact]:
        owner_type = str(kwargs.get("owner_type") or "")
        owner_id = str(kwargs.get("owner_id") or "")
        self.calls.append((owner_type, owner_id))
        assert owner_type in ("agent", "user"), f"unexpected scope owner_type {owner_type!r}"
        owner: Literal["agent", "user"] = "agent" if owner_type == "agent" else "user"
        return [_scoped_fact(owner, owner_id)]


def _scoped_fact(owner_type: Literal["agent", "user"], owner_id: str) -> Fact:
    """A fact that names the scope it was returned for, so leakage is traceable."""
    return Fact(
        id=f"fact_{owner_type}",
        subject=owner_id,
        predicate="SEGMENTED_INTO",
        object=f"object for {owner_type}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        source_ids=["chunk_1"],
        meta={"domain": "kb_doc", "fact_text": f"fact owned by {owner_type}"},
        owner_type=owner_type,
        owner_id=owner_id,
        salience=0.0,
        confidence=0.7,
    )


def _two_scope_request() -> RetrievalRequest:
    """A request built the way production builds one: agent scope + user scope."""
    return RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant_test",
            agent_id="agent:test",
            request_id="req-1",
            user_id="user:123",
        )
    )


def _fanout_pack() -> ContextPack:
    return ContextPack(
        user_id="user:123",
        query_text="network segmentation",
        owner_type="agent",
        owner_id="agent:test",
        active_lanes=["semantic"],
        active_domains=["kb_doc"],
    )


async def _run_fanout(action: SearchSemanticAction) -> tuple[_RecordingEnv, ContextPack]:
    env = _RecordingEnv()
    controller = RLMController(llm=None, env=env)
    request = _two_scope_request()
    pack = _fanout_pack()
    await controller._execute_action(
        action=action,
        request=request,
        pack=pack,
        query_embedding=[0.0, 0.0, 0.0],
        scopes=list(request.scopes),
        trace_id="trace-1",
        step=1,
    )
    return env, pack


def test_unpinned_action_queries_each_scope_with_its_own_owner_pair() -> None:
    """An unpinned action must produce one store call per scope, correctly paired.

    This is the property the module docstring asserts and the unpinning work
    depends on. If `_scopes_for_action` ever collapsed to a single scope, or
    the loop reused one owner pair for every call, retrieval would silently
    stop reaching one of the two owners.
    """
    import asyncio

    env, _pack = asyncio.run(_run_fanout(SearchSemanticAction(k=10, owner_type=None)))

    assert env.calls == [("agent", "agent:test"), ("user", "user:123")], (
        f"expected one call per scope with its own owner pair, got {env.calls}"
    )


def test_unpinned_action_merges_both_scopes_without_cross_contamination() -> None:
    """Per-scope results merge into the pack, each still carrying its own owner."""
    import asyncio

    _env, pack = asyncio.run(_run_fanout(SearchSemanticAction(k=10, owner_type=None)))

    merged = {(f.owner_type, f.owner_id) for f in pack.facts}
    assert merged == {("agent", "agent:test"), ("user", "user:123")}, (
        f"both scopes' items must survive the merge unmixed, got {merged}"
    )


def test_pinned_action_queries_only_the_scope_it_names() -> None:
    """Regression: pinning still narrows to one scope.

    Guards the opposite failure from the two above - a fanout fix that widened
    pinned actions too would make the personal graph branch query the agent
    scope it deliberately excludes.
    """
    import asyncio

    env, _pack = asyncio.run(_run_fanout(SearchSemanticAction(k=10, owner_type="user")))

    assert env.calls == [("user", "user:123")], (
        f"a pinned action must reach exactly the scope it names, got {env.calls}"
    )
