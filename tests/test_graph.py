"""Graph lane: core operations, lane retrieval, pipeline integration, entity seeding.

Covers GraphCore node/edge operations, graph lane retrieval through the
retrieval pipeline, graph updates from process_turn, and RLM entity seeding.
"""
from __future__ import annotations
from datetime import datetime, timezone
from tests.helpers.graph_adapter import RecordingGraphAdapter
from tests.helpers.runtime import TEST_AGENT_ID, init_uma_for_tests
from uma.common.results import ContextBundle
from uma.common.types import Episode
from uma.common.types.types_fact import Fact
from uma.memory.graph.core import GraphCore
from uma.retrieve.rlm.decisions import deterministic_decision
import pytest
import pytest_asyncio

AGENT_ID = TEST_AGENT_ID

# ── test_graph_core ──────────────────────────────────────────







def test_insert_fact_triplet_sanitizes_predicate_and_stamps_owner():
    adapter = RecordingGraphAdapter()
    core = GraphCore(adapter)

    core.insert_fact_triplet(
        fact_id="f1",
        subject="user:u1",
        predicate="Bad-REL!!",  # must be sanitized
        object="tea",
        tenant_id="tenant-1",
        owner_type="user",
        owner_id="user:u1",
        source_chunk_id="c1",
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
        scope_model_version="v2",
    )

    assert adapter.queries, "Expected Cypher execution"

    cypher, params = adapter.queries[0]

    assert "BAD_REL" in cypher or "RELATES_TO" in cypher
    assert params.get("tenant_id") == "tenant-1"
    assert params.get("owner_type") == "user"
    assert params.get("owner_id") == "user:u1"
    assert params.get("fact_id") == "f1"
    assert params.get("source_chunk_id") == "c1"
    assert params.get("scope_model_version") == "v2"


def test_episode_edges_have_ownership():
    """add_episode delegates to GraphUpdater.add_episode_node; verify ownership stamping."""
    adapter = RecordingGraphAdapter()
    core = GraphCore(adapter)

    ep = Episode(
        id="ep1",
        timestamp=datetime.utcnow(),
        summary="Test episode",
        user_id="u1",
        tenant_id="tenant-1",
        owner_type="user",
        owner_id="user:u1",
    )

    core.add_episode(ep)

    assert adapter.queries, "No Cypher executed for episode insertion"

    matched = False
    for cypher, params in adapter.queries:
        if "HAS_EPISODE" in cypher:
            matched = True
            assert params.get("tenant_id") == "tenant-1"
            assert params.get("owner_type") == "user"
            assert params.get("owner_id") == "user:u1"
            assert params.get("scope_model_version") == "v2"
            break

    assert matched, "Expected HAS_EPISODE relationship write"


def test_neighbors_enforce_tenant_and_owner_scope_in_query() -> None:
    adapter = RecordingGraphAdapter()
    core = GraphCore(adapter)

    out = core.neighbors(
        user_id="user:u1",
        node_id="node-1",
        tenant_id="tenant-1",
        owner_type="workspace",
        owner_id="workspace:alpha",
        predicate_scope=["likes"],
        depth=2,
        k=3,
    )
    assert out == []
    assert adapter.queries
    cypher, params = adapter.queries[0]
    assert "r.tenant_id = $tenant_id" in cypher
    assert params["tenant_id"] == "tenant-1"
    assert params["owner_type"] == "workspace"
    assert params["owner_id"] == "workspace:alpha"


def test_neighbors_excludes_quarantined_fact_nodes_in_query() -> None:
    """Read-time equivalent of every SQL store's `AND quarantined_at IS
    NULL` (ticket 14). Non-Fact nodes never carry the property at all, so
    `IS NULL` passes them through unaffected — verified by the second
    assertion below with a plain `RecordingGraphAdapter` result."""
    adapter = RecordingGraphAdapter()
    adapter.next_results.append([{"node": {"id": "user:u1"}, "labels": ["User"], "properties": {}}])
    core = GraphCore(adapter)

    out = core.neighbors(
        user_id="user:u1",
        node_id="node-1",
        tenant_id="tenant-1",
        owner_type="user",
        owner_id="user:u1",
    )

    cypher, _params = adapter.queries[0]
    assert "m.quarantined_at IS NULL" in cypher
    assert out == [{"node": {"id": "user:u1"}, "labels": ["User"], "properties": {}}]


def test_set_fact_quarantine_scopes_the_update_to_owner_and_reports_match() -> None:
    adapter = RecordingGraphAdapter()
    adapter.next_results.append([{"id": "fact_1"}])
    core = GraphCore(adapter)

    updated = core.set_fact_quarantine(
        "fact_1",
        "2026-09-18T00:00:00+00:00",
        tenant_id="tenant-1",
        owner_type="user",
        owner_id="user:u1",
    )

    assert updated is True
    cypher, params = adapter.queries[0]
    assert "MATCH (f:Fact {id: $fact_id})" in cypher
    assert "SET f.quarantined_at = $quarantined_at" in cypher
    assert params["fact_id"] == "fact_1"
    assert params["quarantined_at"] == "2026-09-18T00:00:00+00:00"
    assert params["tenant_id"] == "tenant-1"
    assert params["owner_type"] == "user"
    assert params["owner_id"] == "user:u1"


def test_set_fact_quarantine_reinstate_clears_the_property() -> None:
    adapter = RecordingGraphAdapter()
    adapter.next_results.append([])  # no node in this scope -> no match
    core = GraphCore(adapter)

    updated = core.set_fact_quarantine(
        "fact_missing",
        None,
        tenant_id="tenant-1",
        owner_type="user",
        owner_id="user:u1",
    )

    assert updated is False
    _cypher, params = adapter.queries[0]
    assert params["quarantined_at"] is None


def test_resolve_nodes_enforces_tenant_and_owner_scope() -> None:
    adapter = RecordingGraphAdapter()
    adapter.next_results.append([{"node_id": "workspace-node"}])
    core = GraphCore(adapter)

    out = core.resolve_nodes(
        tenant_id="tenant-1",
        owner_type="workspace",
        owner_id="workspace:alpha",
        names=["Workspace Node"],
        domain_scope=["kb_doc"],
        limit=5,
    )
    assert out == ["workspace-node"]
    cypher, params = adapter.queries[0]
    assert "r.tenant_id = $tenant_id" in cypher
    assert params["tenant_id"] == "tenant-1"
    assert params["owner_type"] == "workspace"
    assert params["owner_id"] == "workspace:alpha"


def test_raw_graph_query_is_gated_in_normal_runtime_flow() -> None:
    core = GraphCore(RecordingGraphAdapter())
    with pytest.raises(RuntimeError, match="unsafe"):
        core.query("MATCH (n) RETURN n", params={})


def test_insert_fact_triplet_rejects_system_scope() -> None:
    core = GraphCore(RecordingGraphAdapter())
    ok = core.insert_fact_triplet(
        fact_id="f-system",
        subject="ops",
        predicate="RUNS",
        object="job",
        tenant_id="tenant-1",
        owner_type="system",
        owner_id="system:ops",
        source_chunk_id="c1",
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
    )
    assert ok is False


def test_add_fact_skips_already_quarantined_fact() -> None:
    """Ticket 14: the turn-path (`GraphCore.add_facts`) and maintenance-path
    (`uma/common/maintenance.py`) callers both go through
    `GraphUpdater.add_fact`, which previously wrote every fact regardless of
    `quarantined_at` — unlike the document-ingest path's `add_facts_batch`,
    which already skipped them. A fact already quarantined by
    `SemanticCore.ingest`'s write-time scan must never reach the graph."""
    adapter = RecordingGraphAdapter()
    core = GraphCore(adapter)
    fact = Fact(
        id="fact_q1",
        subject="user:u1",
        predicate="prefers",
        object="chocolate",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tenant_id="tenant-1",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.0,
        quarantined_at=datetime.now(timezone.utc),
    )

    core.add_facts([fact])

    assert adapter.queries == [], "a quarantined fact must never reach insert_fact_triplet"


def test_add_fact_writes_a_clean_fact_normally() -> None:
    """Companion to the quarantine-skip test: an ordinary, non-quarantined
    fact must still reach the graph unaffected by the new check."""
    adapter = RecordingGraphAdapter()
    core = GraphCore(adapter)
    fact = Fact(
        id="fact_clean",
        subject="user:u1",
        predicate="prefers",
        object="tea",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tenant_id="tenant-1",
        owner_type="user",
        owner_id="user:u1",
        trust_score=0.7,
        quarantined_at=None,
    )

    core.add_facts([fact])

    assert adapter.queries, "a clean fact must still reach insert_fact_triplet"


# ── test_graph_lane_retrieval ──────────────────────────────────────────





@pytest_asyncio.fixture
async def uma_graph(tmp_path):
    """UMAMemory instance with RecordingGraphAdapter wired in."""
    mem = await init_uma_for_tests(
        tmp_path,
        graph_backend="tests.helpers.graph_adapter:RecordingGraphAdapter",
        graph_config={},
    )
    try:
        yield mem
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass


@pytest_asyncio.fixture
async def uma_no_graph(tmp_path):
    """UMAMemory instance with graph backend disabled."""
    mem = await init_uma_for_tests(tmp_path, graph_backend="disabled")
    try:
        yield mem
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_ingest_writes_graph_edges(uma_graph, tmp_path):
    """Graph adapter must receive Cypher writes after document ingest."""
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "The UMA memory system provides long-term knowledge retention for AI agents. "
        "It supports document ingestion pipelines that parse, chunk, embed, and index content. "
        "Fact extraction uses LLM prompts to derive structured subject-predicate-object triplets. "
        "Graph edges connect extracted facts enabling relational and temporal retrieval. "
        "The graph lane integrates with the RLM controller for structured context assembly."
    )

    adapter: RecordingGraphAdapter = uma_graph.graph_core.adapter
    queries_before = len(adapter.queries)

    report = await uma_graph.ingest_document(
        str(doc), owner_type="agent", owner_id="agent-default",
    )

    assert report.chunks_created > 0, "Expected at least one chunk to be created"
    assert len(adapter.queries) > queries_before, (
        "Expected graph Cypher queries after ingest; none recorded. "
        "Verify update_graph is called during fact extraction."
    )


@pytest.mark.asyncio
async def test_retrieve_context_with_graph_enabled(uma_graph, tmp_path):
    """retrieve_context must return a ContextBundle with a `graph` attribute when graph is enabled."""
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "UMA supports structured graph memory via the GraphCore subsystem. "
        "Facts are extracted from documents and stored as subject-predicate-object triplets. "
        "The graph lane enables relational and temporal retrieval across documents and episodes. "
        "Graph edges carry full provenance: fact_id, owner_type, owner_id, source_chunk_id, timestamps."
    )
    await uma_graph.ingest_document(str(doc), owner_type="agent", owner_id="agent-default")

    result = await uma_graph.retrieve_context(query_text="How does UMA graph memory work?", user_id="user-test", agent_id=AGENT_ID)

    assert isinstance(result, ContextBundle), "retrieve_context must return a ContextBundle"
    assert hasattr(result, "graph"), "ContextBundle must include a `graph` attribute when graph is enabled"


@pytest.mark.asyncio
async def test_retrieve_context_with_graph_disabled(uma_no_graph, tmp_path):
    """retrieve_context must not crash and must expose the standard attributes when graph is disabled."""
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "UMA is a modular memory runtime. It runs with or without a graph backend. "
        "Disabling graph still allows chunk and fact retrieval."
    )
    await uma_no_graph.ingest_document(str(doc), owner_type="agent", owner_id="agent-default")

    result = await uma_no_graph.retrieve_context(query_text="What is UMA?", user_id="user-test", agent_id=AGENT_ID)

    assert isinstance(result, ContextBundle), "retrieve_context must return a ContextBundle even without graph"
    for attr in ("chunks", "facts", "episodic"):
        assert hasattr(result, attr), f"Expected attribute '{attr}' on ContextBundle"


@pytest.mark.asyncio
async def test_graph_disabled_graph_core_is_none(uma_no_graph):
    """When graph backend is 'disabled', graph_core must be None."""
    assert uma_no_graph.graph_core is None, (
        "graph_core should be None when graph_backend='disabled'"
    )


# ── test_pipeline_graph ──────────────────────────────────────────





@pytest.mark.asyncio
async def test_pipeline_updates_graph_with_facts_and_temporal_links(tmp_path):
    mem = await init_uma_for_tests(
        tmp_path,
        graph_backend="tests.helpers.graph_adapter:RecordingGraphAdapter",
        graph_config={},
    )
    try:
        await mem.process_turn(
            user_id="user:u1",
            user_msg="I like sushi.",
            assistant_reply="Good to know.",
            session_id="session-a",
            agent_id=AGENT_ID,
        )
        await mem.process_turn(
            user_id="user:u1",
            user_msg="I also like pizza.",
            assistant_reply="Pizza is delicious.",
            session_id="session-a",
            agent_id=AGENT_ID,
        )

        adapter = getattr(mem.graph_core, "adapter", None)
        queries = getattr(adapter, "queries", None)
        assert isinstance(queries, list) and queries, "expected graph adapter to record cypher queries"

        cyphers = [q for q, _params in queries]
        assert any("HAS_EPISODE" in c for c in cyphers)
        assert any("MERGE (f:Fact" in c or "MERGE (f:Fact" in c.replace("\n", " ") for c in cyphers)
        assert any("PRECEDES" in c for c in cyphers), "expected temporal PRECEDES/FOLLOWS edges"
        assert any((params or {}).get("tenant_id") == "default" for _c, params in queries)
        assert any((params or {}).get("owner_type") == "user" for _c, params in queries)
        assert any((params or {}).get("scope_model_version") == "v2" for _c, params in queries)
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass


# ── test_rlm_graph_entity_seeding ──────────────────────────────────────────





def _kb_fact(*, predicate: str = "SEGMENTED_INTO", text: str = "iam vpc kms") -> Fact:
    return Fact(
        id="fact_test",
        subject="cloud_security_architecture",
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


class _Coverage:
    needs_semantic = False
    needs_clusters = False


def test_topical_graph_expansion_seeds_from_entities_not_user_id() -> None:
    class _Pack:
        graph = []
        facts = [_kb_fact()]
        chunks = []
        steps = []
        query_text = "How should IAM and VPC be used in a multi-tier architecture?"
        intent = "topical"
        owner_type = "agent"
        owner_id = "agent:test"
        user_id = "user:123"

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={
            "chunk_fallback_enabled": False,
            "graph_predicate_limit": 2,
            "graph_expansion_available": True,
        },
    )
    assert decision is not None
    actions = [a for a in decision.actions if a.action == "expand_graph"]
    assert actions, "expected topical graph expansion actions"
    assert all(a.subject != "user:123" for a in actions)
    assert actions[0].subject in {"iam", "vpc"}


def test_personal_graph_expansion_keeps_user_anchor() -> None:
    class _Pack:
        graph = []
        facts = [
            Fact(
                id="fact_test",
                subject="user:123",
                predicate="LIKES",
                object="sushi",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                source_ids=[],
                meta={"domain": "user_profile", "fact_text": "user likes sushi"},
                owner_type="user",
                owner_id="user:123",
                salience=0.0,
                confidence=0.7,
            )
        ]
        chunks = []
        steps = []
        query_text = "What do I like?"
        intent = "personal"
        owner_type = "user"
        owner_id = "user:123"
        user_id = "user:123"

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={
            "chunk_fallback_enabled": False,
            "graph_predicate_limit": 2,
            "next_predicate_scope": lambda _p, _limit: ["LIKES"],
            "graph_expansion_available": True,
        },
    )
    assert decision is not None
    actions = [a for a in decision.actions if a.action == "expand_graph"]
    assert actions
    assert actions[0].subject == "user:123"
    assert actions[0].predicate == "LIKES"


class _CoverageClustersOutstanding:
    """Coverage where cluster summaries are wanted but unattainable.

    `needs_clusters` is `prefer_clusters and semantic_enough and
    cluster_summaries < min_cluster_summaries`. Cluster summaries only exist
    after consolidation, which is caller-invoked by design and may never have
    run for a given corpus - so this state is permanent, not transient.
    """

    needs_semantic = False
    needs_clusters = True


def test_graph_expansion_not_blocked_by_unattainable_cluster_coverage() -> None:
    """Graph traversal must not be gated behind episodic cluster summaries.

    The two are different lanes with no ordering dependency: clusters
    summarize episodes, graph walks typed relationships. Gating graph on
    cluster coverage makes graph permanently unreachable on any corpus where
    consolidation has not been run, even when a graph backend is configured,
    connected and populated.
    """

    class _Pack:
        graph = []
        facts = [_kb_fact()]
        chunks = []
        steps = []
        query_text = "How should IAM and VPC be used in a multi-tier architecture?"
        intent = "topical"
        owner_type = "agent"
        owner_id = "agent:test"
        user_id = "user:123"

    decision = deterministic_decision(
        _Pack(),
        _CoverageClustersOutstanding(),
        cfg={
            "chunk_fallback_enabled": False,
            "graph_predicate_limit": 2,
            "graph_expansion_available": True,
        },
    )
    assert decision is not None
    actions = [a for a in decision.actions if a.action == "expand_graph"]
    assert actions, (
        "graph expansion was skipped because cluster coverage was outstanding; "
        "clusters are an episodic concern and must not gate graph traversal"
    )


def test_topical_graph_expansion_is_not_pinned_to_first_scope() -> None:
    """Topical graph expansion must span every scope the request carries.

    `pack.owner_type` is assigned from `scopes[0]`, which is an ordering
    accident, not a statement about where graph data lives. Stamping it onto
    the action makes `_scopes_for_action` narrow execution to that one scope,
    so a request carrying both an agent and a user scope only ever queries the
    first. Graph relationships written under the user scope are then
    unreachable whenever the agent scope happens to sort first.

    Leaving `owner_type` unset is the existing canonical signal for
    "scope-agnostic": `_scopes_for_action` returns every scope for it. This
    widens nothing - the scopes are the ones the caller already granted.
    """

    class _Pack:
        graph = []
        facts = [_kb_fact()]
        chunks = []
        steps = []
        query_text = "How should IAM and VPC be used in a multi-tier architecture?"
        intent = "topical"
        owner_type = "agent"  # scopes[0] happened to be the agent scope
        owner_id = "agent:test"
        user_id = "user:123"

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={
            "chunk_fallback_enabled": False,
            "graph_predicate_limit": 2,
            "graph_expansion_available": True,
        },
    )
    assert decision is not None
    actions = [a for a in decision.actions if a.action == "expand_graph"]
    assert actions, "expected topical graph expansion actions"
    assert all(a.owner_type is None for a in actions), (
        "topical expansion pinned owner_type to scopes[0]; graph data owned by "
        "another scope in the same request becomes unreachable"
    )


def _preference_fact() -> Fact:
    return Fact(
        id="fact_pref",
        subject="user:123",
        predicate="LIKES",
        object="sushi",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        source_ids=[],
        meta={"domain": "user_profile", "fact_text": "user likes sushi"},
        owner_type="user",
        owner_id="user:123",
        salience=0.0,
        confidence=0.7,
    )


def test_personal_graph_expansion_fires_when_agent_scope_sorts_first() -> None:
    """A personal query must reach the user-anchored graph lane regardless of scope order.

    The branch gate used to require `pack.owner_type == "user"`. That value is
    `scopes[0].owner_type`, so on a request carrying both an agent and a user
    scope - the ordinary case - a personal question fell through to topical
    handling and the LIKES/PREFERS lane was never consulted. The user anchor
    (`pack.user_id`), not the scope ordering, is what makes personal expansion
    meaningful.
    """

    class _Pack:
        graph = []
        facts = [_preference_fact()]
        chunks = []
        steps = []
        query_text = "What do I like?"
        intent = "personal"
        owner_type = "agent"  # scopes[0] happened to be the agent scope
        owner_id = "agent:test"
        user_id = "user:123"

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={
            "chunk_fallback_enabled": False,
            "graph_predicate_limit": 2,
            "next_predicate_scope": lambda _p, _limit: ["LIKES"],
            "graph_expansion_available": True,
        },
    )
    assert decision is not None
    actions = [a for a in decision.actions if a.action == "expand_graph"]
    assert actions, "personal graph expansion did not fire when scopes[0] was the agent scope"
    assert actions[0].subject == "user:123", "personal expansion must stay anchored on the user id"
    assert actions[0].owner_type == "user", (
        "personal expansion anchors on user_id and must execute against the user scope"
    )


def _preference_fact_kb_domain() -> Fact:
    """A preference fact tagged kb_doc, not user_profile.

    Synthetic but deliberate: it isolates the branch's own logic from how
    domain classification behaves in production. See the ticket for the open
    question of how often this combination occurs on real corpora.
    """
    return Fact(
        id="fact_pref_kb",
        subject="user:123",
        predicate="LIKES",
        object="sushi",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        source_ids=[],
        meta={"domain": "kb_doc", "fact_text": "user likes sushi"},
        owner_type="user",
        owner_id="user:123",
        salience=0.0,
        confidence=0.7,
    )


def test_personal_graph_expansion_falls_through_to_topical_when_predicate_scope_empty() -> None:
    """A personal query must not lose the topical fallback when no user-profile
    predicate survives domain filtering.

    Reproduced against the pre-fix commit (1f6eb9f) with an identical pack:
    old code (gated on `pack.owner_type == "user"`, false here) fell through
    to the topical branch and returned an ExpandGraphAction seeded from the
    query's 'alice' entity. New code's wider gate (`pack.user_id` truthy)
    entered the personal branch, found predicate_scope empty after domain
    filtering, and returned `[]` unconditionally via the branch's own
    `return actions` - losing the fallback entirely.
    """

    class _Pack:
        graph = []
        facts = [_preference_fact_kb_domain()]
        chunks = []
        steps = []
        query_text = "What do I like about Alice?"
        intent = "personal"
        owner_type = "agent"
        owner_id = "agent:test"
        user_id = "user:123"
        active_domains = ["kb_doc"]  # does not include user_profile

    def next_predicate_scope(pack: object, limit: int) -> list[str]:
        # Mirrors the controller's real cfg wiring: PREFERENCE_PREDICATES are
        # stripped whenever "user_profile" is absent from active_domains.
        from uma.retrieve.rlm.controller import _filter_predicates_for_domains
        from uma.retrieve.rlm.decisions import next_predicate_scope as raw_scope

        return _filter_predicates_for_domains(
            raw_scope(
                facts=getattr(pack, "facts", []) or [],
                predicate_weights=None,
                graph_predicate_limit=limit,
            ),
            active_domains=set(getattr(pack, "active_domains", []) or []),
        )

    decision = deterministic_decision(
        _Pack(),
        _Coverage(),
        cfg={
            "chunk_fallback_enabled": False,
            "graph_predicate_limit": 2,
            "graph_expansion_available": True,
            "next_predicate_scope": next_predicate_scope,
        },
    )
    assert decision is not None, (
        "personal branch swallowed the topical fallback when predicate_scope "
        "resolved empty; the query's 'alice' entity should have seeded a "
        "topical expansion instead"
    )
    actions = [a for a in decision.actions if a.action == "expand_graph"]
    assert actions, "expected a topical fallback ExpandGraphAction"
    assert actions[0].subject == "alice"
    assert actions[0].domain_scope == ["kb_doc"]
