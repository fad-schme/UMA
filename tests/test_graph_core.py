from __future__ import annotations

from datetime import datetime

import pytest

from uma.core.graph.core import TemporalGraphCore
from uma.core.graph.updater import GraphUpdater
from uma.types import Episode

from tests.helpers.graph_adapter import RecordingGraphAdapter


def test_insert_fact_triplet_sanitizes_predicate_and_stamps_owner():
    adapter = RecordingGraphAdapter()
    core = TemporalGraphCore(adapter)

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

    # Predicate must be sanitized to a safe relationship type
    assert "BAD_REL" in cypher or "RELATES_TO" in cypher

    # Ownership and provenance are stamped as top-level parameters
    assert params.get("tenant_id") == "tenant-1"
    assert params.get("owner_type") == "user"
    assert params.get("owner_id") == "user:u1"
    assert params.get("fact_id") == "f1"
    assert params.get("source_chunk_id") == "c1"
    assert params.get("scope_model_version") == "v2"


def test_episode_edges_have_ownership():
    adapter = RecordingGraphAdapter()
    graph_core = TemporalGraphCore(adapter)
    updater = GraphUpdater(graph_core)

    ep = Episode(
        id="ep1",
        timestamp=datetime.utcnow(),
        summary="Test episode",
        user_id="u1",
        tenant_id="tenant-1",
        owner_type="user",
        owner_id="user:u1",
    )

    updater.add_episode_node(ep)

    assert adapter.queries, "No Cypher executed for episode insertion"

    # Find the HAS_EPISODE query and validate ownership stamping
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
    core = TemporalGraphCore(adapter)

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


def test_resolve_nodes_enforces_tenant_and_owner_scope() -> None:
    adapter = RecordingGraphAdapter()
    adapter.next_results.append([{"node_id": "workspace-node"}])
    core = TemporalGraphCore(adapter)

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
    core = TemporalGraphCore(RecordingGraphAdapter())
    with pytest.raises(RuntimeError, match="unsafe"):
        core.query("MATCH (n) RETURN n", params={})


def test_insert_fact_triplet_rejects_system_scope() -> None:
    core = TemporalGraphCore(RecordingGraphAdapter())
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
