from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from uma.retrieve.rlm.environment import UMAMemoryEnvironment
from uma.retrieve.rlm.request import RetrievalRequest
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import RuntimeContext
from uma.common.types import Episode, Fact
from tests.helpers.graph_adapter import RecordingGraphAdapter


@pytest.mark.asyncio
async def test_environment_get_query_embedding_shape(uma_memory):
    env = UMAMemoryEnvironment(uma_memory)
    vec = await env.get_query_embedding("hello world")
    assert isinstance(vec, list)
    assert len(vec) == int(getattr(uma_memory.embedder, "dimension", 0))


@pytest.mark.asyncio
async def test_environment_fetch_facts_by_ids_is_owner_scoped(uma_memory):
    memory = uma_memory
    env = UMAMemoryEnvironment(memory)
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id or "agent-default",
            request_id="req-env-facts",
            user_id="user:u1",
        )
    )

    now = datetime.now(timezone.utc)
    owner_type = "user"
    owner_id = "user:u1"
    emb = (await memory.embedder.embed(["coffee"]))[0]

    fact = Fact(
        id="fact_1",
        subject="user",
        predicate="LIKES",
        object="coffee",
        created_at=now,
        updated_at=now,
        source_ids=[],
        confidence=0.9,
        salience=0.8,
        owner_type=owner_type,
        owner_id=owner_id,
        meta={},
    )
    await memory.semantic_core.upsert_fact(fact, emb)

    facts = await env.fetch_facts_by_ids(request, ["fact_1"], owner_type=owner_type, owner_id=owner_id)
    assert facts and getattr(facts[0], "id", None) == "fact_1"

    # Wrong scope => no results.
    wrong = await env.fetch_facts_by_ids(
        request,
        ["fact_1"],
        owner_type="agent",
        owner_id=memory.agent_id,
    )
    assert wrong == []


@pytest.mark.asyncio
async def test_environment_graph_neighbors_returns_empty_when_no_edges(uma_memory):
    env = UMAMemoryEnvironment(uma_memory)
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant-test",
            agent_id=uma_memory.agent_id or "agent-default",
            request_id="req-env-graph",
            user_id="user:u1",
        )
    )
    out = await env.graph_neighbors(
        request,
        "node1",
        predicate_scope=["LIKES"],
        depth=2,
        k=5,
        owner_type="agent",
        owner_id=uma_memory.agent_id,
    )
    assert out == []

    adapter = getattr(uma_memory.graph_core, "adapter", None)
    queries = getattr(adapter, "queries", None)
    if isinstance(queries, list) and queries:
        _cypher, params = queries[-1]
        assert params["tenant_id"] == "tenant-test"
        assert params["owner_type"] == "agent"
        assert params["owner_id"] == (uma_memory.agent_id or "agent-default")


@pytest.mark.asyncio
async def test_environment_graph_resolve_nodes_is_tenant_and_owner_scoped(uma_memory):
    env = UMAMemoryEnvironment(uma_memory)
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant-test",
            agent_id=uma_memory.agent_id or "agent-default",
            request_id="req-env-graph-resolve",
            user_id="user:u1",
        )
    )
    adapter = RecordingGraphAdapter()
    adapter.next_results.append([{"node_id": "resolved-node"}])
    uma_memory.graph_core.adapter = adapter

    out = await env.graph_resolve_nodes(
        request,
        names=["Resolved Node"],
        owner_type="agent",
        owner_id=uma_memory.agent_id,
        limit=5,
    )
    assert out == ["resolved-node"]
    assert adapter.queries
    _cypher, params = adapter.queries[-1]
    assert params["tenant_id"] == "tenant-test"
    assert params["owner_type"] == "agent"
    assert params["owner_id"] == (uma_memory.agent_id or "agent-default")


@pytest.mark.asyncio
async def test_environment_graph_neighbors_rejects_workspace_scope_in_runtime_flow(uma_memory):
    env = UMAMemoryEnvironment(uma_memory)
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant-test",
            agent_id=uma_memory.agent_id or "agent-default",
            request_id="req-env-graph-workspace",
            user_id="user:u1",
        )
    )
    with pytest.raises(ValueError, match="invalid owner_type"):
        await env.graph_neighbors(
            request,
            "node1",
            depth=1,
            k=5,
            owner_type="workspace",
            owner_id="workspace:alpha",
        )


def test_environment_time_range_helpers():
    start = 1000
    end = 500
    assert UMAMemoryEnvironment._sanitize_time_range({"start": start, "end": end, "offset": -5}) == {
        "start": start,
        "offset": 0,
    }

    now = datetime.now(timezone.utc)
    eps = [
        Episode(id="e1", timestamp=now - timedelta(days=2), summary="x", user_id="u1", owner_type="user", owner_id="user:u1"),
        Episode(id="e2", timestamp=now, summary="y", user_id="u1", owner_type="user", owner_id="user:u1"),
    ]
    filtered = UMAMemoryEnvironment._filter_time_range(eps, {"start": now - timedelta(days=1)})
    assert [e.id for e in filtered] == ["e2"]
