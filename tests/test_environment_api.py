from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from uma.core.retrieval.rlm.environment import UMAMemoryEnvironment
from uma.types import Episode, Fact


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

    facts = await env.fetch_facts_by_ids("user:u1", ["fact_1"], owner_type=owner_type, owner_id=owner_id)
    assert facts and getattr(facts[0], "id", None) == "fact_1"

    # Wrong scope => no results.
    wrong = await env.fetch_facts_by_ids("user:u1", ["fact_1"], owner_type="agent", owner_id=memory.agent_id)
    assert wrong == []


@pytest.mark.asyncio
async def test_environment_graph_neighbors_returns_empty_when_no_edges(uma_memory):
    env = UMAMemoryEnvironment(uma_memory)
    out = await env.graph_neighbors("user:u1", "node1", predicate_scope=["LIKES"], depth=2, k=5)
    assert out == []


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
