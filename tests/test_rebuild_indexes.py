from datetime import datetime, timedelta

import pytest

from uma.core.utils.identity import normalize_user_id
from uma.types import Episode
from uma.types import Fact
from uma.types import Skill


@pytest.mark.asyncio
async def test_rebuild_vector_indexes(uma_memory):
    memory = uma_memory

    user_id = "user:123"
    owner_id = normalize_user_id(user_id)
    embedding = (await memory.embedder.embed(["hello"]))[0]

    episode = Episode(
        id="ep-1",
        timestamp=datetime.utcnow(),
        summary="hello",
        user_id=user_id,
        owner_type="user",
        owner_id=owner_id,
        raw="hello world",
        tags=["test"],
        embedding=embedding,
    )
    await memory.episodic_core.add_episode(episode, embedding)

    fact = Fact(
        id="fact_1",
        subject=owner_id,
        predicate="prefers",
        object="coffee",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=[episode.id],
        confidence=0.9,
        owner_type="user",
        owner_id=owner_id,
    )
    await memory.semantic_core.upsert_fact(fact, embedding)

    skill = Skill(
        id="skill_1",
        name="Make coffee",
        description="Brews a cup of coffee.",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        owner_type="user",
        owner_id=owner_id,
        trigger_phrases=["coffee"],
        trigger_patterns=[],
        plan={"steps": ["boil", "brew"]},
        tools=["kettle"],
        example="Make coffee",
        meta={"tag": "demo"},
    )
    await memory.procedural_core.add_skill(skill, embedding)

    memory.episodic_core.vector_index().delete([episode.id])
    memory.semantic_core.vector_index().delete([fact.id])
    memory.procedural_core.vector_index().delete([skill.id])

    result = await memory.rebuild_vector_indexes(owner_type="user", owner_id=owner_id)
    assert result["status"] in ("ok", "degraded")
    assert episode.id in memory.episodic_core.vector_index()._vectors
    assert fact.id in memory.semantic_core.vector_index()._vectors
    assert skill.id in memory.procedural_core.vector_index()._vectors
    assert memory.episodic_core.vector_index()._metadata[episode.id]["tenant_id"] == "default"
    assert memory.episodic_core.vector_index()._metadata[episode.id]["session_id"] is None
    assert memory.episodic_core.vector_index()._metadata[episode.id]["scope_model_version"] == "v2"
    assert memory.semantic_core.vector_index()._metadata[fact.id]["tenant_id"] == "default"
    assert memory.semantic_core.vector_index()._metadata[fact.id]["session_id"] is None
    assert memory.semantic_core.vector_index()._metadata[fact.id]["scope_model_version"] == "v2"
    assert memory.procedural_core.vector_index()._metadata[skill.id]["tenant_id"] == "default"
    assert memory.procedural_core.vector_index()._metadata[skill.id]["workspace_id"] is None
    assert memory.procedural_core.vector_index()._metadata[skill.id]["scope_model_version"] == "v2"


@pytest.mark.asyncio
async def test_rebuild_derived_indexes_replays_graph_from_authoritative_scope(uma_memory):
    memory = uma_memory
    owner_id = normalize_user_id("user:u1")
    base_ts = datetime.utcnow()

    episode_1 = Episode(
        id="ep-graph-1",
        timestamp=base_ts,
        summary="first summary",
        user_id=owner_id,
        owner_type="user",
        owner_id=owner_id,
        tenant_id="tenant-a",
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
        workspace_id=None,
        meta={"turn_id": "turn-a"},
    )
    episode_2 = Episode(
        id="ep-graph-2",
        timestamp=base_ts + timedelta(seconds=1),
        summary="second summary",
        user_id=owner_id,
        owner_type="user",
        owner_id=owner_id,
        tenant_id="tenant-a",
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
        workspace_id=None,
        meta={"turn_id": "turn-b"},
    )
    embedding = (await memory.embedder.embed(["hello graph"]))[0]
    await memory.episodic_core.add_episode(episode_1, embedding)
    await memory.episodic_core.add_episode(episode_2, embedding)

    fact_1 = Fact(
        id="fact_graph_1",
        subject=owner_id,
        predicate="likes",
        object="tea",
        created_at=base_ts,
        updated_at=base_ts,
        source_ids=["chunk-a"],
        confidence=0.9,
        owner_type="user",
        owner_id=owner_id,
        tenant_id="tenant-a",
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
        meta={"turn_id": "turn-a"},
    )
    fact_2 = Fact(
        id="fact_graph_2",
        subject=owner_id,
        predicate="prefers",
        object="coffee",
        created_at=base_ts + timedelta(seconds=1),
        updated_at=base_ts + timedelta(seconds=1),
        source_ids=["chunk-b"],
        confidence=0.9,
        owner_type="user",
        owner_id=owner_id,
        tenant_id="tenant-a",
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
        meta={"turn_id": "turn-b"},
    )
    await memory.semantic_core.upsert_fact(fact_1, embedding)
    await memory.semantic_core.upsert_fact(fact_2, embedding)

    adapter = getattr(memory.graph_core, "adapter", None)
    assert adapter is not None
    adapter.queries.clear()

    result = await memory.rebuild_derived_indexes(
        owner_type="user",
        owner_id=owner_id,
        include_procedural=False,
    )

    assert result["status"] in ("ok", "degraded")
    assert result["graph"]["status"] == "ok"
    assert result["graph"]["episodes"] == 2
    assert result["graph"]["facts"] == 2
    assert result["graph"]["episode_fact_links"] == 2
    assert result["graph"]["temporal_links"] == 1

    assert memory.episodic_core.vector_index()._metadata[episode_1.id]["tenant_id"] == "tenant-a"
    assert memory.episodic_core.vector_index()._metadata[episode_1.id]["session_id"] == "session-a"
    assert memory.semantic_core.vector_index()._metadata[fact_1.id]["tenant_id"] == "tenant-a"
    assert memory.semantic_core.vector_index()._metadata[fact_1.id]["session_id"] == "session-a"

    params_list = [params or {} for _cypher, params in adapter.queries]
    assert any(params.get("episode_id") == "ep-graph-1" and params.get("tenant_id") == "tenant-a" for params in params_list)
    assert any(params.get("fact_id") == "fact_graph_1" and params.get("scope_model_version") == "v2" for params in params_list)
    assert any(params.get("ep_id") == "ep-graph-1" and params.get("fact_id") == "fact_graph_1" for params in params_list)
    assert any(params.get("a") == "ep-graph-1" and params.get("b") == "ep-graph-2" for params in params_list)

    first_semantic_meta = dict(memory.semantic_core.vector_index()._metadata[fact_1.id])
    first_query_count = len(adapter.queries)

    result_again = await memory.rebuild_derived_indexes(
        owner_type="user",
        owner_id=owner_id,
        include_procedural=False,
    )

    assert result_again["status"] in ("ok", "degraded")
    assert result_again["graph"] == result["graph"]
    assert memory.semantic_core.vector_index()._metadata[fact_1.id] == first_semantic_meta
    assert len(adapter.queries) == first_query_count * 2


@pytest.mark.asyncio
async def test_rebuild_vector_indexes_preserves_promoted_workspace_fact_scope(uma_memory):
    memory = uma_memory
    embedding = (await memory.embedder.embed(["workspace fact"]))[0]

    fact = Fact(
        id="fact_workspace_rebuild",
        subject="workspace:alpha",
        predicate="contains",
        object="runbook",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=["chunk-workspace"],
        confidence=0.7,
        owner_type="workspace",
        owner_id="workspace:alpha",
        tenant_id="tenant-w",
        workspace_id="workspace:alpha",
        session_id=None,
        origin_agent_id="agent-a",
        origin_user_id="user:u1",
        origin_session_id="session-a",
        meta={"promotion_source": "session"},
    )
    await memory.semantic_core.upsert_fact(fact, embedding)
    memory.semantic_core.vector_index().delete([fact.id])

    result = await memory.rebuild_vector_indexes(
        owner_type="workspace",
        owner_id="workspace:alpha",
        include_episodic=False,
        include_procedural=False,
    )

    assert result["status"] in ("ok", "degraded")
    metadata = memory.semantic_core.vector_index()._metadata[fact.id]
    assert metadata["tenant_id"] == "tenant-w"
    assert metadata["owner_type"] == "workspace"
    assert metadata["owner_id"] == "workspace:alpha"
    assert metadata["workspace_id"] == "workspace:alpha"
    assert metadata["session_id"] is None
    assert metadata["scope_model_version"] == "v2"
