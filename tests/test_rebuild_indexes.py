from datetime import datetime, timedelta
import asyncio
import threading

import pytest

from uma.common import maintenance as maintenance_module
from uma.common.identity import normalize_user_id
from uma.common.types import Episode
from uma.common.types import Fact
from uma.common.types import Skill
from uma.retrieve.user_query_helper import build_fact_embedding_text


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
    assert memory.episodic_core.vector_index()._scopes[episode.id] == ("default", "user", owner_id)
    assert isinstance(memory.episodic_core.vector_index()._extra.get(episode.id), dict)
    assert memory.semantic_core.vector_index()._scopes[fact.id] == ("default", "user", owner_id)
    assert memory.semantic_core.vector_index()._extra[fact.id]["subject"] == owner_id
    assert memory.semantic_core.vector_index()._extra[fact.id]["predicate"] == "prefers"
    assert memory.procedural_core.vector_index()._scopes[skill.id] == ("default", "user", owner_id)
    assert memory.procedural_core.vector_index()._extra[skill.id]["name"] == "Make coffee"


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
        tenant_id="tenant-a",
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

    assert memory.episodic_core.vector_index()._scopes[episode_1.id] == ("tenant-a", "user", owner_id)
    assert memory.semantic_core.vector_index()._scopes[fact_1.id] == ("tenant-a", "user", owner_id)

    params_list = [params or {} for _cypher, params in adapter.queries]
    assert any(params.get("episode_id") == "ep-graph-1" and params.get("tenant_id") == "tenant-a" for params in params_list)
    assert any(params.get("fact_id") == "fact_graph_1" and params.get("scope_model_version") == "v2" for params in params_list)
    assert any(params.get("ep_id") == "ep-graph-1" and params.get("fact_id") == "fact_graph_1" for params in params_list)
    assert any(params.get("a") == "ep-graph-1" and params.get("b") == "ep-graph-2" for params in params_list)

    first_semantic_meta = dict(memory.semantic_core.vector_index()._extra[fact_1.id])
    first_query_count = len(adapter.queries)

    result_again = await memory.rebuild_derived_indexes(
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        include_procedural=False,
    )

    assert result_again["status"] in ("ok", "degraded")
    assert result_again["graph"] == result["graph"]
    assert memory.semantic_core.vector_index()._extra[fact_1.id] == first_semantic_meta
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
        tenant_id="tenant-w",
        owner_type="workspace",
        owner_id="workspace:alpha",
        include_episodic=False,
        include_procedural=False,
    )

    assert result["status"] in ("ok", "degraded")
    assert memory.semantic_core.vector_index()._scopes[fact.id] == ("tenant-w", "workspace", "workspace:alpha")
    metadata = memory.semantic_core.vector_index()._extra[fact.id]
    assert metadata["subject"] == "workspace:alpha"
    assert metadata["predicate"] == "contains"


@pytest.mark.asyncio
async def test_rebuild_derived_indexes_is_tenant_scoped_for_identical_owner_tuple(uma_memory):
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    base_ts = datetime.utcnow()
    embedding = (await memory.embedder.embed(["tenant scoped rebuild"]))[0]

    await memory.episodic_core.add_episode(
        Episode(
            id="ep-tenant-a",
            timestamp=base_ts,
            summary="tenant a episode",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-a",
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        embedding,
    )
    await memory.episodic_core.add_episode(
        Episode(
            id="ep-tenant-b",
            timestamp=base_ts,
            summary="tenant b episode",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-b",
            session_id="session-b",
            origin_agent_id="agent-b",
            origin_user_id=owner_id,
            origin_session_id="session-b",
            meta={"turn_id": "turn-b"},
        ),
        embedding,
    )

    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_tenant_rebuild_a",
            subject=owner_id,
            predicate="LIKES",
            object="alpha",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-a"],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_tenant_rebuild_b",
            subject=owner_id,
            predicate="LIKES",
            object="beta",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-b"],
            confidence=0.9,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-b",
            origin_agent_id="agent-b",
            origin_user_id=owner_id,
            origin_session_id="session-b",
            meta={"turn_id": "turn-b"},
        ),
        embedding,
    )

    await memory.procedural_core.add_skill(
        Skill(
            id="skill_tenant_rebuild_a",
            name="Tenant A Skill",
            description="alpha",
            created_at=base_ts,
            updated_at=base_ts,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            trigger_phrases=["alpha"],
            trigger_patterns=[],
            plan={"steps": ["a"]},
            tools=["tool-a"],
            example="alpha",
            meta={},
        ),
        embedding,
    )
    await memory.procedural_core.add_skill(
        Skill(
            id="skill_tenant_rebuild_b",
            name="Tenant B Skill",
            description="beta",
            created_at=base_ts,
            updated_at=base_ts,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            trigger_phrases=["beta"],
            trigger_patterns=[],
            plan={"steps": ["b"]},
            tools=["tool-b"],
            example="beta",
            meta={},
        ),
        embedding,
    )

    memory.episodic_core.vector_index().delete(["ep-tenant-a", "ep-tenant-b"])
    memory.semantic_core.vector_index().delete(["fact_tenant_rebuild_a", "fact_tenant_rebuild_b"])
    memory.procedural_core.vector_index().delete(["skill_tenant_rebuild_a", "skill_tenant_rebuild_b"])
    memory.graph_core.adapter.queries.clear()

    result = await memory.rebuild_derived_indexes(
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
    )

    assert result["status"] in ("ok", "degraded")
    assert result["vector"]["report"]["episodic"]["count"] == 1
    assert result["vector"]["report"]["semantic"]["count"] == 1
    assert result["vector"]["report"]["procedural"]["count"] == 1
    assert result["graph"]["episodes"] == 1
    assert result["graph"]["facts"] == 1

    assert "ep-tenant-a" in memory.episodic_core.vector_index()._vectors
    assert "ep-tenant-b" not in memory.episodic_core.vector_index()._vectors
    assert "fact_tenant_rebuild_a" in memory.semantic_core.vector_index()._vectors
    assert "fact_tenant_rebuild_b" not in memory.semantic_core.vector_index()._vectors
    assert "skill_tenant_rebuild_a" in memory.procedural_core.vector_index()._vectors
    assert "skill_tenant_rebuild_b" not in memory.procedural_core.vector_index()._vectors

    params_list = [params or {} for _cypher, params in memory.graph_core.adapter.queries]
    assert any(params.get("tenant_id") == "tenant-a" and params.get("episode_id") == "ep-tenant-a" for params in params_list)
    assert not any(params.get("tenant_id") == "tenant-b" for params in params_list)


@pytest.mark.asyncio
async def test_vector_rebuild_lock_prevents_overlapping_execution(uma_memory, monkeypatch: pytest.MonkeyPatch):
    memory = uma_memory
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = {"count": 0}
    completions: list[str] = []

    async def slow_unlocked(*args, **kwargs):
        calls["count"] += 1
        entered.set()
        await release.wait()
        return {"status": "ok", "report": {}}

    monkeypatch.setattr(maintenance_module, "_rebuild_vector_indexes_unlocked", slow_unlocked)

    async def call(name: str) -> None:
        await maintenance_module.rebuild_vector_indexes(memory, owner_type="user", owner_id="user:u1")
        completions.append(name)

    first = asyncio.create_task(call("first"))
    await entered.wait()
    second = asyncio.create_task(call("second"))
    await asyncio.sleep(0.05)

    assert calls["count"] == 1
    assert completions == []

    release.set()
    await asyncio.gather(first, second)

    assert calls["count"] == 2
    assert sorted(completions) == ["first", "second"]


@pytest.mark.asyncio
async def test_graph_rebuild_lock_prevents_overlapping_execution(uma_memory, monkeypatch: pytest.MonkeyPatch):
    memory = uma_memory
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = {"count": 0}
    completions: list[str] = []

    async def slow_unlocked(*args, **kwargs):
        calls["count"] += 1
        entered.set()
        await release.wait()
        return {"status": "ok", "episodes": 0, "facts": 0, "episode_fact_links": 0, "temporal_links": 0}

    monkeypatch.setattr(maintenance_module, "_rebuild_graph_from_authoritative_stores_unlocked", slow_unlocked)

    async def call(name: str) -> None:
        await maintenance_module._rebuild_graph_from_authoritative_stores(
            memory,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=normalize_user_id("user:u1"),
            include_graph=True,
        )
        completions.append(name)

    first = asyncio.create_task(call("first"))
    await entered.wait()
    second = asyncio.create_task(call("second"))
    await asyncio.sleep(0.05)

    assert calls["count"] == 1
    assert completions == []

    release.set()
    await asyncio.gather(first, second)

    assert calls["count"] == 2
    assert sorted(completions) == ["first", "second"]


@pytest.mark.asyncio
async def test_graph_rebuild_clears_scoped_materialization_before_replay(uma_memory):
    memory = uma_memory
    owner_id = normalize_user_id("user:u1")
    base_ts = datetime.utcnow()
    embedding = (await memory.embedder.embed(["graph rebuild scope"]))[0]

    await memory.episodic_core.add_episode(
        Episode(
            id="ep-scope-clear",
            timestamp=base_ts,
            summary="scope clear",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-clear",
            session_id="session-clear",
            origin_agent_id="agent-clear",
            origin_user_id=owner_id,
            origin_session_id="session-clear",
            meta={"turn_id": "turn-clear"},
        ),
        embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_scope_clear",
            subject=owner_id,
            predicate="likes",
            object="clear-tea",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-clear"],
            confidence=0.9,
            tenant_id="tenant-clear",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-clear",
            origin_agent_id="agent-clear",
            origin_user_id=owner_id,
            origin_session_id="session-clear",
            meta={"turn_id": "turn-clear"},
        ),
        embedding,
    )

    adapter = memory.graph_core.adapter
    adapter.queries.clear()

    result = await memory.rebuild_derived_indexes(
        tenant_id="tenant-clear",
        owner_type="user",
        owner_id=owner_id,
        include_procedural=False,
    )

    assert result["status"] in ("ok", "degraded")
    assert len(adapter.queries) >= 3
    clear_queries = adapter.queries[:3]
    clear_params = [params or {} for _cypher, params in clear_queries]
    assert all(params.get("tenant_id") == "tenant-clear" for params in clear_params)
    assert all(params.get("owner_type") == "user" for params in clear_params)
    assert all(params.get("owner_id") == owner_id for params in clear_params)
    assert "DELETE r" in clear_queries[0][0]
    assert "DETACH DELETE f" in clear_queries[1][0]
    assert "DETACH DELETE e" in clear_queries[2][0]


@pytest.mark.asyncio
async def test_graph_rebuild_clear_is_scoped_to_requested_owner(uma_memory):
    memory = uma_memory
    owner_id = normalize_user_id("user:shared")
    base_ts = datetime.utcnow()
    embedding = (await memory.embedder.embed(["graph rebuild owner scope"]))[0]

    await memory.episodic_core.add_episode(
        Episode(
            id="ep-clear-a",
            timestamp=base_ts,
            summary="tenant a",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-a",
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        embedding,
    )
    await memory.episodic_core.add_episode(
        Episode(
            id="ep-clear-b",
            timestamp=base_ts + timedelta(seconds=1),
            summary="tenant b",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-b",
            session_id="session-b",
            origin_agent_id="agent-b",
            origin_user_id=owner_id,
            origin_session_id="session-b",
            meta={"turn_id": "turn-b"},
        ),
        embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_clear_a",
            subject=owner_id,
            predicate="likes",
            object="alpha",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-a"],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_clear_b",
            subject=owner_id,
            predicate="likes",
            object="beta",
            created_at=base_ts + timedelta(seconds=1),
            updated_at=base_ts + timedelta(seconds=1),
            source_ids=["chunk-b"],
            confidence=0.9,
            tenant_id="tenant-b",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-b",
            origin_agent_id="agent-b",
            origin_user_id=owner_id,
            origin_session_id="session-b",
            meta={"turn_id": "turn-b"},
        ),
        embedding,
    )

    adapter = memory.graph_core.adapter
    adapter.queries.clear()

    await memory.rebuild_derived_indexes(
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        include_procedural=False,
    )

    clear_params = [(params or {}) for _cypher, params in adapter.queries[:3]]
    assert all(params.get("tenant_id") == "tenant-a" for params in clear_params)
    assert not any(params.get("tenant_id") == "tenant-b" for params in clear_params)


@pytest.mark.asyncio
async def test_live_write_overlap_with_vector_rebuild_keeps_retrieval_scoped(uma_memory, monkeypatch: pytest.MonkeyPatch):
    memory = uma_memory
    owner_id = normalize_user_id("user:scope-a")
    other_owner_id = normalize_user_id("user:scope-b")
    base_ts = datetime.utcnow()

    existing_fact = Fact(
        id="fact_overlap_existing",
        subject=owner_id,
        predicate="LIKES",
        object="coffee",
        created_at=base_ts,
        updated_at=base_ts,
        source_ids=["chunk-existing"],
        confidence=0.9,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
    )
    live_fact = Fact(
        id="fact_overlap_live",
        subject=owner_id,
        predicate="LIKES",
        object="tea",
        created_at=base_ts + timedelta(seconds=1),
        updated_at=base_ts + timedelta(seconds=1),
        source_ids=["chunk-live"],
        confidence=0.9,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        session_id="session-a",
        origin_agent_id="agent-a",
        origin_user_id=owner_id,
        origin_session_id="session-a",
    )
    other_fact = Fact(
        id="fact_overlap_other_tenant",
        subject=other_owner_id,
        predicate="LIKES",
        object="juice",
        created_at=base_ts,
        updated_at=base_ts,
        source_ids=["chunk-other"],
        confidence=0.9,
        tenant_id="tenant-b",
        owner_type="user",
        owner_id=other_owner_id,
        session_id="session-b",
        origin_agent_id="agent-b",
        origin_user_id=other_owner_id,
        origin_session_id="session-b",
    )

    existing_embedding, live_embedding, other_embedding = await memory.embedder.embed(
        [
            build_fact_embedding_text(existing_fact),
            build_fact_embedding_text(live_fact),
            build_fact_embedding_text(other_fact),
        ]
    )
    await memory.semantic_core.upsert_fact(existing_fact, existing_embedding)
    await memory.semantic_core.upsert_fact(other_fact, other_embedding)

    original_list_facts = memory.semantic_core.list_facts_for_owner
    entered = asyncio.Event()
    release = asyncio.Event()
    blocked = {"done": False}

    async def paused_list_facts_for_owner(*, tenant_id: str, owner_type: str, owner_id: str, limit=None):
        if (
            not blocked["done"]
            and tenant_id == "tenant-a"
            and owner_type == "user"
            and owner_id == normalize_user_id("user:scope-a")
        ):
            blocked["done"] = True
            entered.set()
            await release.wait()
        return await original_list_facts(
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            limit=limit,
        )

    monkeypatch.setattr(memory.semantic_core, "list_facts_for_owner", paused_list_facts_for_owner)

    rebuild_task = asyncio.create_task(
        memory.rebuild_vector_indexes(
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            include_episodic=False,
            include_procedural=False,
        )
    )
    await entered.wait()

    await memory.semantic_core.upsert_fact(live_fact, live_embedding)
    release.set()
    rebuild_result = await rebuild_task

    assert rebuild_result["status"] in ("ok", "degraded")

    tenant_a_results = await memory.semantic_core.search(
        query_embedding=live_embedding,
        tenant_id="tenant-a",
        owner_type="user",
        owner_id=owner_id,
        k=10,
        query_text=build_fact_embedding_text(live_fact),
    )
    assert tenant_a_results
    assert all(getattr(fact, "tenant_id", None) == "tenant-a" for fact in tenant_a_results)
    assert all(getattr(fact, "owner_id", None) == owner_id for fact in tenant_a_results)
    assert not any(getattr(fact, "id", None) == other_fact.id for fact in tenant_a_results)

    tenant_b_results = await memory.semantic_core.search(
        query_embedding=other_embedding,
        tenant_id="tenant-b",
        owner_type="user",
        owner_id=other_owner_id,
        k=10,
        query_text=build_fact_embedding_text(other_fact),
    )
    assert tenant_b_results
    assert all(getattr(fact, "tenant_id", None) == "tenant-b" for fact in tenant_b_results)
    assert all(getattr(fact, "owner_id", None) == other_owner_id for fact in tenant_b_results)
    assert not any(getattr(fact, "id", None) == live_fact.id for fact in tenant_b_results)


@pytest.mark.asyncio
async def test_deferred_graph_update_overlap_with_graph_rebuild_keeps_scope_isolated(
    uma_memory,
    monkeypatch: pytest.MonkeyPatch,
):
    memory = uma_memory
    owner_id = normalize_user_id("user:graph-a")
    base_ts = datetime.utcnow()
    seed_embedding = (await memory.embedder.embed(["graph overlap seed"]))[0]

    await memory.episodic_core.add_episode(
        Episode(
            id="ep-graph-overlap-a",
            timestamp=base_ts,
            summary="seed episode",
            user_id=owner_id,
            owner_type="user",
            owner_id=owner_id,
            tenant_id="tenant-a",
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        seed_embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_graph_overlap_a",
            subject=owner_id,
            predicate="LIKES",
            object="alpha",
            created_at=base_ts,
            updated_at=base_ts,
            source_ids=["chunk-a"],
            confidence=0.9,
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            session_id="session-a",
            origin_agent_id="agent-a",
            origin_user_id=owner_id,
            origin_session_id="session-a",
            meta={"turn_id": "turn-a"},
        ),
        seed_embedding,
    )

    adapter = memory.graph_core.adapter
    adapter.queries.clear()

    original_list_episodes = memory.episodic_core.list_episodes
    entered = asyncio.Event()
    release = asyncio.Event()
    blocked = {"done": False}

    async def paused_list_episodes(tenant_id: str, owner_type: str, owner_id: str):
        if (
            not blocked["done"]
            and tenant_id == "tenant-a"
            and owner_type == "user"
            and owner_id == normalize_user_id("user:graph-a")
        ):
            blocked["done"] = True
            entered.set()
            await release.wait()
        return await original_list_episodes(tenant_id, owner_type, owner_id)

    monkeypatch.setattr(memory.episodic_core, "list_episodes", paused_list_episodes)

    rebuild_task = asyncio.create_task(
        memory.rebuild_derived_indexes(
            tenant_id="tenant-a",
            owner_type="user",
            owner_id=owner_id,
            include_procedural=False,
        )
    )
    await entered.wait()

    await memory.process_turn(
        user_id="user:graph-b",
        user_msg="I like tea",
        assistant_reply="Noted that you like tea.",
        session_id="session-b",
        tenant_id="tenant-b",
        workspace_id="workspace-b",
        extra_meta={"request_id": "req-graph-b"},
    )

    release.set()
    rebuild_result = await rebuild_task

    assert rebuild_result["status"] in ("ok", "degraded")
    clear_queries = [
        (cypher, params or {})
        for cypher, params in adapter.queries
        if "DELETE r" in cypher or "DETACH DELETE f" in cypher or "DETACH DELETE e" in cypher
    ]
    assert len(clear_queries) == 3
    assert all(params.get("tenant_id") == "tenant-a" for _cypher, params in clear_queries)
    assert all(params.get("owner_type") == "user" for _cypher, params in clear_queries)
    assert all(params.get("owner_id") == owner_id for _cypher, params in clear_queries)

    params_list = [params or {} for _cypher, params in adapter.queries]
    assert any(params.get("tenant_id") == "tenant-b" and params.get("owner_id") == normalize_user_id("user:graph-b") for params in params_list)
    assert any(params.get("tenant_id") == "tenant-a" and params.get("owner_id") == owner_id for params in params_list)
    assert not any(
        (params.get("tenant_id") == "tenant-b")
        and (
            "DELETE r" in cypher
            or "DETACH DELETE f" in cypher
            or "DETACH DELETE e" in cypher
        )
        for cypher, params in adapter.queries
    )


@pytest.mark.asyncio
async def test_semantic_search_drops_vector_candidates_without_committed_sql_row(
    uma_memory,
    monkeypatch: pytest.MonkeyPatch,
):
    memory = uma_memory
    owner_id = normalize_user_id("user:transient")
    fact = Fact(
        id="fact_transient_visibility",
        subject=owner_id,
        predicate="LIKES",
        object="transient coffee",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=["chunk-transient"],
        confidence=0.9,
        tenant_id="tenant-transient",
        owner_type="user",
        owner_id=owner_id,
        session_id="session-transient",
        origin_agent_id="agent-transient",
        origin_user_id=owner_id,
        origin_session_id="session-transient",
    )
    embedding = (await memory.embedder.embed([build_fact_embedding_text(fact)]))[0]

    vector_index = memory.semantic_core.vector_index()
    real_upsert = vector_index.upsert
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def blocking_upsert(*args, **kwargs):
        real_upsert(*args, **kwargs)
        entered.set()
        if not release.wait(timeout=2.0):
            raise TimeoutError("timed out waiting to release vector upsert")

    monkeypatch.setattr(vector_index, "upsert", blocking_upsert)

    def writer() -> None:
        try:
            asyncio.run(memory.semantic_core.upsert_fact(fact, embedding))
        except BaseException as exc:  # pragma: no cover - failure capture only
            failures.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(timeout=1.0)

    transient_results = await memory.semantic_core.search(
        query_embedding=embedding,
        tenant_id="tenant-transient",
        owner_type="user",
        owner_id=owner_id,
        k=10,
        query_text=build_fact_embedding_text(fact),
    )
    assert transient_results == []

    release.set()
    thread.join(timeout=2.0)
    assert not failures

    committed_results = await memory.semantic_core.search(
        query_embedding=embedding,
        tenant_id="tenant-transient",
        owner_type="user",
        owner_id=owner_id,
        k=10,
        query_text=build_fact_embedding_text(fact),
    )
    assert [item.id for item in committed_results] == [fact.id]
