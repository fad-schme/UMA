from __future__ import annotations

from datetime import datetime

import pytest

from uma.retrieve.rlm.decisions import RetrievalAction
from uma.retrieve.rlm.request import RetrievalRequest
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import Episode, Fact, RuntimeContext


def _request_for_session(memory, *, include_legacy_turn_data: bool = False) -> RetrievalRequest:
    return RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=memory.agent_id or "agent-default",
            request_id="req-legacy-turn",
            user_id="user:u1",
            session_id="session-a",
        ),
        include_legacy_turn_data=include_legacy_turn_data,
    )


@pytest.mark.asyncio
async def test_default_retrieval_excludes_legacy_user_global_turn_data(uma_memory) -> None:
    memory = uma_memory
    embedding = (await memory.embedder.embed(["legacy turn retrieval"]))[0]
    env = memory.memory_env
    assert env is not None

    legacy_fact = Fact(
        id="fact_legacy_turn",
        subject="user:u1",
        predicate="LIKES",
        object="tea",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=[],
        owner_type="user",
        owner_id="user:u1",
        tenant_id=DEFAULT_TENANT_ID,
        session_id=None,
        origin_agent_id=memory.agent_id,
        origin_user_id="user:u1",
        origin_session_id=None,
        meta={"turn_id": "turn-legacy"},
    )
    canonical_fact = Fact(
        id="fact_canonical_turn",
        subject="user:u1",
        predicate="LIKES",
        object="coffee",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_ids=[],
        owner_type="user",
        owner_id="user:u1",
        tenant_id=DEFAULT_TENANT_ID,
        session_id="session-a",
        origin_agent_id=memory.agent_id,
        origin_user_id="user:u1",
        origin_session_id="session-a",
        meta={"turn_id": "turn-canonical"},
    )
    await memory.semantic_core.upsert_fact(legacy_fact, embedding)
    await memory.semantic_core.upsert_fact(canonical_fact, embedding)

    legacy_episode = Episode(
        id="episode_legacy_turn",
        timestamp=datetime.utcnow(),
        summary="legacy turn episode",
        user_id="user:u1",
        owner_type="user",
        owner_id="user:u1",
        tenant_id=DEFAULT_TENANT_ID,
        session_id=None,
        origin_agent_id=memory.agent_id,
        origin_user_id="user:u1",
        origin_session_id=None,
        meta={"turn_id": "turn-legacy"},
    )
    canonical_episode = Episode(
        id="episode_canonical_turn",
        timestamp=datetime.utcnow(),
        summary="canonical turn episode",
        user_id="user:u1",
        owner_type="user",
        owner_id="user:u1",
        tenant_id=DEFAULT_TENANT_ID,
        session_id="session-a",
        origin_agent_id=memory.agent_id,
        origin_user_id="user:u1",
        origin_session_id="session-a",
        meta={"turn_id": "turn-canonical"},
    )
    await memory.episodic_core.add_episode(legacy_episode, embedding)
    await memory.episodic_core.add_episode(canonical_episode, embedding)

    request = _request_for_session(memory)

    facts = await env.fetch_more_facts(
        request=request,
        predicate="LIKES",
        k=10,
        owner_type="user",
        owner_id="user:u1",
    )
    fact_ids = [fact.id for fact in facts]
    assert fact_ids == ["fact_canonical_turn"]

    episodes = await env.execute_action(
        request=request,
        action=RetrievalAction(action="search_episodic", k=10, reason="test"),
        query_text="turn episode",
        query_embedding=embedding,
        owner_type="user",
        owner_id="user:u1",
        default_k=10,
    )
    episode_ids = [episode.id for episode in episodes]
    assert episode_ids == ["episode_canonical_turn"]


@pytest.mark.asyncio
async def test_explicit_legacy_turn_inclusion_returns_legacy_and_canonical_without_reinterpreting_rows(uma_memory) -> None:
    memory = uma_memory
    embedding = (await memory.embedder.embed(["legacy include retrieval"]))[0]
    env = memory.memory_env
    assert env is not None

    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_legacy_include",
            subject="user:u1",
            predicate="KNOWS",
            object="legacy",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            source_ids=[],
            owner_type="user",
            owner_id="user:u1",
            tenant_id=DEFAULT_TENANT_ID,
            session_id=None,
            origin_agent_id=memory.agent_id,
            origin_user_id="user:u1",
            origin_session_id=None,
            meta={"turn_id": "turn-legacy-include"},
        ),
        embedding,
    )
    await memory.semantic_core.upsert_fact(
        Fact(
            id="fact_canonical_include",
            subject="user:u1",
            predicate="KNOWS",
            object="canonical",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            source_ids=[],
            owner_type="user",
            owner_id="user:u1",
            tenant_id=DEFAULT_TENANT_ID,
            session_id="session-a",
            origin_agent_id=memory.agent_id,
            origin_user_id="user:u1",
            origin_session_id="session-a",
            meta={"turn_id": "turn-canonical-include"},
        ),
        embedding,
    )

    request = _request_for_session(memory, include_legacy_turn_data=True)
    facts = await env.fetch_more_facts(
        request=request,
        predicate="KNOWS",
        k=10,
        owner_type="user",
        owner_id="user:u1",
    )

    fact_ids = {fact.id for fact in facts}
    assert fact_ids == {"fact_legacy_include", "fact_canonical_include"}

    by_id = {fact.id: fact for fact in facts}
    assert by_id["fact_legacy_include"].session_id is None
    assert by_id["fact_canonical_include"].session_id == "session-a"


def test_legacy_turn_compatibility_flag_does_not_broaden_runtime_owner_scopes(uma_memory) -> None:
    request = _request_for_session(uma_memory, include_legacy_turn_data=True)
    assert request.include_legacy_turn_data is True
    assert [scope.owner_type for scope in request.scopes] == ["agent", "user"]
