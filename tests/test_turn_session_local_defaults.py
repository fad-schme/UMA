from __future__ import annotations

import pytest

from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import RuntimeContext, SCOPE_MODEL_VERSION


@pytest.mark.asyncio
async def test_process_turn_writes_session_local_episode_and_fact_provenance(uma_memory) -> None:
    mem = uma_memory
    assert mem.agent_id

    await mem.process_turn(
        user_id="user:u1",
        user_msg="hello",
        assistant_reply="user likes coffee.",
        session_id="session-a",
        extra_meta={"request_id": "req-turn-a"},
    )

    epi_conn = mem._stores["episodic"]._conn()
    try:
        epi_rows = mem._stores["episodic"]._query_all(
            epi_conn,
            """
            SELECT owner_type, owner_id, tenant_id, session_id, origin_agent_id, origin_user_id,
                   origin_session_id, scope_model_version
            FROM episodes
            WHERE owner_type=? AND owner_id=?
            """,
            params=["user", "user:u1"],
            log_context="test_process_turn_episode_scope",
        )
        assert len(epi_rows) == 1
        row = epi_rows[0]
        assert row["tenant_id"] == DEFAULT_TENANT_ID
        assert row["session_id"] == "session-a"
        assert row["origin_agent_id"] == mem.agent_id
        assert row["origin_user_id"] == "user:u1"
        assert row["origin_session_id"] == "session-a"
        assert row["scope_model_version"] == SCOPE_MODEL_VERSION
    finally:
        epi_conn.close()

    sem_conn = mem._stores["semantic"]._conn()
    try:
        fact_rows = mem._stores["semantic"]._query_all(
            sem_conn,
            """
            SELECT owner_type, owner_id, tenant_id, session_id, origin_agent_id, origin_user_id,
                   origin_session_id, scope_model_version, object
            FROM facts
            WHERE owner_type=? AND owner_id=?
            ORDER BY id ASC
            """,
            params=["user", "user:u1"],
            log_context="test_process_turn_fact_scope",
        )
        assert fact_rows
        assert any("coffee" in str(row["object"]) for row in fact_rows)
        for row in fact_rows:
            assert row["tenant_id"] == DEFAULT_TENANT_ID
            assert row["session_id"] == "session-a"
            assert row["origin_agent_id"] == mem.agent_id
            assert row["origin_user_id"] == "user:u1"
            assert row["origin_session_id"] == "session-a"
            assert row["scope_model_version"] == SCOPE_MODEL_VERSION
    finally:
        sem_conn.close()


@pytest.mark.asyncio
async def test_process_turn_requires_explicit_session_id(uma_memory) -> None:
    with pytest.raises(ValueError, match="requires a non-empty session_id"):
        await uma_memory.process_turn(
            user_id="user:u1",
            user_msg="hello",
            assistant_reply="user likes coffee.",
        )


@pytest.mark.asyncio
async def test_retrieval_does_not_see_prior_session_turn_artifacts_by_default(uma_memory) -> None:
    mem = uma_memory
    assert mem.agent_id

    await mem.process_turn(
        user_id="user:u1",
        user_msg="first",
        assistant_reply="user likes coffee.",
        session_id="session-a",
    )
    await mem.process_turn(
        user_id="user:u1",
        user_msg="second",
        assistant_reply="user likes tea.",
        session_id="session-b",
    )

    sem_conn = mem._stores["semantic"]._conn()
    try:
        rows = mem._stores["semantic"]._query_all(
            sem_conn,
            "SELECT id, object FROM facts WHERE owner_type=? AND owner_id=? ORDER BY id ASC",
            params=["user", "user:u1"],
            log_context="test_turn_session_fact_ids",
        )
        fact_ids = [row["id"] for row in rows]
    finally:
        sem_conn.close()

    req_a = mem.runtime._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=mem.agent_id,
            request_id="req-a",
            user_id="user:u1",
            session_id="session-a",
        )
    )
    req_b = mem.runtime._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id=mem.agent_id,
            request_id="req-b",
            user_id="user:u1",
            session_id="session-b",
        )
    )
    facts_a = await mem.memory_env.fetch_facts_by_ids(
        req_a,
        fact_ids,
        owner_type="user",
        owner_id="user:u1",
    )
    facts_b = await mem.memory_env.fetch_facts_by_ids(
        req_b,
        fact_ids,
        owner_type="user",
        owner_id="user:u1",
    )

    objects_a = {str(getattr(f, "object", "")) for f in facts_a}
    objects_b = {str(getattr(f, "object", "")) for f in facts_b}
    assert "coffee" in objects_a
    assert "coffee" not in objects_b


@pytest.mark.asyncio
async def test_retrieval_does_not_share_turn_artifacts_across_agents(uma_memory) -> None:
    mem = uma_memory
    assert mem.agent_id

    mem._agent_id = "agent-a"
    await mem.process_turn(
        user_id="user:u1",
        user_msg="first",
        assistant_reply="user likes coffee.",
        session_id="shared-session",
    )

    mem._agent_id = "agent-b"
    await mem.process_turn(
        user_id="user:u1",
        user_msg="second",
        assistant_reply="user likes tea.",
        session_id="shared-session",
    )

    sem_conn = mem._stores["semantic"]._conn()
    try:
        rows = mem._stores["semantic"]._query_all(
            sem_conn,
            "SELECT id, object FROM facts WHERE owner_type=? AND owner_id=? ORDER BY id ASC",
            params=["user", "user:u1"],
            log_context="test_turn_agent_fact_ids",
        )
        fact_ids = [row["id"] for row in rows]
    finally:
        sem_conn.close()

    req_a = mem.runtime._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-a",
            request_id="req-agent-a",
            user_id="user:u1",
            session_id="shared-session",
        )
    )
    req_b = mem.runtime._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-b",
            request_id="req-agent-b",
            user_id="user:u1",
            session_id="shared-session",
        )
    )
    facts_a = await mem.memory_env.fetch_facts_by_ids(
        req_a,
        fact_ids,
        owner_type="user",
        owner_id="user:u1",
    )
    facts_b = await mem.memory_env.fetch_facts_by_ids(
        req_b,
        fact_ids,
        owner_type="user",
        owner_id="user:u1",
    )

    objects_a = {str(getattr(f, "object", "")) for f in facts_a}
    objects_b = {str(getattr(f, "object", "")) for f in facts_b}
    assert "coffee" in objects_a
    assert "tea" not in objects_a
    assert "tea" in objects_b
    assert "coffee" not in objects_b


@pytest.mark.asyncio
async def test_process_turn_rejects_legacy_turn_write_mode_without_session_id(uma_memory) -> None:
    with pytest.raises(ValueError, match="requires a non-empty session_id"):
        await uma_memory.process_turn(
            user_id="user:u1",
            user_msg="legacy",
            assistant_reply="user likes cocoa.",
            extra_meta={"legacy_turn_write_mode": True},
        )
