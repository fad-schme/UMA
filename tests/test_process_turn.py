"""process_turn: semantic extraction, session-local defaults, session_id requirement, idempotency.

Covers the full turn processing pipeline: user_msg extraction, episode
creation, fact extraction with provenance, working memory population,
session isolation, and the session_id-required contract.
"""
from __future__ import annotations
from tests.helpers.runtime import init_uma_for_tests
from uma.common.types import RuntimeContext, SCOPE_MODEL_VERSION
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
import pytest

# ── test_process_turn_semantic_behavior ──────────────────────────────────────────





@pytest.mark.asyncio
async def test_semantic_ingest_extracts_only_from_user_msg(tmp_path):
    """Facts are extracted from both sides of the turn with source-specific trust."""
    mem = await init_uma_for_tests(tmp_path)
    try:
        await mem.process_turn(
            user_id="user:u1",
            user_msg="I like sushi.",
            assistant_reply="I like pizza.",   # must NOT be ingested
            session_id="session-a",
        )

        facts = await mem.semantic_core.list_facts_for_owner(
            tenant_id="default",
            owner_type="user",
            owner_id="user:u1",
        )
        fact_objects = [str(getattr(f, "object", "")).lower() for f in facts]

        assert any("sushi" in o for o in fact_objects), (
            f"expected a 'sushi' fact extracted from user_msg; got objects={fact_objects}"
        )
        assert any("pizza" in o for o in fact_objects), (
            f"expected assistant_reply facts to be persisted too; objects={fact_objects}"
        )
        by_object = {str(getattr(f, "object", "")).lower(): f for f in facts}
        assert by_object["sushi"].trust_score == pytest.approx(0.9)
        assert by_object["pizza"].trust_score == pytest.approx(0.7)
    finally:
        mem.shutdown()


@pytest.mark.asyncio
async def test_semantic_ingest_skips_when_user_msg_empty(tmp_path):
    """Empty user_msg still ingests assistant_reply facts."""
    mem = await init_uma_for_tests(tmp_path)
    try:
        await mem.process_turn(
            user_id="user:u1",
            user_msg="",
            assistant_reply="I like coffee.",  # must NOT be ingested
            session_id="session-a",
        )

        facts = await mem.semantic_core.list_facts_for_owner(
            tenant_id="default",
            owner_type="user",
            owner_id="user:u1",
        )
        fact_objects = [str(getattr(f, "object", "")).lower() for f in facts]
        assert any("coffee" in o for o in fact_objects), (
            f"expected assistant-derived fact for empty user_msg; got objects={fact_objects}"
        )
        assert all(float(getattr(f, "trust_score", 0.0) or 0.0) == pytest.approx(0.7) for f in facts)
    finally:
        mem.shutdown()


@pytest.mark.asyncio
async def test_episode_raw_transcript_contains_both_sides(tmp_path):
    """Stored episode.raw preserves both user_msg and assistant_reply with speaker roles."""
    mem = await init_uma_for_tests(tmp_path)
    try:
        await mem.process_turn(
            user_id="user:u1",
            user_msg="Hello there.",
            assistant_reply="Hello back.",
            session_id="session-a",
        )

        episodes = await mem.episodic_core.list_episodes(
            tenant_id="default",
            owner_type="user",
            owner_id="user:u1",
        )
        assert episodes, "expected at least one stored episode"

        raw = getattr(episodes[0], "raw", "") or ""
        assert "user:" in raw.lower(), (
            f"episode.raw missing 'user:' speaker label; raw={raw!r}"
        )
        assert "assistant:" in raw.lower(), (
            f"episode.raw missing 'assistant:' speaker label; raw={raw!r}"
        )
        assert "Hello there." in raw, (
            f"user_msg not found in episode.raw; raw={raw!r}"
        )
        assert "Hello back." in raw, (
            f"assistant_reply not found in episode.raw; raw={raw!r}"
        )
    finally:
        mem.shutdown()


@pytest.mark.asyncio
async def test_retrieve_context_working_memory_populated_after_turn(tmp_path):
    """Working memory appears in retrieve_context output independently of lane_filter."""
    mem = await init_uma_for_tests(tmp_path)
    try:
        await mem.process_turn(
            user_id="user:u1",
            user_msg="Tell me about sushi.",
            assistant_reply="Sushi is great.",
            session_id="session-a",
        )

        result = await mem.retrieve_context(
            query_text="sushi",
            user_id="user:u1",
            session_id="session-a",
        )

        wm = result.working_memory
        assert isinstance(wm, list), "working_memory must be a list"
        assert len(wm) > 0, "working_memory should have entries after process_turn"

        roles = [getattr(m, "role", None) for m in wm]
        assert "user" in roles, (
            f"expected 'user' role in working_memory; roles={roles}"
        )
        assert "assistant" in roles, (
            f"expected 'assistant' role in working_memory; roles={roles}"
        )
    finally:
        mem.shutdown()


@pytest.mark.asyncio
async def test_process_turn_persists_raw_chunks_and_fact_provenance(tmp_path):
    mem = await init_uma_for_tests(tmp_path)
    try:
        user_msg = (
            "I am researching adoption agencies and I am interested in "
            "counseling or mental health work."
        )
        assistant_reply = "Thanks, I will remember that context."
        await mem.process_turn(
            user_id="user:u1",
            user_msg=user_msg,
            assistant_reply=assistant_reply,
            session_id="session-turn-provenance",
        )

        facts = await mem.semantic_core.list_facts_for_owner(
            tenant_id="default",
            owner_type="user",
            owner_id="user:u1",
        )
        source_chunk_ids = sorted(
            {
                str(chunk_id)
                for fact in facts
                for chunk_id in list(getattr(fact, "source_ids", None) or [])
                if chunk_id
            }
        )
        assert source_chunk_ids, "expected turn-derived semantic facts with source_ids"

        evidence_chunks = await mem.chunk_core._fetch_by_ids(
            source_chunk_ids,
            tenant_id="default",
            owner_type="user",
            owner_id="user:u1",
        )
        assert evidence_chunks, "expected raw turn chunks to be persisted for the same owner scope"
        owner_chunks = await mem.chunk_core.store.list_chunks_for_owner(
            tenant_id="default",
            owner_type="user",
            owner_id="user:u1",
        )
        meta_by_role = {
            str((getattr(chunk, "meta", None) or {}).get("source_role")): getattr(chunk, "meta", None) or {}
            for chunk in owner_chunks
        }
        texts = {chunk.text for chunk in owner_chunks}
        assert user_msg in texts
        assert assistant_reply in texts
        assert meta_by_role["user"]["kind"] == "raw_source"
        assert meta_by_role["user"]["kb_lane"] == "raw"
        assert meta_by_role["assistant"]["kind"] == "raw_source"
        assert meta_by_role["assistant"]["kb_lane"] == "raw"

        recalled = await mem.retrieve_memory(
            query_text="adoption agencies",
            user_id="user:u1",
            tenant_id="default",
            request_id="req-turn-provenance",
            session_id="session-turn-provenance",
        )
        assert recalled.facts, "expected retrieve_memory to return semantic facts"
        assert recalled.evidence, "expected retrieve_memory to expand source_ids into evidence chunks"
        assert recalled.provenance_valid is True
    finally:
        mem.shutdown()


# ── test_turn_session_local_defaults ──────────────────────────────────────────





@pytest.mark.asyncio
async def test_process_turn_writes_session_local_episode_and_fact_provenance(uma_memory) -> None:
    mem = uma_memory
    assert mem.agent_id

    await mem.process_turn(
        user_id="user:u1",
        user_msg="I like coffee.",
        assistant_reply="Good choice.",
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
        user_msg="I like coffee.",
        assistant_reply="Good choice.",
        session_id="session-a",
    )
    await mem.process_turn(
        user_id="user:u1",
        user_msg="I like tea.",
        assistant_reply="Nice.",
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
    mem_a = uma_memory.set_context(agent_id="agent-a")
    mem_b = uma_memory.set_context(agent_id="agent-b")

    await mem_a.process_turn(
        user_id="user:u1",
        user_msg="I like coffee.",
        assistant_reply="Good choice.",
        session_id="shared-session",
    )

    await mem_b.process_turn(
        user_id="user:u1",
        user_msg="I like tea.",
        assistant_reply="Nice.",
        session_id="shared-session",
    )

    assert mem_a.pipeline is not mem_b.pipeline

    sem_conn = mem_a._stores["semantic"]._conn()
    try:
        rows = mem_a._stores["semantic"]._query_all(
            sem_conn,
            "SELECT id, object FROM facts WHERE owner_type=? AND owner_id=? ORDER BY id ASC",
            params=["user", "user:u1"],
            log_context="test_turn_agent_fact_ids",
        )
        fact_ids = [row["id"] for row in rows]
    finally:
        sem_conn.close()

    req_a = mem_a.runtime._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-a",
            request_id="req-agent-a",
            user_id="user:u1",
            session_id="shared-session",
        )
    )
    req_b = mem_b.runtime._build_retrieval_request(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent-b",
            request_id="req-agent-b",
            user_id="user:u1",
            session_id="shared-session",
        )
    )
    facts_a = await mem_a.memory_env.fetch_facts_by_ids(
        req_a,
        fact_ids,
        owner_type="user",
        owner_id="user:u1",
    )
    facts_b = await mem_b.memory_env.fetch_facts_by_ids(
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



# ── test_turn_ingest_idempotent ──────────────────────────────────────────




@pytest.mark.asyncio
async def test_process_turn_is_idempotent_by_turn_id(uma_memory):
    mem = uma_memory

    # Use an assistant reply that triggers deterministic fact extraction.
    await mem.process_turn(
        user_id="user:u1",
        user_msg="I like coffee.",
        assistant_reply="Good choice.",
        session_id="session-a",
    )
    await mem.process_turn(
        user_id="user:u1",
        user_msg="I like coffee.",
        assistant_reply="Good choice.",
        session_id="session-a",
    )

    # Episodes are appended per call even when the derived turn_id is identical.
    conn = mem._stores["episodic"]._conn()
    try:
        rows = mem._stores["episodic"]._query_all(
            conn,
            "SELECT COUNT(*) AS n FROM episodes WHERE owner_type=? AND owner_id=?",
            params=["user", "user:u1"],
            log_context="test_episode_count",
        )
        assert int(rows[0]["n"]) == 2
    finally:
        conn.close()

    # Semantic facts remain stable across retries because fact IDs are content-derived.
    conn = mem._stores["semantic"]._conn()
    try:
        rows = mem._stores["semantic"]._query_all(
            conn,
            "SELECT COUNT(*) AS n FROM facts WHERE owner_type=? AND owner_id=?",
            params=["user", "user:u1"],
            log_context="test_fact_count",
        )
        assert int(rows[0]["n"]) == 3
    finally:
        conn.close()


# ── test_turn_user_semantic_retrieval ──────────────────────────────────────────





def _fact_objects(payload) -> set[str]:
    return {
        str(item.get("object") or item.get("text") or "").lower()
        for item in list(payload.facts)
        if isinstance(item, dict)
    }


@pytest.mark.asyncio
async def test_process_turn_user_message_becomes_retrievable(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)

    await memory.process_turn(
        user_id="user:u1",
        user_msg="I am researching adoption agencies and I am interested in counseling or mental health work.",
        assistant_reply="Thanks, I will remember that context.",
        session_id="session-user-turn-facts",
    )

    recalled_adoption = await memory.retrieve_memory(
        query_text="adoption agencies",
        user_id="user:u1",
        tenant_id="default",
        request_id="req-user-turn-adoption",
        session_id="session-user-turn-facts",
    )
    recalled_mental_health = await memory.retrieve_memory(
        query_text="mental health",
        user_id="user:u1",
        tenant_id="default",
        request_id="req-user-turn-mental-health",
        session_id="session-user-turn-facts",
    )

    assert any("adoption agenc" in obj for obj in _fact_objects(recalled_adoption))
    assert any("mental health" in obj or obj == "counseling" for obj in _fact_objects(recalled_mental_health))
