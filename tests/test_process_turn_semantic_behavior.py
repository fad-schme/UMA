"""
Tests for process_turn semantic memory behavior.

Invariants verified:
- Semantic facts are extracted from both user_msg and assistant_reply.
- Empty user_msg still allows assistant_reply-derived semantic facts.
- The deferred path ingests both sides of the turn.
- Episodes store the full transcript (user_msg + assistant_reply with speaker roles).
- Working memory is a distinct, un-lane-filtered section in retrieve_context output.
"""
from __future__ import annotations

import types

import pytest

from tests.helpers.runtime import init_uma_for_tests


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
async def test_deferred_path_ingests_user_msg_not_assistant_reply(tmp_path):
    """With defer_post_turn enabled, draining the queue extracts facts from both sides."""
    mem = await init_uma_for_tests(tmp_path)
    try:
        mem.pipeline_cfg = types.SimpleNamespace(
            defer_post_turn=True,
            post_turn_queue_max=50,
        )

        await mem.process_turn(
            user_id="user:u1",
            user_msg="I like sushi.",
            assistant_reply="I like pizza.",   # must NOT be ingested
            session_id="session-a",
        )

        from uma.ingest.pipeline import MemoryPipeline
        pipeline = mem.pipeline
        assert isinstance(pipeline, MemoryPipeline)

        processed = await pipeline.process_post_turn_queue()
        assert processed == 1, f"expected 1 deferred task; got {processed}"

        facts = await mem.semantic_core.list_facts_for_owner(
            tenant_id="default",
            owner_type="user",
            owner_id="user:u1",
        )
        fact_objects = [str(getattr(f, "object", "")).lower() for f in facts]

        assert any("sushi" in o for o in fact_objects), (
            f"expected 'sushi' fact via deferred user_msg ingest; got objects={fact_objects}"
        )
        assert any("pizza" in o for o in fact_objects), (
            f"expected deferred assistant_reply ingest too; objects={fact_objects}"
        )
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

        wm = result.get("working_memory", [])
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
        assert recalled["facts"], "expected retrieve_memory to return semantic facts"
        assert recalled["evidence"], "expected retrieve_memory to expand source_ids into evidence chunks"
        assert recalled["provenance_valid"] is True
    finally:
        mem.shutdown()
