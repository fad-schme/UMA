"""
PR2 — trust_score end-to-end: classifier-derived values after process_turn and ingest_document.

Tests:
- process_turn produces an episode with trust_score == 0.7 (turn_assistant, authenticated session).
- process_turn produces user facts with trust_score == 0.9 (turn_user, authenticated session).
- ingest_document produces chunks with trust_score == 0.7 (document source).
"""
from __future__ import annotations

import pytest

from tests.helpers.runtime import init_uma_for_tests


_FIXTURE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This document is used to verify PR2 trust scoring on ingest. "
    "Every stored chunk must carry a classifier-derived trust score. "
    "Memory systems need reliable trust tracking per artifact source."
)


@pytest.fixture
def fixture_doc(tmp_path) -> str:
    doc_path = tmp_path / "pr2_test_doc.txt"
    doc_path.write_text(_FIXTURE_TEXT, encoding="utf-8")
    return str(doc_path)


@pytest.mark.asyncio
async def test_process_turn_episode_trust_score(uma_memory):
    """Episode from a turn with a session_id must have trust_score == 0.7."""
    mem = uma_memory

    await mem.process_turn(
        user_id="user:alice",
        user_msg="I enjoy hiking in the mountains.",
        assistant_reply="That sounds like a great hobby.",
        session_id="session-pr2-ep",
    )

    epi_store = mem._stores["episodic"]
    episodes = await epi_store.list_episodes(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert episodes, "expected at least one episode after process_turn"

    ep = episodes[0]
    assert ep.trust_score == pytest.approx(0.7), (
        f"episode trust_score must be 0.7 (turn_assistant, authenticated); got {ep.trust_score}"
    )


@pytest.mark.asyncio
async def test_process_turn_facts_trust_score(uma_memory):
    """Facts extracted from a turn with a session_id must have trust_score == 0.9."""
    mem = uma_memory

    await mem.process_turn(
        user_id="user:alice",
        user_msg="I like hiking and rock climbing.",
        assistant_reply="Those are excellent outdoor activities.",
        session_id="session-pr2-facts",
    )

    sem_store = mem._stores["semantic"]
    facts = await sem_store.list_facts_for_owner(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )

    if not facts:
        pytest.skip("fake_llm produced no facts for this input; skipping assertion")

    for fact in facts:
        assert fact.trust_score == pytest.approx(0.9), (
            f"fact id={fact.id} must have trust_score=0.9 (turn_user, authenticated); got {fact.trust_score}"
        )


@pytest.mark.asyncio
async def test_ingest_document_chunks_trust_score(tmp_path, fixture_doc):
    """Chunks from ingest_document must have trust_score == 0.7 (document source)."""
    mem = await init_uma_for_tests(tmp_path)
    try:
        report = await mem.ingest_document(
            fixture_doc,
            owner_type="user",
            owner_id="user:alice",
        )
        assert report.chunks_created > 0, "expected at least one chunk"

        chunk_store = mem._stores["chunk"]
        conn = chunk_store._conn()
        try:
            rows = chunk_store._query_all(
                conn,
                "SELECT id, trust_score FROM chunks WHERE owner_id = ?",
                params=["user:alice"],
                log_context="test_pr2_ingest_trust_score",
            )
        finally:
            conn.close()

        assert rows, "expected chunk rows in DB"
        for row in rows:
            assert row["trust_score"] is not None
            assert abs(float(row["trust_score"]) - 0.7) < 1e-6, (
                f"chunk {row['id']} must have trust_score=0.7 (document source); got {row['trust_score']}"
            )
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass
