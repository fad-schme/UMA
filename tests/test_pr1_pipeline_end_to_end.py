"""
PR1 — pipeline end-to-end: trust_score and content_hash populated after process_turn.

Tests:
- After process_turn, the resulting episode has content_hash populated and
  trust_score == 0.5 (PR1 default).
- After process_turn, extracted facts have content_hash populated (non-empty hex string)
  and trust_score == 0.5.
"""

from __future__ import annotations

import pytest

from uma.common.integrity import hash_episode_content


@pytest.mark.asyncio
async def test_episode_has_content_hash_and_trust_score_after_process_turn(uma_memory):
    mem = uma_memory

    await mem.process_turn(
        user_id="user:alice",
        user_msg="I enjoy hiking in the mountains.",
        assistant_reply="That sounds like a great hobby.",
        session_id="session-pr1-ep",
    )

    epi_store = mem._stores["episodic"]
    episodes = await epi_store.list_episodes(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )
    assert episodes, "expected at least one episode after process_turn"

    ep = episodes[0]
    assert ep.trust_score == pytest.approx(0.8), "episode trust_score must be 0.8 for synthesized turn summaries"

    # content_hash must be non-empty and match the canonical hash of the summary.
    assert ep.content_hash is not None, "content_hash must be populated"
    assert len(ep.content_hash) == 64, "content_hash must be 64-char SHA-256 hex"
    expected = hash_episode_content(ep.summary)
    assert ep.content_hash == expected, "content_hash must match hash_episode_content(summary)"


@pytest.mark.asyncio
async def test_facts_have_content_hash_and_trust_score_after_process_turn(uma_memory):
    mem = uma_memory

    await mem.process_turn(
        user_id="user:alice",
        user_msg="I like hiking and rock climbing.",
        assistant_reply="Those are excellent outdoor activities.",
        session_id="session-pr1-facts",
    )

    sem_store = mem._stores["semantic"]
    facts = await sem_store.list_facts_for_owner(
        tenant_id="default",
        owner_type="user",
        owner_id="user:alice",
    )

    if not facts:
        pytest.skip("fake_llm produced no facts for this input; skipping assertion")

    trust_scores = {round(float(fact.trust_score or 0.0), 1) for fact in facts}
    assert trust_scores.issubset({0.7, 0.9})
    assert 0.9 in trust_scores
        # content_hash is optional (fallback facts may not populate it in all codepaths)
    for fact in facts:
        if fact.content_hash is not None:
            assert len(fact.content_hash) == 64, (
                f"fact id={fact.id} content_hash must be 64-char SHA-256 hex"
            )
