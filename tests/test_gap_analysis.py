"""Gap analysis: stale and thinly-supported facts surfaced on retrieve_memory.

Gap analysis is reporting, not filtering — these tests pin that a flagged fact
is still returned, and that the signals are computed from the raw domain
objects (where `created_at` and `trust_score` live) rather than the serialized
projections, which drop both.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from uma.common.types import Fact
from uma.common.types.types_chunk import Chunk
from uma.retrieve.gaps import (
    DEFAULT_MAX_SUPPORT_AGE_DAYS,
    DEFAULT_MIN_SUPPORT_TRUST,
    GAP_STALE_SUPPORT,
    GAP_WEAK_SUPPORT,
    assess_gaps,
    gap_thresholds,
)


from tests.helpers.runtime import TEST_AGENT_ID

AGENT_ID = TEST_AGENT_ID
NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


def _chunk(chunk_id: str, *, age_days: int = 1, trust: float = 0.9) -> Chunk:
    created = NOW - timedelta(days=age_days)
    return Chunk(
        id=chunk_id,
        doc_id="doc-1",
        text="supporting text",
        page_range=(1, 1),
        position=0,
        source_path="doc-1.txt",
        source_hash="hash-1",
        created_at=created,
        updated_at=created,
        owner_type="user",
        owner_id="user:u1",
        trust_score=trust,
    )


def _fact(fact_id: str, *, source_ids: list[str]) -> Fact:
    return Fact(
        id=fact_id,
        subject="team",
        predicate="USES",
        object="kubernetes",
        created_at=NOW,
        updated_at=NOW,
        source_ids=source_ids,
        owner_type="user",
        owner_id="user:u1",
    )


def test_fresh_well_supported_fact_produces_no_gap() -> None:
    gaps = assess_gaps(
        facts=[_fact("f1", source_ids=["c1", "c2"])],
        chunks=[_chunk("c1"), _chunk("c2")],
        now=NOW,
    )
    assert gaps == []


def test_stale_support_is_flagged_from_the_newest_chunk() -> None:
    """Staleness keys on the *freshest* support, not the oldest."""
    old_only = assess_gaps(
        facts=[_fact("f1", source_ids=["c1", "c2"])],
        chunks=[_chunk("c1", age_days=900), _chunk("c2", age_days=800)],
        now=NOW,
    )
    assert [g["reason"] for g in old_only] == [GAP_STALE_SUPPORT]
    assert old_only[0]["fact_id"] == "f1"
    assert old_only[0]["newest_support_age_days"] == 800
    assert old_only[0]["support_count"] == 2

    # One recent chunk is enough to clear the flag.
    with_recent = assess_gaps(
        facts=[_fact("f1", source_ids=["c1", "c2"])],
        chunks=[_chunk("c1", age_days=900), _chunk("c2", age_days=3)],
        now=NOW,
    )
    assert with_recent == []


def test_weak_support_requires_both_single_source_and_low_trust() -> None:
    single_low = assess_gaps(
        facts=[_fact("f1", source_ids=["c1"])],
        chunks=[_chunk("c1", trust=0.51)],
        now=NOW,
    )
    assert [g["reason"] for g in single_low] == [GAP_WEAK_SUPPORT]
    assert single_low[0]["support_count"] == 1
    assert single_low[0]["support_trust"] == pytest.approx(0.51)

    # A single high-trust source is not a gap.
    assert assess_gaps(
        facts=[_fact("f1", source_ids=["c1"])],
        chunks=[_chunk("c1", trust=0.95)],
        now=NOW,
    ) == []

    # Two low-trust sources corroborate each other — also not a gap.
    assert assess_gaps(
        facts=[_fact("f1", source_ids=["c1", "c2"])],
        chunks=[_chunk("c1", trust=0.51), _chunk("c2", trust=0.51)],
        now=NOW,
    ) == []


def test_a_fact_can_be_both_stale_and_weakly_supported() -> None:
    gaps = assess_gaps(
        facts=[_fact("f1", source_ids=["c1"])],
        chunks=[_chunk("c1", age_days=900, trust=0.51)],
        now=NOW,
    )
    assert {g["reason"] for g in gaps} == {GAP_STALE_SUPPORT, GAP_WEAK_SUPPORT}


def test_unsupported_fact_is_not_reported_here() -> None:
    """Unsupported claims are provenance's job, not the gap report's."""
    gaps = assess_gaps(
        facts=[_fact("f1", source_ids=["missing-chunk"])],
        chunks=[_chunk("c1")],
        now=NOW,
    )
    assert gaps == []


def test_future_timestamp_is_not_a_staleness_signal() -> None:
    """Clock skew must not manufacture a gap (nor a negative age)."""
    gaps = assess_gaps(
        facts=[_fact("f1", source_ids=["c1"])],
        chunks=[_chunk("c1", age_days=-30)],
        now=NOW,
    )
    assert gaps == []


def test_assess_gaps_is_total_on_malformed_input() -> None:
    """A malformed record is skipped, never reported as a gap, never raises."""
    assert assess_gaps(facts=[], chunks=[], now=NOW) == []
    assert assess_gaps(facts=[_fact("f1", source_ids=["c1"])], chunks=[], now=NOW) == []
    assert assess_gaps(facts=[object()], chunks=[_chunk("c1")], now=NOW) == []


def test_gap_thresholds_fall_back_to_defaults() -> None:
    assert gap_thresholds(None) == (DEFAULT_MAX_SUPPORT_AGE_DAYS, DEFAULT_MIN_SUPPORT_TRUST)

    class _Cfg:
        gap_max_support_age_days = 30
        gap_min_support_trust = 0.8

    assert gap_thresholds(_Cfg()) == (30, 0.8)

    class _Bad:
        gap_max_support_age_days = "not-a-number"
        gap_min_support_trust = 0.8

    assert gap_thresholds(_Bad()) == (DEFAULT_MAX_SUPPORT_AGE_DAYS, DEFAULT_MIN_SUPPORT_TRUST)


@pytest.mark.asyncio
async def test_retrieve_memory_surfaces_gaps_without_dropping_the_fact(uma_memory) -> None:
    """End-to-end: gaps reach MemoryResult and the flagged fact still returns."""
    memory = uma_memory
    await memory.process_turn(
        user_id="user:u1",
        user_msg="what does the team use?",
        assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
        session_id="session-gaps",
        agent_id=AGENT_ID,
    )
    await memory.pipeline.await_pending_background()

    result = await memory.retrieve_memory(
        query_text="what does the team use?",
        user_id="user:u1",
        session_id="session-gaps",
        agent_id=AGENT_ID,
    )

    # Contract: the field is always present and always a list.
    assert isinstance(result.gaps, list)
    for gap in result.gaps:
        assert gap["reason"] in {GAP_STALE_SUPPORT, GAP_WEAK_SUPPORT}
        assert "fact_id" in gap and "text" in gap

    # Freshly ingested evidence must not be flagged stale.
    assert not any(g["reason"] == GAP_STALE_SUPPORT for g in result.gaps)


@pytest.mark.asyncio
async def test_retrieve_memory_gap_thresholds_come_from_retrieval_config(uma_memory) -> None:
    """The runtime reads thresholds off config, and flagged facts still return.

    Seeds a fact against a real ingested chunk rather than relying on the test
    extractor, then raises the trust bar above that chunk's score. Without the
    config actually being read, the 0.6 default would flag nothing and this
    could not tell a wired path from a hardcoded one.
    """
    memory = uma_memory
    query = "what does the team use?"
    await memory.process_turn(
        user_id="user:u1",
        user_msg=query,
        assistant_reply="the team uses kubernetes cluster orchestration for production workloads.",
        session_id="session-gaps-cfg",
        agent_id=AGENT_ID,
    )
    await memory.pipeline.await_pending_background()

    seed = await memory.retrieve_memory(
        query_text=query, user_id="user:u1", session_id="session-gaps-cfg", include_debug=True,
        agent_id=AGENT_ID,
    )
    chunk_ids = [
        getattr(chunk, "id", None)
        for chunk in ((seed.debug or {}).get("evidence") or [])
        if getattr(chunk, "id", None)
    ]
    assert chunk_ids, "the turn must have produced at least one chunk to support a fact"

    now = datetime.now(timezone.utc)
    supported = Fact(
        id="fact_seeded_gap",
        subject="team",
        predicate="USES",
        object="kubernetes cluster orchestration for production workloads",
        created_at=now,
        updated_at=now,
        source_ids=[chunk_ids[0]],
        owner_type="user",
        owner_id="user:u1",
        confidence=0.9,
        salience=0.9,
    )
    embedding = (await memory.embedder.embed([str(supported.object)]))[0]
    await memory.semantic_core.upsert_fact(supported, embedding)

    baseline = await memory.retrieve_memory(
        query_text=query, user_id="user:u1", session_id="session-gaps-cfg",
        agent_id=AGENT_ID,
    )
    assert baseline.facts, "seeded fact must be retrievable for the guard to be exercised"
    assert baseline.gaps == [], "a single high-trust source is not a gap at the default bar"

    # Raise the bar above the supporting chunk's trust score.
    memory.retrieval_cfg.gap_min_support_trust = 0.95

    flagged = await memory.retrieve_memory(
        query_text=query, user_id="user:u1", session_id="session-gaps-cfg",
        agent_id=AGENT_ID,
    )
    weak = [g for g in flagged.gaps if g["reason"] == GAP_WEAK_SUPPORT]
    assert weak, "raising the trust bar must flag the single-source fact"
    assert weak[0]["fact_id"] == "fact_seeded_gap"
    assert weak[0]["support_count"] == 1
    # Reporting only — the facts are still returned untouched.
    assert len(flagged.facts) == len(baseline.facts)
