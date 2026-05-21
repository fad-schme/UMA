"""
test_pr5_threshold_filter.py
================================
Verifies the min_trust_score threshold filter:
- min_trust_score=0.0 (default): all non-quarantined candidates pass.
- min_trust_score=0.5: candidates below threshold are dropped.
- Candidate exactly at the threshold passes (inclusive comparison >=).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.retrieve.ranking import Ranker
from uma.common.types import Fact, Episode, Skill, Chunk

_NOW = datetime.now(timezone.utc)
_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:alice")


def _fact(fid: str, trust: float) -> Fact:
    return Fact(
        id=fid, subject="user:alice", predicate="likes", object="thing",
        created_at=_NOW, updated_at=_NOW, trust_score=trust, **_SCOPE,
    )


# ---------------------------------------------------------------------------
# Default threshold (0.0): no filtering
# ---------------------------------------------------------------------------

def test_default_min_trust_passes_all_candidates():
    """With min_trust_score=0.0 (default), every non-quarantined candidate passes."""
    ranker = Ranker()  # min_trust_score=0.0

    facts = [_fact("a", 0.9), _fact("b", 0.5), _fact("c", 0.0)]
    result = ranker.rank_facts(facts, query_text="thing")

    ids = {f.id for f in result}
    assert ids == {"a", "b", "c"}, "all candidates must pass the default threshold"


# ---------------------------------------------------------------------------
# min_trust_score=0.5 drops below-threshold candidates
# ---------------------------------------------------------------------------

def test_threshold_drops_low_trust_facts():
    ranker = Ranker(min_trust_score=0.5)

    above = _fact("above", trust=0.8)
    exact = _fact("exact", trust=0.5)
    below = _fact("below", trust=0.3)

    result = ranker.rank_facts([above, exact, below], query_text="thing")
    ids = {f.id for f in result}

    assert "above" in ids
    assert "exact" in ids
    assert "below" not in ids, "candidate below threshold must be dropped"


def test_threshold_inclusive_at_exact_boundary():
    """Candidate with trust_score == min_trust_score must NOT be dropped (>= comparison)."""
    ranker = Ranker(min_trust_score=0.5)
    exact = _fact("exact", trust=0.5)

    result = ranker.rank_facts([exact], query_text="thing")
    assert len(result) == 1 and result[0].id == "exact"


def test_threshold_drops_zero_trust():
    ranker = Ranker(min_trust_score=0.5)
    zero = _fact("zero", trust=0.0)
    high = _fact("high", trust=0.9)

    result = ranker.rank_facts([zero, high], query_text="thing")
    assert [f.id for f in result] == ["high"]


def test_threshold_all_dropped_returns_empty():
    ranker = Ranker(min_trust_score=0.9)
    facts = [_fact("a", 0.1), _fact("b", 0.2), _fact("c", 0.3)]

    result = ranker.rank_facts(facts, query_text="thing")
    assert result == []


# ---------------------------------------------------------------------------
# Filter applies to all item types
# ---------------------------------------------------------------------------

def test_threshold_filter_on_episodes():
    from uma.common.types import Episode

    ep_ok = Episode(
        id="ep_ok", timestamp=_NOW, summary="ok",
        user_id="user:alice", trust_score=0.8, **_SCOPE,
    )
    ep_bad = Episode(
        id="ep_bad", timestamp=_NOW, summary="bad",
        user_id="user:alice", trust_score=0.2, **_SCOPE,
    )

    ranker = Ranker(min_trust_score=0.5)
    result = ranker.rank_episodes([ep_ok, ep_bad], query_text="ok")
    assert [e.id for e in result] == ["ep_ok"]


def test_threshold_filter_on_skills():
    from uma.common.types import Skill

    sk_ok = Skill(
        id="sk_ok", name="clean_skill", description="fine",
        created_at=_NOW, updated_at=_NOW, trust_score=0.7, **_SCOPE,
    )
    sk_bad = Skill(
        id="sk_bad", name="bad_skill", description="untrusted",
        created_at=_NOW, updated_at=_NOW, trust_score=0.1, **_SCOPE,
    )

    ranker = Ranker(min_trust_score=0.5)
    result = ranker.rank_skills([sk_ok, sk_bad], query_text="skill")
    assert [s.id for s in result] == ["sk_ok"]


def test_threshold_filter_on_chunks():
    from uma.common.types import Chunk

    ch_ok = Chunk(
        id="ch_ok", doc_id="d1", text="trusted content",
        page_range=(0, 1), position=0, source_path="f.pdf", source_hash="h",
        created_at=_NOW, updated_at=_NOW, trust_score=0.9, **_SCOPE,
    )
    ch_bad = Chunk(
        id="ch_bad", doc_id="d1", text="untrusted content",
        page_range=(1, 2), position=1, source_path="f.pdf", source_hash="h",
        created_at=_NOW, updated_at=_NOW, trust_score=0.2, **_SCOPE,
    )

    ranker = Ranker(min_trust_score=0.5)
    result = ranker.rank_chunks([ch_ok, ch_bad], query_text="content")
    assert [c.id for c in result] == ["ch_ok"]


# ---------------------------------------------------------------------------
# Truncation is independent of filtering
# ---------------------------------------------------------------------------

def test_filter_runs_before_truncation():
    """After filter drops below-threshold items, truncate operates on the smaller pool."""
    ranker = Ranker(min_trust_score=0.5)

    facts = [_fact("a", 0.9), _fact("b", 0.8), _fact("c", 0.1), _fact("d", 0.0)]
    filtered = ranker.rank_facts(facts, query_text="thing")
    truncated = ranker.truncate(filtered, k=1)

    # Only "a" and "b" should survive the filter; truncate gives us the top 1.
    assert len(truncated) == 1
    assert truncated[0].trust_score >= 0.5
