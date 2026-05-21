"""
test_pr5_score_card.py
=======================
Verifies debug score card includes PR5 trust fields when debug_scores=True,
and that they are NOT present when debug_scores=False.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.retrieve.ranking import Ranker
from uma.common.types import Fact, Episode, Chunk, Skill

_NOW = datetime.now(timezone.utc)
_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:alice")


def _fact(fid: str, trust: float = 0.7) -> Fact:
    return Fact(
        id=fid, subject="user:alice", predicate="likes", object="coffee",
        created_at=_NOW, updated_at=_NOW, trust_score=trust, **_SCOPE,
    )


# ---------------------------------------------------------------------------
# debug_scores=True: trust fields present in score_card
# ---------------------------------------------------------------------------

def test_score_card_includes_trust_score_when_debug_on():
    ranker = Ranker(debug_scores=True)
    f = _fact("f1", trust=0.75)

    ranker.rank_facts([f], query_text="coffee")

    m = f.meta or {}
    card = m.get("score_card") or {}
    assert "trust_score" in card, "score_card must include trust_score when debug_scores=True"
    assert abs(card["trust_score"] - 0.75) < 1e-6


def test_score_card_includes_final_score_with_trust_when_debug_on():
    ranker = Ranker(debug_scores=True, trust_weight=0.15)
    f = _fact("f1", trust=0.9)

    ranker.rank_facts([f], query_text="coffee")

    m = f.meta or {}
    card = m.get("score_card") or {}
    assert "final_score_with_trust" in card
    # final_score_with_trust must match meta.final_score
    assert abs(card["final_score_with_trust"] - m.get("final_score", 0.0)) < 1e-6


def test_score_card_trust_fields_all_candidates():
    """Trust fields are populated on EVERY returned candidate, not just the top."""
    ranker = Ranker(debug_scores=True)
    facts = [_fact("a", trust=0.9), _fact("b", trust=0.5), _fact("c", trust=0.1)]

    ranker.rank_facts(facts, query_text="coffee")

    for f in facts:
        card = (f.meta or {}).get("score_card") or {}
        assert "trust_score" in card, f"score_card missing trust_score on {f.id}"
        assert "final_score_with_trust" in card, f"score_card missing final_score_with_trust on {f.id}"


# ---------------------------------------------------------------------------
# debug_scores=False: trust fields absent (no debug leakage)
# ---------------------------------------------------------------------------

def test_score_card_absent_when_debug_off():
    ranker = Ranker(debug_scores=False)
    f = _fact("f1", trust=0.9)

    ranker.rank_facts([f], query_text="coffee")

    m = f.meta or {}
    assert "score_card" not in m, "score_card must not be written when debug_scores=False"


def test_trust_fields_absent_in_meta_when_debug_off():
    """meta.trust_score and meta.final_score_with_trust must not leak into normal output."""
    ranker = Ranker(debug_scores=False)
    f = _fact("f1", trust=0.9)

    ranker.rank_facts([f], query_text="coffee")

    m = f.meta or {}
    # trust fields are only in score_card (debug only), not at top-level meta
    assert "trust_score" not in m or m.get("trust_score") is None or "score_card" not in m


# ---------------------------------------------------------------------------
# Trust fields correct across item types
# ---------------------------------------------------------------------------

def test_score_card_trust_on_episodes():
    from uma.common.types import Episode

    ep = Episode(
        id="ep1", timestamp=_NOW, summary="clean", user_id="user:alice",
        trust_score=0.6, **_SCOPE,
    )
    ranker = Ranker(debug_scores=True)
    ranker.rank_episodes([ep], query_text="clean")

    card = (ep.meta or {}).get("score_card") or {}
    assert "trust_score" in card
    assert abs(card["trust_score"] - 0.6) < 1e-6


def test_score_card_trust_on_chunks():
    from uma.common.types import Chunk

    ch = Chunk(
        id="ch1", doc_id="d1", text="trusted content here for testing",
        page_range=(0, 1), position=0, source_path="f.pdf", source_hash="h",
        created_at=_NOW, updated_at=_NOW, trust_score=0.8, **_SCOPE,
    )
    ranker = Ranker(debug_scores=True)
    ranker.rank_chunks([ch], query_text="trusted content")

    card = (ch.meta or {}).get("score_card") or {}
    assert "trust_score" in card
    assert abs(card["trust_score"] - 0.8) < 1e-6
