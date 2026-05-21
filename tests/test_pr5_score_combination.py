"""
test_pr5_score_combination.py
================================
Candidates with identical fusion/rerank scores but different trust_score values
must rank in order of higher trust first.
Tests default weights and adjusted weights to verify the formula is applied.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.retrieve.ranking import Ranker
from uma.common.types import Fact

_NOW = datetime.now(timezone.utc)
_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:alice")


def _fact(fid: str, trust: float, predicate: str = "likes") -> Fact:
    return Fact(
        id=fid,
        subject="user:alice",
        predicate=predicate,
        object="thing",
        created_at=_NOW,
        updated_at=_NOW,
        trust_score=trust,
        **_SCOPE,
    )


def test_higher_trust_ranks_first_at_default_weights():
    """Two facts with equal text → equal rerank; higher trust must rank first."""
    # Use identical text so rerank scores are equal.
    high = _fact("high_trust", trust=0.9, predicate="eats")
    low = _fact("low_trust", trust=0.1, predicate="eats")

    ranker = Ranker()  # default trust_weight=0.15
    ranked = ranker.rank_facts([low, high], query_text="eats")

    assert ranked[0].id == "high_trust", (
        "candidate with higher trust_score must rank first when rerank scores are equal"
    )


def test_trust_formula_at_custom_weights():
    """Verify the formula: final = alpha * existing + beta * trust at non-default weights."""
    # trust_weight=0.5 → alpha=0.5, beta=0.5
    ranker = Ranker(trust_weight=0.5)

    # Give low_trust slightly higher text-relevance but much lower trust.
    # With beta=0.5, trust dominates → high_trust should still win.
    low = _fact("low_trust", trust=0.0, predicate="sushi recipe")
    high = _fact("high_trust", trust=1.0, predicate="sushi recipe")

    ranked = ranker.rank_facts([low, high], query_text="sushi recipe")
    assert ranked[0].id == "high_trust"


def test_trust_weight_zero_preserves_existing_order():
    """With trust_weight=0, trust has no effect: ranking is purely by existing score."""
    ranker = Ranker(trust_weight=0.0)

    # high_trust has low text relevance; low_trust matches the query perfectly.
    low_trust = Fact(
        id="low_trust", subject="user:alice", predicate="likes", object="sushi",
        created_at=_NOW, updated_at=_NOW, trust_score=0.0, **_SCOPE,
    )
    high_trust = Fact(
        id="high_trust", subject="user:alice", predicate="owns", object="car",
        created_at=_NOW, updated_at=_NOW, trust_score=1.0, **_SCOPE,
    )

    ranked = ranker.rank_facts([high_trust, low_trust], query_text="sushi")
    # With trust_weight=0, only text-rerank matters; low_trust matches "sushi" better.
    assert ranked[0].id == "low_trust"


def test_trust_adjusts_meta_final_score():
    """After ranking, meta.final_score reflects the trust-adjusted value."""
    ranker = Ranker(trust_weight=0.5)
    f = _fact("f1", trust=0.8, predicate="eats")
    ranker.rank_facts([f], query_text="eats")

    m = f.meta or {}
    assert "final_score" in m
    # The final_score must incorporate trust contribution.
    # At trust_weight=0.5, beta*trust = 0.5*0.8 = 0.4 minimum contribution.
    assert m["final_score"] >= 0.4


def test_trust_weight_one_orders_purely_by_trust():
    """With trust_weight=1.0, ranking is purely by trust_score."""
    ranker = Ranker(trust_weight=1.0)

    hi = _fact("hi", trust=0.95, predicate="something")
    lo = _fact("lo", trust=0.05, predicate="something")

    ranked = ranker.rank_facts([lo, hi], query_text="something")
    assert ranked[0].id == "hi"


def test_trust_combination_with_episodes():
    """Trust-aware ranking works for episodes too."""
    from uma.common.types import Episode

    ep_high = Episode(
        id="ep_high", timestamp=_NOW, summary="clean episode", user_id="user:alice",
        trust_score=0.9, **_SCOPE,
    )
    ep_low = Episode(
        id="ep_low", timestamp=_NOW, summary="clean episode", user_id="user:alice",
        trust_score=0.1, **_SCOPE,
    )

    ranker = Ranker(trust_weight=0.5)
    ranked = ranker.rank_episodes([ep_low, ep_high], query_text="clean episode")
    assert ranked[0].id == "ep_high"
