"""
semantic/scorer.py
==================

SalienceScorer computes a score in [0, 1] for each Fact.

Factors:
- confidence
- recency (time-decay function)
- predicate importance weighting

Extend this file if you want a richer salience model.

Coding agent instructions
-------------------------
- Extend predicate weights for your domain.
- Consider adding cross-referencing, frequency, or usage stats.

NOTE:
-------------------------
UMA is data-agnostic. This scorer must not hardcode domain predicates.
If you ever want predicate weighting, it should be provided via configuration,
not embedded as constants here.

"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from uma.common.types import Fact

logger = logging.getLogger(__name__)


class SalienceScorer:
    """
    Compute a salience score for a Fact.
    """

    def __init__(self, decay_half_life_days: float = 180.0) -> None:
        self._half_life = max(1.0, float(decay_half_life_days))

    def score(self, fact: Fact, *, durability: float = 1.0) -> float:
        """Score a fact's salience.

        `durability` is an optional [0, 1] multiplier for how likely the fact
        is to remain relevant over time, independent of `confidence` (which
        measures extraction certainty, not memorability). A transient detail
        the extractor is fully confident it read correctly ("waiting for the
        bus") still shouldn't score as salient as a durable one ("has a
        long-term career goal") — confidence alone can't tell them apart, so
        callers with a per-fact durability signal (e.g. an LLM-provided
        durability flag at extraction time) should pass it here rather than
        folding it into `confidence` itself.
        """
        now = datetime.now(timezone.utc)
        updated = fact.updated_at.replace(tzinfo=timezone.utc)

        age_days = max(0.0, (now - updated).total_seconds() / 86400.0)

        base_conf = fact.confidence if fact.confidence is not None else 0.5
        decay_factor = 0.5 ** (age_days / self._half_life)
        durability_factor = max(0.0, min(1.0, float(durability)))

        salience = max(0.0, min(1.0, float(base_conf) * float(decay_factor) * durability_factor))
        return salience
