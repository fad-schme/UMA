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

    def score(self, fact: Fact) -> float:
        now = datetime.now(timezone.utc)
        updated = fact.updated_at.replace(tzinfo=timezone.utc)

        # Days since last update
        age_days = max(0.0, (now - updated).total_seconds() / 86400.0)

        base_conf = fact.confidence if fact.confidence is not None else 0.5
        # Half-life ~180 days
        decay_factor = 0.5 ** (age_days / 180.0)

        salience = max(0.0, min(1.0, float(base_conf) * float(decay_factor)))
        return salience