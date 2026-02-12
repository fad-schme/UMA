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
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict

from ...types import Fact

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
        decay_factor = 0.5 ** (age_days / 180.0)

        predicate_weights: Dict[str, float] = {
            "prefers": 1.2,
            "likes": 1.1,
            "dislikes": 1.1,
            "works_on": 1.0,
        }
        weight = predicate_weights.get(fact.predicate, 1.0)

        salience = max(0.0, min(1.0, base_conf * decay_factor * weight))

        # logger.debug(
        #     "Salience: id=%s conf=%.2f age=%.1f decay=%.2f weight=%.2f -> %.3f",
        #     fact.id,
        #     base_conf,
        #     age_days,
        #     decay_factor,
        #     weight,
        #     salience,
        # )

        return salience