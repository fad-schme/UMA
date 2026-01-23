"""
PromotionPolicy v1 — User/Project → Agent Knowledge Promotion

This module defines a conservative, rule-based promotion policy
for elevating semantic facts into the agent's global knowledge base.

Design goals
------------
- Explicit and auditable promotion
- No silent learning
- No LLM usage (v1)
- Safe defaults

Promotion is a POLICY decision, not a storage concern.
"""

from __future__ import annotations

import logging
from typing import Iterable, Set

from ...types_fact import Fact

logger = logging.getLogger(__name__)


class PromotionPolicy:
    """
    Rule-based promotion policy (v1).

    A fact is eligible for promotion if:
      - it is user/project scoped
      - it has sufficient salience
      - it has sufficient confidence
      - its predicate is not ephemeral
    """
    max_promotions_per_turn: int = 5

    def is_enabled(self) -> bool:
        """Return True if promotion is enabled."""
        return True

    def __init__(
        self,
        agent_id: str,
        min_salience: float = 0.7,
        min_confidence: float = 0.8,
        blocked_predicates: Iterable[str] | None = None,
    ) -> None:
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("PromotionPolicy requires a non-empty agent_id")

        self.agent_id = agent_id
        self.min_salience = float(min_salience)
        self.min_confidence = float(min_confidence)
        self.blocked_predicates: Set[str] = {
            p.lower() for p in (blocked_predicates or [])
        }

        # Default blocked predicates (ephemeral / personal)
        self.blocked_predicates |= {
            "said",
            "asked",
            "mentioned",
            "felt",
            "thought",
            "wanted",
            "liked",      # preferences should not auto-promote
            "disliked",
        }

        logger.info(
            "PromotionPolicy initialized agent_id=%s min_salience=%.2f min_confidence=%.2f",
            self.agent_id,
            self.min_salience,
            self.min_confidence,
        )

    # ------------------------------------------------------------------ #
    # Eligibility
    # ------------------------------------------------------------------ #

    def is_eligible(self, fact: Fact) -> bool:
        """
        Check whether a fact should be promoted to agent scope.

        Returns
        -------
        bool
            True if eligible, False otherwise.
        """
        # Already agent knowledge
        if fact.owner_type == "agent":
            return False

        # Must be user or project scoped
        if fact.owner_type not in ("user", "project"):
            return False

        # Salience threshold
        if fact.salience < self.min_salience:
            return False

        # Confidence threshold (None treated as low confidence)
        if fact.confidence is None or fact.confidence < self.min_confidence:
            return False

        # Predicate must not be ephemeral
        if fact.predicate.lower() in self.blocked_predicates:
            return False

        # Object must be serializable / stable
        try:
            _ = str(fact.object)
        except Exception:
            return False

        return True

    # ------------------------------------------------------------------ #
    # Promotion
    # ------------------------------------------------------------------ #

    def promote(self, fact: Fact) -> Fact:
        """
        Return a NEW Fact promoted to agent scope.

        IMPORTANT:
        - Does NOT modify the original fact
        - Preserves provenance via meta
        """
        if not self.is_eligible(fact):
            raise ValueError("Fact is not eligible for promotion")

        promoted = Fact(
            id=fact.id,
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            created_at=fact.created_at,
            updated_at=fact.updated_at,
            source_ids=list(fact.source_ids),
            confidence=fact.confidence,
            meta=dict(fact.meta),
            salience=fact.salience,
            owner_type="agent",
            owner_id=self.agent_id,
        )

        # Provenance tracking
        promoted.meta["promoted_from"] = {
            "owner_type": fact.owner_type,
            "owner_id": fact.owner_id,
        }
        promoted.meta["promotion_policy"] = "v1"

        logger.info(
            "Promoted fact id=%s predicate=%s to agent scope=%s",
            fact.id,
            fact.predicate,
            self.agent_id,
        )

        return promoted