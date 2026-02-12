"""
PromotionPolicy v1 — User → Agent Knowledge Promotion

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

from ..utils.accessors import get_attr_or_key

from ...types import Fact

logger = logging.getLogger(__name__)


class PromotionPolicy:
    """
    Rule-based promotion policy (v1).

    A fact is eligible for promotion if:
      - it is user scoped
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
        self.min_object_chars = 12
        self.max_object_chars = 800
        self.require_source_chunk = True
        self.allowed_source_types = {"pdf", "doc", "docx", "text", "wiki", "kb"}
        self.blocked_source_types = {"chat", "wm", "working_memory"}
        self.blocked_subject_prefixes = ("user:", "session:", "email:", "phone:")
        self.blocked_object_patterns = {
            "password", "ssn", "social security", "credit card", "api key", "secret",
        }

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

        # Must be user scoped
        if fact.owner_type not in ("user",):
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

        # Validate object length bounds to avoid promoting trivial or huge blobs
        try:
            obj_text = str(get_attr_or_key(fact, "object") or "").strip()
        except Exception:
            return False
        if not obj_text or len(obj_text) < self.min_object_chars or len(obj_text) > self.max_object_chars:
            return False

        # Enforce source provenance if required
        meta = get_attr_or_key(fact, "meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        src_type = str(meta.get("source_type") or "").strip().lower()
        if self.require_source_chunk:
            src_chunk = meta.get("source_chunk_id") or (fact.source_ids[0] if fact.source_ids else None)
            if not src_chunk:
                return False
        if src_type and src_type in self.blocked_source_types:
            return False
        if self.allowed_source_types and src_type and src_type not in self.allowed_source_types:
            return False

        # Block user-identifying or personal subjects from promotion
        subj = str(get_attr_or_key(fact, "subject") or "").strip().lower()
        if subj.startswith(self.blocked_subject_prefixes):
            return False

        # Block likely sensitive content
        lowered = obj_text.lower()
        if any(p in lowered for p in self.blocked_object_patterns):
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
    
    def promote_and_update_graph(self, fact: Fact, graph_core) -> Fact:
        """
        Promote a fact to agent scope AND update the graph accordingly.

        This ensures DAT consistency:
        - Agent-level facts must have corresponding agent-scoped graph edges
        - Original user facts remain untouched

        Parameters
        ----------
        fact : Fact
            Original fact (user scoped).
        graph_core : TemporalGraphCore
            Graph core used to write agent-scoped edges.
        """
        promoted = self.promote(fact)

        # Write promoted fact to graph with agent ownership
        try:
            graph_core.insert_fact_triplet(
                fact_id=promoted.id,
                subject=promoted.subject,
                predicate=promoted.predicate,
                object=promoted.object,
                owner_type=promoted.owner_type,
                owner_id=promoted.owner_id,
                source_chunk_id=promoted.meta.get("source_chunk_id"),
                created_at=promoted.created_at,
                updated_at=promoted.updated_at,
            )
        except Exception:
            logger.exception(
                "PromotionPolicy: graph update failed for promoted fact id=%s",
                promoted.id,
            )

        return promoted
