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

import hashlib
import logging
from typing import Iterable, Optional, Set

from ..utils.accessors import get_attr_or_key

from ...types import Fact, SCOPE_MODEL_VERSION, TargetOwner, make_target_owner

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
            if not (isinstance(getattr(fact, "source_ids", None), list) and fact.source_ids and fact.source_ids[0]):
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

    def select_target_owner(self, fact: Fact) -> Optional[TargetOwner]:
        """
        Return the explicit promotion target for the current policy, if any.

        Current policy defaults:
        - session-local user fact -> broader user fact
        - broader user fact -> agent fact
        """
        if not self.is_eligible(fact):
            return None

        tenant_id = str(getattr(fact, "tenant_id", None) or "default")
        workspace_id = getattr(fact, "workspace_id", None)
        if getattr(fact, "session_id", None):
            return make_target_owner(
                tenant_id=tenant_id,
                owner_type="user",
                owner_id=str(getattr(fact, "owner_id", "") or ""),
                workspace_id=workspace_id,
                allowed_owner_types={"user"},
            )
        if getattr(fact, "owner_type", None) == "user":
            return make_target_owner(
                tenant_id=tenant_id,
                owner_type="agent",
                owner_id=self.agent_id,
                workspace_id=workspace_id,
                allowed_owner_types={"agent"},
            )
        return None

    @staticmethod
    def _source_scope_kind(fact: Fact) -> str:
        if getattr(fact, "session_id", None):
            return "session"
        return str(getattr(fact, "owner_type", "") or "")

    @staticmethod
    def _promotion_id(fact: Fact, target_owner: TargetOwner) -> str:
        digest = hashlib.sha256(
            "|".join(
                [
                    str(getattr(fact, "id", "") or ""),
                    str(target_owner.tenant_id),
                    str(target_owner.owner_type),
                    str(target_owner.owner_id),
                ]
            ).encode("utf-8")
        ).hexdigest()
        return f"fact_prom_{digest[:24]}"

    def _validate_target_owner(self, fact: Fact, target_owner: TargetOwner) -> None:
        source_tenant_id = str(getattr(fact, "tenant_id", None) or "default")
        if target_owner.tenant_id != source_tenant_id:
            raise ValueError("Promotion target tenant_id must match source fact tenant_id")

        if target_owner.owner_type == "system":
            raise ValueError("Promotion to system scope is not supported")

        source_scope = self._source_scope_kind(fact)
        if source_scope == "session":
            if target_owner.owner_type == "user":
                source_owner_id = str(getattr(fact, "owner_id", "") or "")
                if target_owner.owner_id != source_owner_id:
                    raise ValueError("Session -> user promotion must preserve the source user owner_id")
                return
            if target_owner.owner_type == "workspace":
                return
            raise ValueError("Session-local facts may only be promoted to user or workspace scope")

        if source_scope == "user":
            if target_owner.owner_type != "agent":
                raise ValueError("User-scoped facts may only be promoted to agent scope")
            return

        raise ValueError(f"Unsupported promotion source scope: {source_scope!r}")

    # ------------------------------------------------------------------ #
    # Promotion
    # ------------------------------------------------------------------ #

    def promote(
        self,
        fact: Fact,
        *,
        target_owner: Optional[TargetOwner] = None,
        reason: str = "promotion_policy_v2",
    ) -> Fact:
        """
        Return a NEW promoted Fact with explicit target ownership.

        IMPORTANT:
        - Does NOT modify the original fact
        - Preserves provenance and lineage via meta
        """
        if target_owner is None:
            target_owner = self.select_target_owner(fact)
        if target_owner is None:
            raise ValueError("No explicit promotion target available for fact")

        if not self.is_eligible(fact):
            raise ValueError("Fact is not eligible for promotion")
        self._validate_target_owner(fact, target_owner)

        source_session_id = getattr(fact, "session_id", None)
        source_workspace_id = getattr(fact, "workspace_id", None)
        promotion_id = self._promotion_id(fact, target_owner)
        promoted_meta = dict(getattr(fact, "meta", None) or {})
        promoted_meta["promotion"] = {
            "source_fact_id": fact.id,
            "source_owner_type": fact.owner_type,
            "source_owner_id": fact.owner_id,
            "source_scope_kind": self._source_scope_kind(fact),
            "source_session_id": source_session_id,
            "target_owner_type": target_owner.owner_type,
            "target_owner_id": target_owner.owner_id,
            "tenant_id": target_owner.tenant_id,
            "policy": "v2",
            "reason": reason,
        }
        promoted_meta["promoted_from"] = {
            "fact_id": fact.id,
            "owner_type": fact.owner_type,
            "owner_id": fact.owner_id,
            "session_id": source_session_id,
        }
        promoted_meta["promotion_policy"] = "v2"

        promoted = Fact(
            id=promotion_id,
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            created_at=fact.created_at,
            updated_at=fact.updated_at,
            source_ids=list(fact.source_ids),
            confidence=fact.confidence,
            meta=promoted_meta,
            salience=fact.salience,
            owner_type=target_owner.owner_type,
            owner_id=target_owner.owner_id,
            tenant_id=target_owner.tenant_id,
            workspace_id=(
                target_owner.workspace_id
                if target_owner.workspace_id is not None
                else (target_owner.owner_id if target_owner.owner_type == "workspace" else source_workspace_id)
            ),
            session_id=None,
            origin_agent_id=getattr(fact, "origin_agent_id", None),
            origin_user_id=getattr(fact, "origin_user_id", None),
            origin_session_id=getattr(fact, "origin_session_id", None),
            scope_model_version=SCOPE_MODEL_VERSION,
        )

        logger.info(
            "Promoted fact source_id=%s to %s:%s",
            fact.id,
            target_owner.owner_type,
            target_owner.owner_id,
        )

        return promoted
    
    def promote_and_update_graph(
        self,
        fact: Fact,
        graph_core,
        *,
        target_owner: Optional[TargetOwner] = None,
        reason: str = "promotion_policy_v2",
    ) -> Fact:
        """
        Promote a fact and update the graph accordingly.

        This ensures DAT consistency:
        - promoted facts have corresponding graph edges in the promoted scope
        - original facts remain untouched
        """
        promoted = self.promote(fact, target_owner=target_owner, reason=reason)

        try:
            graph_core.insert_fact_triplet(
                fact_id=promoted.id,
                subject=promoted.subject,
                predicate=promoted.predicate,
                object=promoted.object,
                owner_type=promoted.owner_type,
                owner_id=promoted.owner_id,
                source_chunk_id=(promoted.source_ids[0] if promoted.source_ids else None),
                created_at=promoted.created_at,
                updated_at=promoted.updated_at,
                domain=(promoted.meta.get("domain") if isinstance(getattr(promoted, "meta", None), dict) else None),
            )
        except Exception:
            logger.exception(
                "PromotionPolicy: graph update failed for promoted fact id=%s",
                promoted.id,
            )

        return promoted
