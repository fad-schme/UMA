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
import math
from typing import Iterable, Optional

from uma.common.accessors import get_attr_or_key
from uma.common.ownership import validate_explicit_owner
from uma.common.types import AgentProfile, Fact, QualifierDecision, SCOPE_MODEL_VERSION
from uma.common.trust import SourceDescriptor, score_source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scope-match thresholds (memory-promotion feature)
# ---------------------------------------------------------------------------
# These are codebase constants, not YAML-tunable, until we have calibration
# data. Promotion is a policy inside the codebase for v1; user-visible
# tunability lands as a follow-up when the numbers stop being guesses.
#
# SCOPE_COSINE_THRESHOLD is deliberately generous (0.6): the embedding
# branch is the fallback when the deterministic keyword match misses, and
# a too-tight threshold would reject relevant facts phrased differently
# from the profile description. A tighter threshold (0.75+) can be set
# later without breaking the API.
SCOPE_COSINE_THRESHOLD: float = 0.6


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors.

    Returns 0.0 on shape mismatch (defensive — a mismatch means the
    embedder that produced the fact vector disagrees with the one that
    produced the profile, which is a configuration bug we should not
    silently score as "similar"). Uses the same epsilon-guard pattern
    as ``InMemoryVectorIndex._cosine`` to avoid divide-by-zero.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) + 1e-8
    nb = math.sqrt(sum(y * y for y in b)) + 1e-8
    return dot / (na * nb)


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
        self.blocked_predicates: set[str] = {
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
        except Exception as exc:
            logger.debug(
                "PromotionPolicy.is_eligible: object coercion failed: %s",
                exc,
                exc_info=True,
            )
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
        except Exception as exc:
            logger.debug(
                "PromotionPolicy.is_eligible: object stability check failed: %s",
                exc,
                exc_info=True,
            )
            return False

        return True

    # ------------------------------------------------------------------ #
    # Scope-match layer (memory-promotion feature)
    # ------------------------------------------------------------------ #

    def qualifies_for_agent_kb(
        self,
        fact: Fact,
        agent_profile: AgentProfile,
        fact_embedding: list[float],
    ) -> QualifierDecision:
        """Composite gate for agent-KB promotion.

        Combines the existing :meth:`is_eligible` content gates with a
        scope-match against the agent's declared focus. This is the
        ONLY pathway into the agent's KB — there is no
        is_eligible-only fallback for callers without a profile.

        Gate order (matters for ``decision.reasons``):
            1. Quarantine: ``fact.quarantined_at is None`` — cheap and
               short-circuits everything else so we never embed or
               log-tag a quarantined fact.
            2. Existing ``is_eligible`` — content-level gates
               (salience, confidence, predicate/subject/object rules,
               source-type allowlist, PII blocklist).
            3. Scope match — deterministic keyword match on
               ``focus_areas`` OR cosine similarity between the fact
               embedding and the profile embedding. Either passing is
               sufficient.

        This is a pure decision function — no I/O, no logging. The
        caller (``MemoryPipeline._maybe_promote_facts``) does the
        embedding lookup, the promotion write, and the ``logger.debug``
        on drops.

        Parameters
        ----------
        fact
            The candidate fact.
        agent_profile
            The bound agent profile. Callers gate on
            ``get_agent_profile`` returning non-None before invoking
            this method.
        fact_embedding
            Required. The fact's embedding, matching the profile's
            embedder. Callers gate on the embedding being present
            (facts without one cannot be promoted regardless — the
            agent KB needs the vector to search them later).
        """
        reasons: list[str] = []

        # Gate 1: quarantine
        quarantine_ok = getattr(fact, "quarantined_at", None) is None
        if not quarantine_ok:
            reasons.append("quarantined")
            return QualifierDecision(
                passed=False,
                reasons=reasons,
                quarantine_ok=False,
                is_eligible=False,
                scope_matched=False,
            )

        # Gate 2: existing content eligibility (salience, confidence, PII,
        # predicate/source blocklist, etc.)
        eligible = self.is_eligible(fact)
        if not eligible:
            reasons.append("ineligible")
            return QualifierDecision(
                passed=False,
                reasons=reasons,
                quarantine_ok=True,
                is_eligible=False,
                scope_matched=False,
            )

        # Gate 3: scope match — deterministic OR embedding branch
        scope_matched = self._scope_matches(fact, agent_profile, fact_embedding)
        if not scope_matched:
            reasons.append("scope_mismatch")
            return QualifierDecision(
                passed=False,
                reasons=reasons,
                quarantine_ok=True,
                is_eligible=True,
                scope_matched=False,
            )

        return QualifierDecision(
            passed=True,
            reasons=reasons,
            quarantine_ok=True,
            is_eligible=True,
            scope_matched=True,
        )

    @staticmethod
    def _scope_matches(
        fact: Fact,
        agent_profile: AgentProfile,
        fact_embedding: list[float],
    ) -> bool:
        """Return True if the fact is in-scope for the agent's profile.

        Two branches, OR-combined:
          (a) any ``focus_area`` appears as a case-insensitive substring
              in ``subject + predicate + object`` — subsumes both spec
              §6.2 "tag intersection" and "keyword hit" for the
              Q5=derive-on-the-fly answer.
          (b) cosine similarity between ``fact_embedding`` and
              ``agent_profile.profile_embedding`` is at least
              ``SCOPE_COSINE_THRESHOLD``.
        """
        # Deterministic branch — cheap; do first.
        try:
            fact_text = (
                f"{fact.subject} {fact.predicate} {str(fact.object)}"
            ).lower()
        except Exception as exc:
            logger.debug(
                "PromotionPolicy._scope_matches: fact_text construction failed: %s",
                exc,
                exc_info=True,
            )
            fact_text = ""
        if fact_text:
            for focus_area in agent_profile.focus_areas:
                if focus_area and focus_area.lower() in fact_text:
                    return True

        # Embedding branch
        sim = _cosine(fact_embedding, agent_profile.profile_embedding)
        return sim >= SCOPE_COSINE_THRESHOLD

    def select_promotion_target(self, fact: Fact) -> Optional[tuple[str, str, str, str | None]]:
        """
        Return the explicit promotion target for the current policy, if any.

        Current policy defaults:
        - session-local user fact -> broader user fact
        - broader user fact -> agent fact
        """
        if not self.is_eligible(fact):
            return None

        if getattr(fact, "session_id", None):
            return (
                str(getattr(fact, "tenant_id", None) or "default"),
                "user",
                str(getattr(fact, "owner_id", "") or ""),
                getattr(fact, "workspace_id", None),
            )
        if getattr(fact, "owner_type", None) == "user":
            return (
                str(getattr(fact, "tenant_id", None) or "default"),
                "agent",
                self.agent_id,
                getattr(fact, "workspace_id", None),
            )
        return None

    @staticmethod
    def _source_scope_kind(fact: Fact) -> str:
        if getattr(fact, "session_id", None):
            return "session"
        return str(getattr(fact, "owner_type", "") or "")

    @staticmethod
    def _promotion_id(
        fact: Fact,
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
    ) -> str:
        digest = hashlib.sha256(
            "|".join(
                [
                    str(getattr(fact, "id", "") or ""),
                    tenant_id,
                    owner_type,
                    owner_id,
                ]
            ).encode("utf-8")
        ).hexdigest()
        return f"fact_prom_{digest[:24]}"

    def _validate_promotion_target(
        self,
        fact: Fact,
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
    ) -> None:
        source_tenant_id = str(getattr(fact, "tenant_id", None) or "default")
        if tenant_id != source_tenant_id:
            raise ValueError("Promotion target tenant_id must match source fact tenant_id")

        if owner_type == "system":
            raise ValueError("Promotion to system scope is not supported")

        source_scope = self._source_scope_kind(fact)
        if source_scope == "session":
            if owner_type == "user":
                source_owner_id = str(getattr(fact, "owner_id", "") or "")
                if owner_id != source_owner_id:
                    raise ValueError("Session -> user promotion must preserve the source user owner_id")
                return
            if owner_type == "workspace":
                return
            raise ValueError("Session-local facts may only be promoted to user or workspace scope")

        if source_scope == "user":
            if owner_type != "agent":
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
        tenant_id: str | None = None,
        owner_type: str | None = None,
        owner_id: str | None = None,
        workspace_id: str | None = None,
        reason: str = "promotion_policy_v2",
    ) -> Fact:
        """
        Return a NEW promoted Fact with explicit target ownership.

        IMPORTANT:
        - Does NOT modify the original fact
        - Preserves provenance and lineage via meta
        """
        if owner_type is None and owner_id is None and tenant_id is None and workspace_id is None:
            target = self.select_promotion_target(fact)
            if target is None:
                raise ValueError("No explicit promotion target available for fact")
            tenant_id, owner_type, owner_id, workspace_id = target
        elif not owner_type or not owner_id:
            raise ValueError("Promotion target owner_type and owner_id are required")

        if owner_type is None or owner_id is None:
            raise ValueError("No explicit promotion target available for fact")

        if not self.is_eligible(fact):
            raise ValueError("Fact is not eligible for promotion")
        owner = validate_explicit_owner(
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            workspace_id=workspace_id,
        )
        tenant_id = str(owner["tenant_id"])
        owner_type = str(owner["owner_type"])
        owner_id = str(owner["owner_id"])
        workspace_id = owner["workspace_id"]
        self._validate_promotion_target(
            fact,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
        )

        source_session_id = getattr(fact, "session_id", None)
        source_workspace_id = getattr(fact, "workspace_id", None)
        promotion_id = self._promotion_id(
            fact,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
        )
        promoted_meta = dict(getattr(fact, "meta", None) or {})
        promoted_meta["promotion"] = {
            "source_fact_id": fact.id,
            "source_owner_type": fact.owner_type,
            "source_owner_id": fact.owner_id,
            "source_scope_kind": self._source_scope_kind(fact),
            "source_session_id": source_session_id,
            "promoted_owner_type": owner_type,
            "promoted_owner_id": owner_id,
            "tenant_id": tenant_id,
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
            owner_type=owner_type,
            owner_id=owner_id,
            tenant_id=tenant_id,
            workspace_id=(
                workspace_id
                if workspace_id is not None
                else (owner_id if owner_type == "workspace" else source_workspace_id)
            ),
            session_id=None,
            origin_agent_id=getattr(fact, "origin_agent_id", None),
            origin_user_id=getattr(fact, "origin_user_id", None),
            origin_session_id=getattr(fact, "origin_session_id", None),
            scope_model_version=SCOPE_MODEL_VERSION,
            trust_score=score_source(SourceDescriptor(kind="promotion", parent_trust_score=getattr(fact, "trust_score", None))),
            content_hash=getattr(fact, "content_hash", None),
        )

        logger.info(
            "Promoted fact source_id=%s to %s:%s",
            fact.id,
            owner_type,
            owner_id,
        )

        return promoted
