"""
semantic/core.py
================

SemanticCore — The unified interface for UMA semantic memory.

Ownership model (vNext)
-----------------------
Retrieval MUST be ownership-scoped ONLY via (owner_type, owner_id).

- owner_type: "user" | "agent"
- owner_id: scope identifier (for user scope: canonical "user:<id>" string)

Fact.subject is treated as OPTIONAL metadata and MUST NOT gate retrieval.

Ingestion/extract APIs still accept a `subject`/user_id string for backward compatibility,
but persisted facts MUST have explicit ownership (owner_type/owner_id).

Coding Agent Instructions
-------------------------
- Keep this interface simple and stable.
- Retrieval must NOT require or accept "subject" gating.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...types import Fact
from ..utils.identity import normalize_user_id
from ..utils.dedupe import dedupe_by_id
from ..utils.user_query_helper import extract_keywords_and_phrases, build_fact_embedding_text
from .ingestor import SemanticIngestor

logger = logging.getLogger(__name__)


class SemanticCore:
    """
    High-level interface for UMA semantic memory.

    Wraps:
        - SemanticIngestor
        - semantic_store (SQL + vector)

    IMPORTANT:
    - Retrieval is ownership-only.
    - No subject-driven retrieval is allowed.
    """

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        semantic_store: Any,
        salience_threshold: float = 0.3,
    ) -> None:
        self.ingestor = SemanticIngestor(
            llm=llm,
            embedder=embedder,
            semantic_store=semantic_store,
            salience_threshold=salience_threshold,
        )
        logger.debug("SemanticCore initialized.")

        self.store = getattr(self.ingestor, "semantic_store", None)
        if self.store is None:
            logger.error("SemanticCore: store missing or unsupported")
            raise RuntimeError("SemanticCore: store missing or unsupported")
        

    # ------------------------------------------------------------------
    # PUBLIC API (WRITE / INGEST)
    # ------------------------------------------------------------------

    async def upsert_fact(self, fact: Fact, embedding: List[float]) -> None:
        """
        Persist a fact + embedding.

        Ownership consistency:
        - owner_type and owner_id MUST be present.
        - We do not infer owner_id from fact.subject.
        """
        if not hasattr(self.store, "upsert_fact"):
            logger.error("SemanticCore.upsert_fact: semantic_store missing")
            raise RuntimeError("SemanticCore.upsert_fact: semantic_store missing")

        fact_id = getattr(fact, "id", None)
        if not fact_id or not isinstance(fact_id, str) or not fact_id.startswith("fact_"):
            logger.error("SemanticCore.upsert_fact: invalid fact id=%r (must start with 'fact_')", fact_id)
            raise ValueError("SemanticCore.upsert_fact: fact.id must start with 'fact_'")

        owner_type = getattr(fact, "owner_type", None)
        owner_id = getattr(fact, "owner_id", None)

        if not owner_type or not owner_id:
            logger.error(
                "SemanticCore.upsert_fact: missing ownership on fact id=%r owner_type=%r owner_id=%r",
                getattr(fact, "id", None),
                owner_type,
                owner_id,
            )
            raise ValueError("SemanticCore.upsert_fact: fact must include owner_type and owner_id")

        try:
            await self.store.upsert_fact(fact, embedding)
        except Exception:
            logger.exception("SemanticCore.upsert_fact failed")
            raise

    def vector_index(self):
        return getattr(self.store, "vector_index", None)

    async def extract(
        self,
        user_id: str,
        text: str,
        *,
        extra_meta: dict | None = None,
    ) -> List[Fact]:
        """
        Extract semantic facts (not persisted).
        Kept for back-compat: accepts raw user_id or canonical "user:<id>".
        """
        try:
            normalized_user_id = normalize_user_id(user_id)
        except Exception:
            logger.exception("SemanticCore.extract: invalid user_id=%r", user_id)
            raise

        return await self.ingestor.extract(normalized_user_id, text, extra_meta=extra_meta)

    async def ingest(
        self,
        user_id: str,
        text: str,
        *,
        extra_meta: dict | None = None,
    ) -> List[Fact]:
        """
        Extract + ingest facts into the semantic self.store.

        Ownership consistency:
        - Facts extracted from user-scoped turns MUST be persisted as:
          owner_type="user", owner_id="user:<id>"
        """
        try:
            normalized_user_id = normalize_user_id(user_id)
        except Exception:
            logger.exception("SemanticCore.ingest: invalid user_id=%r", user_id)
            raise

        facts = await self.ingestor.ingest(normalized_user_id, text, extra_meta=extra_meta)

        # Explicitly set ownership (never infer from "subject" fields)
        for f in facts or []:
            try:
                f.owner_type = "user"
                f.owner_id = normalized_user_id
            except Exception:
                logger.exception("SemanticCore.ingest: failed to set owner on fact id=%s", getattr(f, "id", None))
                raise

        return self._dedup_facts(facts)

    # ------------------------------------------------------------------
    # RETRIEVAL API (OWNERSHIP-ONLY)
    # ------------------------------------------------------------------

    async def list_facts_for_owner(
        self,
        *,
        owner_type: str,
        owner_id: str,
        limit: Optional[int] = None,
    ) -> List[Fact]:
        """
        List facts for an owner scope (ownership-only).
        """
       
        if not hasattr(self.store, "list_facts_for_owner"):
            logger.error("SemanticCore.list_facts_for_owner: store missing or unsupported")
            raise RuntimeError("SemanticCore.list_facts_for_owner: store missing or unsupported")
        if not owner_type or not owner_id:
            logger.error("SemanticCore.list_facts_for_owner requires owner_type and owner_id")
            raise ValueError("SemanticCore.list_facts_for_owner requires owner_type and owner_id")
        try:
            return await self.store.list_facts_for_owner(owner_type=owner_type, owner_id=owner_id, limit=limit)
        except Exception:
            logger.exception("SemanticCore.list_facts_for_owner failed")
            raise

    async def delete_fact(
        self,
        fact_id: str,
        *,
        owner_type: str,
        owner_id: str,
    ) -> None:
        """
        Delete a fact by id, scoped by ownership.

        Notes
        -----
        - This is a maintenance/pruning API, not a retrieval filter.
        - Store must enforce ownership in the delete path.
        """
        
        if not fact_id or not isinstance(fact_id, str):
            logger.error("SemanticCore.delete_fact requires fact_id as a non-empty string")
            raise ValueError("SemanticCore.delete_fact requires fact_id as a non-empty string")
        if not owner_type or not owner_id:
            logger.error("SemanticCore.delete_fact requires owner_type and owner_id")
            raise ValueError("SemanticCore.delete_fact requires owner_type and owner_id")
        try:
            await self.store.delete_fact(fact_id, owner_type=owner_type, owner_id=owner_id)
        except Exception:
            logger.exception("SemanticCore.delete_fact failed")
            raise

    async def search(
        self,
        query_embedding: List[float],
        *,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> List[Fact]:
        """
        Unified semantic retrieval entry point (ownership-only).

        - Vector search is primary.
        - Optional topic/predicate filtering is applied after retrieval.
        - Lexical fallback is allowed but must also be ownership-only.
        """
        if not owner_type or not owner_id:
            logger.error("SemanticCore.search requires owner_type and owner_id")
            raise ValueError("SemanticCore.search requires owner_type and owner_id")

        requested_topic = filters.get("topic") if isinstance(filters, dict) else None
        requested_predicate = filters.get("predicate") if isinstance(filters, dict) else None
        if requested_predicate:
            requested_predicate = str(requested_predicate).upper()

        facts: List[Any] = []
        try:
            logger.debug("SemanticCore.search: path=vector owner=%s:%s", owner_type, owner_id)
            found = await self.store.search(
                query_embedding=query_embedding,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
                offset=int(offset),
            )
            if found:
                facts.extend(found)
        except Exception:
            logger.exception("SemanticCore.search failed owner=%s:%s", owner_type, owner_id)
            raise

        # Optional topic filtering (soft)
        if requested_topic:
            filtered = [f for f in facts if requested_topic in _fact_topics(f)]
            if filtered:
                facts = filtered

        if requested_predicate:
            facts = [f for f in facts if getattr(f, "predicate", "").upper() == requested_predicate]

        # Soft lexical re-rank / filter (ownership-only). This should not hard-gate to zero if vector found good facts.
        if query_text and isinstance(query_text, str) and query_text.strip():
            extracted = extract_keywords_and_phrases(query_text)
            terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
            terms = [t for t in terms if isinstance(t, str) and t]
            if terms and facts:
                lowered_terms = [t.lower() for t in terms]
                original_count = len(facts)
                filtered = []
                for fact in facts:
                    text = build_fact_embedding_text(fact).lower()
                    if any(t in text for t in lowered_terms):
                        filtered.append(fact)
                # Keep if we still have signal; otherwise preserve vector results.
                if filtered:
                    facts = filtered
                    logger.debug("SemanticCore.search: lexical filter kept %d/%d", len(facts), original_count)

        # Lexical fallback ONLY if vector yielded nothing.
        if (not facts) and query_text and hasattr(self.store, "search_text"):
            try:
                logger.debug("SemanticCore.search: lexical fallback owner=%s:%s", owner_type, owner_id)
                # Ownership-only lexical search (no subject param)
                found = await self.store.search_text(
                    query_text,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    limit=int(k),
                )
                if found:
                    facts = list(found)
            except Exception:
                logger.exception("SemanticCore.search: lexical fallback failed owner=%s:%s", owner_type, owner_id)
                raise

        return dedupe_by_id(facts)

    async def fetch_more_facts(
        self,
        predicate: str,
        *,
        owner_type: str,
        owner_id: str,
        k: int,
        offset: int = 0,
    ) -> List[Fact]:
        """
        Fetch additional facts for an owner scope, filtered by predicate using deterministic paging.

        Ownership-only:
        - No subject parameter allowed.
        - Store must provide stable ordering for list_facts_for_owner (e.g., updated_at desc).
        """
        if not owner_type or not owner_id:
            logger.error("SemanticCore.fetch_more_facts requires owner_type and owner_id")
            raise ValueError("SemanticCore.fetch_more_facts requires owner_type and owner_id")

        predicate_u = (predicate or "").upper()
        try:
            if hasattr(self.store, "list_facts_for_owner"):
                facts = await self.store.list_facts_for_owner(owner_type=owner_type, owner_id=owner_id, limit=None)
            else:
                facts = []
        except Exception:
            logger.exception(
                "SemanticCore.fetch_more_facts: list_facts_for_owner failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            raise

        filtered: List[Any] = []
        for f in facts or []:
            pred_val = getattr(f, "predicate", None) if hasattr(f, "predicate") else (
                f.get("predicate") if isinstance(f, dict) else None
            )
            if pred_val and str(pred_val).upper() == predicate_u:
                filtered.append(f)

        offset_i = max(0, int(offset))
        k_i = max(0, int(k))
        windowed = filtered[offset_i : offset_i + k_i] if k_i else filtered[offset_i:]
        return dedupe_by_id(windowed)

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        owner_type: str,
        owner_id: str,
    ) -> List[Fact]:
        """
        Fetch facts by IDs (authoritative payload) with ownership enforcement.
        """
    
        if not hasattr(self.store, "fetch_by_ids"):
            return []
        if not owner_type or not owner_id:
            logger.error("SemanticCore.fetch_by_ids requires owner_type and owner_id")
            raise ValueError("SemanticCore.fetch_by_ids requires owner_type and owner_id")
        try:
            facts = await self.store.fetch_by_ids(ids=ids, owner_type=owner_type, owner_id=owner_id)
            missing = max(0, len(ids or []) - len(facts or []))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "SemanticCore.fetch_by_ids: ids=%d returned=%d owner=%s:%s",
                    len(ids or []),
                    len(facts or []),
                    owner_type,
                    owner_id,
                )
            if missing:
                logger.warning(
                    "SemanticCore.fetch_by_ids: missing=%d owner=%s:%s",
                    missing,
                    owner_type,
                    owner_id,
                )
            # Defensive filter in case upstream store misbehaves.
            filtered = [
                f for f in (facts or [])
                if getattr(f, "owner_type", None) == owner_type
                and getattr(f, "owner_id", None) == owner_id
            ]
            if filtered and len(filtered) != len(facts or []):
                logger.warning(
                    "SemanticCore.fetch_by_ids: dropped %d cross-scope facts owner=%s:%s",
                    len(facts or []) - len(filtered),
                    owner_type,
                    owner_id,
                )
            return filtered
        except Exception:
            logger.exception("SemanticCore.fetch_by_ids failed")
            raise

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _dedup_facts(self, facts: List[Fact]) -> List[Fact]:
        return dedupe_by_id(facts or [])


def _fact_topics(f: Any) -> List[str]:
    meta = getattr(f, "meta", None) or {}
    if isinstance(f, dict):
        meta = f.get("meta") or {}
    if not isinstance(meta, dict):
        return []
    topics = meta.get("topics") or []
    if isinstance(topics, list):
        return [str(t) for t in topics if t]
    if isinstance(topics, str):
        return [topics]
    return []
