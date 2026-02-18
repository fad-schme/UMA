"""
semantic/core.py
================

SemanticCore — The unified interface for UMA semantic memory.

Exposes:
    - extract(subject_or_user_id, text)
    - ingest(subject_or_user_id, text)

Subject convention (v1)
-----------------------
UMA-RLM standardizes semantic subjects as:
    "user:<id>"

Callers MAY pass either:
- raw user_id: "123"
- canonical subject: "user:123"

SemanticCore will normalize automatically.

Coding Agent Instructions
-------------------------
- Keep this interface simple and stable.
- This is the ONLY class MemoryPipeline should call for semantic ingestion.
Note for maintainers:
- Semantic retrieval should go through `search`.
- Lexical search is internal-only via `_search_text`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...types import Fact
from ..utils.identity import ensure_user_subject
from ..utils.dedupe import dedupe_by_id
from ..utils.user_query_helper import extract_keywords_and_phrases, build_fact_embedding_text
from .ingestor import SemanticIngestor

logger = logging.getLogger(__name__)


class SemanticCore:
    """
    High-level interface for UMA semantic memory.

    Wraps:
        - FactExtractor
        - SalienceScorer
        - SemanticIngestor
        - SemanticSQLStore

    This class enforces subject normalization but delegates all business logic
    to SemanticIngestor.
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

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def upsert_fact(self, fact: Fact, embedding: List[float]) -> None:
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "upsert_fact"):
            logger.error("SemanticCore.upsert_fact: semantic_store missing")
            raise RuntimeError("SemanticCore.upsert_fact: semantic_store missing")
        try:
            if not getattr(fact, "owner_id", None):
                fact.owner_type = getattr(fact, "owner_type", None) or "user"
                fact.owner_id = getattr(fact, "subject", "") or ""
            await store.upsert_fact(fact, embedding)
        except Exception:
            logger.exception("SemanticCore.upsert_fact failed")
            raise

    def vector_index(self):
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None:
            raise RuntimeError("SemanticCore.vector_index: semantic_store missing")
        return getattr(store, "vector_index", None)

    async def list_facts_for_subject(
        self,
        subject: str,
        *,
        owner_type: str,
        owner_id: str,
        limit: Optional[int] = None,
    ) -> List[Fact]:
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "list_facts_for_subject"):
            logger.error("SemanticCore.list_facts_for_subject: store missing")
            raise RuntimeError("SemanticCore.list_facts_for_subject: store missing")
        if not owner_type or not owner_id:
            logger.error("SemanticCore.list_facts_for_subject requires owner_type and owner_id")
            raise ValueError("SemanticCore.list_facts_for_subject requires owner_type and owner_id")
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.list_facts_for_subject: invalid subject=%r", subject)
            raise
        try:
            return await store.list_facts_for_subject(
                subj,
                limit=limit,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("SemanticCore.list_facts_for_subject failed")
            raise

    async def extract(self, subject: str, text: str, *, extra_meta: dict | None = None) -> List[Fact]:
        """
        Extract semantic facts (not persisted).

        Parameters
        ----------
        subject : str
            Raw user_id or canonical subject ("user:<id>").
        text : str
            Text to extract facts from (typically assistant reply).

        Returns
        -------
        List[Fact]
            Extracted facts (not persisted).
        """
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.extract: invalid subject=%r", subject)
            raise

        return await self.ingestor.extract(subj, text, extra_meta=extra_meta)

    async def ingest(self, subject: str, text: str, *, extra_meta: dict | None = None) -> List[Fact]:
        """
        Extract + ingest facts into the semantic store.

        Parameters
        ----------
        subject : str
            Raw user_id or canonical subject ("user:<id>").
        text : str
            Text to ingest facts from.

        Returns
        -------
        List[Fact]
            Persisted facts.
        """
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.ingest: invalid subject=%r", subject)
            raise

        facts = await self.ingestor.ingest(subj, text, extra_meta=extra_meta)

        # Enforce ownership scoping for persisted facts.
        # Facts extracted from user-scoped turns should be user-owned under the canonical subject.
        for f in facts or []:
            try:
                f.owner_type = "user"
                f.owner_id = subj
            except Exception:
                logger.exception("SemanticCore.ingest: failed to set owner on fact id=%s", getattr(f, "id", None))
                raise

        # EXPLICIT DEDUP (DAT v1 requirement)
        facts = self._dedup_facts(facts)

        return facts

    # ------------------------------------------------------------------
    # RETRIEVAL API
    # ------------------------------------------------------------------

    async def search(
        self,
        subject: Optional[str],
        query_embedding: List[float],
        *,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        allowed_topics: Optional[List[str]] = None,
    ) -> List[Fact]:
        """
        Unified semantic retrieval entry point.
        Performs vector search, optional topic/predicate filtering, and lexical fallback.

        IMPORTANT (KB lane):
        - When `subject` is None (common for agent-owned KB facts), this method MUST NOT
          hard-filter results by query_text keywords. Otherwise we can drop to 0 results
          even when vector search returned good candidates (the “Qdrant IDs → 0 facts” bug).
        """
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None:
            return []
        subj: Optional[str] = None
        if subject:
            try:
                subj = ensure_user_subject(subject)
            except Exception:
                logger.exception("SemanticCore.search: invalid subject=%r", subject)
                raise

        requested_topic = filters.get("topic") if isinstance(filters, dict) else None
        requested_predicate = filters.get("predicate") if isinstance(filters, dict) else None
        if requested_predicate:
            requested_predicate = str(requested_predicate).upper()

        facts: List[Any] = []
        try:
            try:
                logger.debug(
                    "SemanticCore.search: path=vector owner=%s:%s subject=%s",
                    owner_type,
                    owner_id,
                    subj,
                )
                search_kwargs = dict(
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                    offset=int(offset),
                )
                if subj is not None:
                    search_kwargs["subject"] = subj
                found = await store.search(**search_kwargs)
            except TypeError:
                logger.debug(
                    "SemanticCore.search: path=vector_legacy owner=%s:%s subject=%s",
                    owner_type,
                    owner_id,
                    subj,
                )
                search_kwargs = dict(
                    query_embedding=query_embedding,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                )
                if subj is not None:
                    search_kwargs["subject"] = subj
                found = await store.search(**search_kwargs)
            if found:
                facts.extend(found)
        except Exception:
            logger.exception(
                "SemanticCore.search failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            raise

        # Optional topic filtering (soft)
        if requested_topic:
            filtered = [f for f in facts if requested_topic in _fact_topics(f)]
            if filtered:
                facts = filtered

        if allowed_topics:
            filtered = [
                f for f in facts
                if any(t in allowed_topics for t in _fact_topics(f))
            ]
            if filtered:
                facts = filtered

        if requested_predicate:
            facts = [
                f for f in facts
                if getattr(f, "predicate", "").upper() == requested_predicate
            ]

        # IMPORTANT: only apply lexical hard-filtering when we have a user subject.
        # For KB facts (subject=None), keep vector results (no hard gating).
        if query_text and subj is not None:
            extracted = extract_keywords_and_phrases(query_text)
            terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
            terms = [t for t in terms if isinstance(t, str) and t]
            if terms:
                lowered_terms = [t.lower() for t in terms]
                original_count = len(facts)
                filtered = []
                for fact in facts:
                    text = build_fact_embedding_text(fact).lower()
                    if any(t in text for t in lowered_terms):
                        filtered.append(fact)
                if filtered:
                    facts = filtered
                    logger.debug(
                        "SemanticCore.search: lexical filter kept %d/%d",
                        len(facts),
                        original_count,
                    )
                else:
                    fallback: List[Any] = []
                    try:
                        logger.warning(
                            "SemanticCore.search: lexical fallback used (no vector matches after filter) owner=%s:%s",
                            owner_type,
                            owner_id,
                        )
                        found = await self._search_text(
                            subj,
                            query_text,
                            limit=int(k),
                            owner_type=owner_type,
                            owner_id=owner_id,
                        )
                        if found:
                            fallback.extend(found)
                    except Exception:
                        logger.exception(
                            "SemanticCore.search: lexical fallback owner=%s:%s failed",
                            owner_type,
                            owner_id,
                        )
                        raise
                    if fallback:
                        facts = fallback
                        logger.debug(
                            "SemanticCore.search: lexical fallback returned %d",
                            len(facts),
                        )

        return dedupe_by_id(facts)

    async def fetch_more_facts(
        self,
        subject: Optional[str],
        predicate: str,
        *,
        owner_type: str,
        owner_id: str,
        k: int,
        offset: int = 0,
    ) -> List[Fact]:
        """
        Fetch additional facts for a subject/predicate using deterministic paging.
        """
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None:
            return []
        subj: Optional[str] = None
        if subject:
            try:
                subj = ensure_user_subject(subject)
            except Exception:
                logger.exception("SemanticCore.fetch_more_facts: invalid subject=%r", subject)
                raise
        if not owner_type or not owner_id:
            logger.error("SemanticCore.fetch_more_facts requires owner_type and owner_id")
            raise ValueError("SemanticCore.fetch_more_facts requires owner_type and owner_id")

        predicate_u = (predicate or "").upper()
        try:
            # Deterministic paging MUST be SQL-backed with stable ORDER BY before applying offset.
            if hasattr(store, "list_facts_for_subject") and subj is not None:
                facts = await store.list_facts_for_subject(
                    subj,
                    limit=None,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            elif hasattr(store, "list_facts_for_owner"):
                facts = await store.list_facts_for_owner(
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            else:
                facts = []
        except Exception:
            logger.exception(
                "SemanticCore.fetch_more_facts: list_facts_for_subject failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            raise

        filtered = []
        for f in facts or []:
            pred_val = getattr(f, "predicate", None) if hasattr(f, "predicate") else (
                f.get("predicate") if isinstance(f, dict) else None
            )
            if pred_val and str(pred_val).upper() == predicate_u:
                filtered.append(f)
        # Stable ordering comes from the store (updated_at desc). Apply offset/limit after filtering.
        offset_i = max(0, int(offset))
        k_i = max(0, int(k))
        windowed = filtered[offset_i : offset_i + k_i] if k_i else filtered[offset_i:]
        return dedupe_by_id(windowed)

    async def _search_text(
        self,
        subject: str,
        query_text: str,
        *,
        limit: int = 5,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        """
        Internal lexical-only fact search (store-level LIKE query).
        Used by `search` for fallback only.
        """
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "search_text"):
            return []
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.search_text: invalid subject=%r", subject)
            raise
        if not owner_type or not owner_id:
            logger.error("SemanticCore.search_text requires owner_type and owner_id")
            raise ValueError("SemanticCore.search_text requires owner_type and owner_id")
        try:
            return await store.search_text(
                query_text,
                subject=subj,
                limit=int(limit),
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("SemanticCore.search_text failed")
            raise

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        """
        Fetch facts by IDs (authoritative payload).
        """
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "fetch_by_ids"):
            return []
        if not owner_type or not owner_id:
            logger.error("SemanticCore.fetch_by_ids requires owner_type and owner_id")
            raise ValueError("SemanticCore.fetch_by_ids requires owner_type and owner_id")
        try:
            facts = await store.fetch_by_ids(ids=ids, owner_type=owner_type, owner_id=owner_id)
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
