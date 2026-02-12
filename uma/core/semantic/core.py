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

        if query_text:
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
                        if subj is not None:
                            found = await self._search_text(
                                subj,
                                query_text,
                                limit=int(k),
                                owner_type=owner_type,
                                owner_id=owner_id,
                            )
                        else:
                            logger.warning(
                                "SemanticCore.search: no subject; skipping lexical fallback"
                            )
                            found = []
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
        Fetch facts by ID (authoritative payload) from the store.
        """
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None:
            return []
        if not owner_type or not owner_id:
            logger.error("SemanticCore.fetch_by_ids requires owner_type and owner_id")
            raise ValueError("SemanticCore.fetch_by_ids requires owner_type and owner_id")
        try:
            return await store.fetch_facts_by_ids(
                ids,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("SemanticCore.fetch_by_ids failed")
            raise

    async def fetch_by_predicate(
        self,
        subject: str,
        predicate: str,
        *,
        limit: int = 10,
        offset: int = 0,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        """
        Fetch facts for a subject filtered by predicate (store-backed).
        """
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "fetch_by_predicate"):
            return []
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.fetch_by_predicate: invalid subject=%r", subject)
            raise
        if not owner_type or not owner_id:
            logger.error("SemanticCore.fetch_by_predicate requires owner_type and owner_id")
            raise ValueError("SemanticCore.fetch_by_predicate requires owner_type and owner_id")
        try:
            return await store.fetch_by_predicate(
                subject=subj,
                predicate=predicate,
                limit=int(limit),
                offset=int(offset),
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("SemanticCore.fetch_by_predicate failed")
            raise

    async def upsert_fact(self, fact: Fact, embedding: List[float]) -> bool:
        """
        Persist a single fact + embedding (direct upsert path).
        """
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None:
            return False
        try:
            # Enforce owner scoping for direct upserts as well.
            if not getattr(fact, "owner_type", None):
                fact.owner_type = "user"
            if not getattr(fact, "owner_id", None):
                fact.owner_id = getattr(fact, "subject", "") or ""
            await store.upsert_fact(fact, embedding)
            return True
        except Exception:
            logger.exception("SemanticCore.upsert_fact failed for id=%s", getattr(fact, "id", None))
            return False

    def vector_index(self) -> Any:
        """
        Expose the backing vector index (if present) for diagnostics.
        """
        store = getattr(self.ingestor, "semantic_store", None)
        return getattr(store, "vector_index", None) if store is not None else None

    async def list_facts_for_subject(
        self,
        subject: str,
        *,
        limit: Optional[int] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        """
        List all facts for a subject with optional owner scoping.
        """
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "list_facts_for_subject"):
            return []
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.list_facts_for_subject: invalid subject=%r", subject)
            raise
        if not owner_type or not owner_id:
            logger.error("SemanticCore.list_facts_for_subject requires owner_type and owner_id")
            raise ValueError("SemanticCore.list_facts_for_subject requires owner_type and owner_id")
        try:
            return await store.list_facts_for_subject(
                subject=subj,
                limit=limit,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("SemanticCore.list_facts_for_subject failed")
            raise

    async def delete_fact(
        self,
        fact_id: str,
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a fact by ID from the store.
        """
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "delete_fact"):
            return False
        if not owner_type or not owner_id:
            logger.error("SemanticCore.delete_fact requires owner_type and owner_id")
            raise ValueError("SemanticCore.delete_fact requires owner_type and owner_id")
        try:
            await store.delete_fact(fact_id, owner_type=owner_type, owner_id=owner_id)
            return True
        except Exception:
            logger.exception("SemanticCore.delete_fact failed for id=%s", fact_id)
            raise

    def _dedup_facts(self, facts: List[Fact]) -> List[Fact]:
        """
        Deduplicate facts within a single ingestion batch.

        Dedup key (v1 DAT invariant):
            (owner_type, owner_id, subject, predicate, object)

        This prevents:
        - duplicate SQL inserts
        - duplicate vector entries
        - duplicate graph edges

        Global dedup (across batches) remains the responsibility
        of the semantic store.
        """
        seen = set()
        deduped: List[Fact] = []

        for fact in facts:
            try:
                key = (
                    getattr(fact, "owner_type", None),
                    getattr(fact, "owner_id", None),
                    getattr(fact, "subject", None),
                    getattr(fact, "predicate", None),
                    getattr(fact, "object", None),
                )
            except Exception:
                # If something is malformed, let downstream validation handle it
                deduped.append(fact)
                continue

            if key in seen:
                continue

            seen.add(key)
            deduped.append(fact)

        return deduped


def _fact_topics(fact: Any) -> List[str]:
    meta = getattr(fact, "meta", {}) or {}
    if not isinstance(meta, dict):
        return []
    topics = meta.get("topics")
    if isinstance(topics, list):
        return [str(t) for t in topics if t]
    topic = meta.get("topic")
    if topic:
        return [str(topic)]
    return []
