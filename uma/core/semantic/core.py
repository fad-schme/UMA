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
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...types_fact import Fact
from ..utils.identity import ensure_user_subject
from ..utils.dedupe import dedupe_by_id
from ..utils.user_query_helper import extract_query_terms, expand_query_terms, build_fact_embedding_text
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
        logger.info("SemanticCore initialized.")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def extract(self, subject: str, text: str) -> List[Fact]:
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
            return []

        return await self.ingestor.extract(subj, text)

    async def ingest(self, subject: str, text: str) -> List[Fact]:
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
            return []

        facts = await self.ingestor.ingest(subj, text)

        # EXPLICIT DEDUP (DAT v1 requirement)
        facts = self._dedup_facts(facts)

        return facts

    # ------------------------------------------------------------------
    # RETRIEVAL API
    # ------------------------------------------------------------------

    async def search(
        self,
        subject: str,
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
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None:
            return []
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.search: invalid subject=%r", subject)
            return []

        requested_topic = filters.get("topic") if isinstance(filters, dict) else None
        requested_predicate = filters.get("predicate") if isinstance(filters, dict) else None
        if requested_predicate:
            requested_predicate = str(requested_predicate).upper()

        facts: List[Any] = []
        try:
            try:
                found = await store.search(
                    query_embedding=query_embedding,
                    subject=subj,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                    offset=int(offset),
                )
            except TypeError:
                found = await store.search(
                    query_embedding=query_embedding,
                    subject=subj,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                )
            if found:
                facts.extend(found)
        except Exception:
            logger.exception(
                "SemanticCore.search failed owner=%s:%s",
                owner_type,
                owner_id,
            )

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
            terms = expand_query_terms(query_text) or extract_query_terms(query_text)
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
                    try:
                        fallback: List[Any] = []
                        try:
                            found = await store.search_text(
                                query_text,
                                subject=subj,
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
                        if fallback:
                            facts = fallback
                            logger.debug(
                                "SemanticCore.search: lexical fallback returned %d",
                                len(facts),
                            )
                    except Exception:
                        logger.exception("SemanticCore.search: lexical fallback failed")

        return dedupe_by_id(facts)

    async def fetch_more_facts(
        self,
        subject: str,
        predicate: str,
        *,
        owner_type: str,
        owner_id: str,
        k: int,
        offset: int = 0,
    ) -> List[Fact]:
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None:
            return []
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.fetch_more_facts: invalid subject=%r", subject)
            return []

        predicate_u = (predicate or "").upper()
        facts: List[Any] = []
        try:
            # First, try store.search with offset (works for stores that support offset)
            try:
                found = await store.search(
                    query_embedding=[],
                    subject=subj,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=int(k),
                    offset=int(offset),
                )
            except Exception:
                found = []

            if not found and hasattr(store, "fetch_by_predicate"):
                try:
                    found = await store.fetch_by_predicate(
                        subject=subj,
                        predicate=predicate,
                        limit=int(k),
                        offset=int(offset),
                        owner_type=owner_type,
                        owner_id=owner_id,
                    )
                except Exception:
                    logger.exception(
                        "SemanticCore.fetch_more_facts: predicate fetch failed owner=%s:%s",
                        owner_type,
                        owner_id,
                    )
                    found = []

            if found:
                facts.extend(found)
        except Exception:
            logger.exception(
                "SemanticCore.fetch_more_facts failed owner=%s:%s",
                owner_type,
                owner_id,
            )

        filtered = []
        for f in facts or []:
            pred_val = getattr(f, "predicate", None) if hasattr(f, "predicate") else (
                f.get("predicate") if isinstance(f, dict) else None
            )
            if pred_val and str(pred_val).upper() == predicate_u:
                filtered.append(f)
        return dedupe_by_id(filtered)

    async def search_text(
        self,
        subject: str,
        query_text: str,
        *,
        limit: int = 5,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "search_text"):
            return []
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.search_text: invalid subject=%r", subject)
            return []
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
            return []

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None:
            return []
        try:
            return await store.fetch_facts_by_ids(
                ids,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("SemanticCore.fetch_by_ids failed")
            return []

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
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "fetch_by_predicate"):
            return []
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.fetch_by_predicate: invalid subject=%r", subject)
            return []
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
            return []

    async def upsert_fact(self, fact: Fact, embedding: List[float]) -> bool:
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None:
            return False
        try:
            await store.upsert_fact(fact, embedding)
            return True
        except Exception:
            logger.exception("SemanticCore.upsert_fact failed for id=%s", getattr(fact, "id", None))
            return False

    def vector_index(self) -> Any:
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
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "list_facts_for_subject"):
            return []
        try:
            subj = ensure_user_subject(subject)
        except Exception:
            logger.exception("SemanticCore.list_facts_for_subject: invalid subject=%r", subject)
            return []
        try:
            return await store.list_facts_for_subject(
                subject=subj,
                limit=limit,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("SemanticCore.list_facts_for_subject failed")
            return []

    async def delete_fact(
        self,
        fact_id: str,
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> bool:
        store = getattr(self.ingestor, "semantic_store", None)
        if store is None or not hasattr(store, "delete_fact"):
            return False
        try:
            await store.delete_fact(fact_id, owner_type=owner_type, owner_id=owner_id)
            return True
        except Exception:
            logger.exception("SemanticCore.delete_fact failed for id=%s", fact_id)
            return False

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
