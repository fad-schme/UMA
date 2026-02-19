"""
uma.core.semantic.ingestor
===========================

SemanticIngestor — Extract -> filter -> embed -> upsert.

Safety guarantees
-----------------
- Never raises: returns [] on failures.
- Logs each failure with enough context.
"""

from __future__ import annotations

import logging
from typing import Any, List

from ...types import Fact
from ..utils.user_query_helper import build_fact_embedding_text
from .extractor import FactExtractor
from .scorer import SalienceScorer

logger = logging.getLogger(__name__)


class SemanticIngestor:
    """Production-grade semantic ingestion pipeline."""

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        semantic_store: Any,
        salience_threshold: float = 0.3,
    ) -> None:
        self.extractor = FactExtractor(llm, SalienceScorer())
        self.embedder = embedder
        self.semantic_store = semantic_store
        self.threshold = float(salience_threshold)
        logger.debug("SemanticIngestor initialized (threshold=%.2f).", self.threshold)

    async def extract(self, subject: str, text: str, *, extra_meta: dict | None = None) -> List[Fact]:
        try:
            return await self.extractor.extract_user_facts(
                subject=subject,
                text=text,
                owner_type="user",
                owner_id=subject,
                extra_meta=extra_meta,
            )
        except Exception:
            logger.exception("SemanticIngestor.extract failed.")
            return []

    async def ingest(self, subject: str, text: str, *, extra_meta: dict | None = None) -> List[Fact]:
        candidates = await self.extract(subject, text, extra_meta=extra_meta)
        if not candidates:
            return []

        selected = [f for f in candidates if float(getattr(f, "salience", 0.0) or 0.0) >= self.threshold]
        if not selected:
            logger.debug("SemanticIngestor: no facts above threshold.")
            return []

        # embed text per fact
        embed_texts = [build_fact_embedding_text(f) for f in selected]

        try:
            vectors = await self.embedder.embed(embed_texts)
        except Exception:
            logger.exception("SemanticIngestor: embedding failed.")
            return []

        if not isinstance(vectors, list) or len(vectors) != len(selected):
            logger.error(
                "SemanticIngestor: embedder returned invalid shape: expected=%d got=%r",
                len(selected),
                type(vectors),
            )
            return []

        persisted: List[Fact] = []
        for fact, vec in zip(selected, vectors):
            try:
                # Enforce owner scoping at write-time so stores can safely filter by owner_id.
                if not getattr(fact, "owner_type", None):
                    fact.owner_type = "user"
                if not getattr(fact, "owner_id", None):
                    fact.owner_id = subject
                await self.semantic_store.upsert_fact(fact, vec)
                persisted.append(fact)
            except Exception:
                logger.exception("SemanticIngestor: upsert_fact failed (fact_id=%s).", fact.id)

        logger.info("SemanticIngestor: persisted %d facts.", len(persisted))
        return persisted
