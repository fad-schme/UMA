"""
semantic/ingestor.py
====================

SemanticIngestor — production-grade ingestion engine.

Responsibilities:
    1. Extract facts via FactExtractor
    2. Filter by salience
    3. Embed remaining facts
    4. Upsert into SemanticSQLStore

Coding Agent Instructions
-------------------------
- NEVER let ingestion crash UMA-3; catch & log errors.
- Keep embedding batch size adjustable in the future.
"""

from __future__ import annotations

import logging
from typing import Any, List

from ...types_fact import Fact
from .extractor import FactExtractor
from .scorer import SalienceScorer

logger = logging.getLogger(__name__)


class SemanticIngestor:
    """
    High-level semantic ingestion engine.
    """

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        semantic_store: Any,
        salience_threshold: float = 0.3,
    ):
        self.extractor = FactExtractor(llm, SalienceScorer())
        self.semantic_store = semantic_store
        self.embedder = embedder
        self.threshold = salience_threshold

        logger.info(
            "SemanticIngestor initialized (threshold=%.2f)", salience_threshold
        )

    # ------------------------------------------------------------------
    # EXTRACT ONLY
    # ------------------------------------------------------------------

    async def extract(self, subject: str, text: str) -> List[Fact]:
        return await self.extractor.extract_from_text(subject, text)

    # ------------------------------------------------------------------
    # EXTRACT + INGEST
    # ------------------------------------------------------------------

    async def ingest(self, subject: str, text: str) -> List[Fact]:
        """
        Extract, score, embed, upsert.

        Returns
        -------
        List[Fact]
            Facts that were actually upserted.
        """
        candidates = await self.extract(subject, text)
        if not candidates:
            return []

        selected = [f for f in candidates if f.meta.get("salience", 0.0) >= self.threshold]

        if not selected:
            logger.info("SemanticIngestor: no facts above threshold.")
            return []

        # Build embedding text
        texts = [f"{f.subject} {f.predicate} {f.object}" for f in selected]

        try:
            vectors = await self.embedder.embed(texts)
        except Exception:
            logger.exception("SemanticIngestor: embedding failed.")
            return []

        # Upsert into semantic store
        for fact, emb in zip(selected, vectors):
            try:
                await self.semantic_store.upsert_fact(fact, emb)
            except Exception:
                logger.exception("SemanticIngestor: failed to upsert fact id=%s", fact.id)

        logger.info("SemanticIngestor: persisted %d facts.", len(selected))
        return selected