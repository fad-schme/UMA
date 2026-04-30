"""
uma.memory.semantic.ingestor
===========================

SemanticIngestor — Extract -> filter -> embed -> upsert.

Safety guarantees
-----------------
- Never raises: returns [] on failures.
- Logs each failure with enough context.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from uma.common.types import Fact
from uma.retrieve.user_query_helper import build_fact_embedding_text
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
        # Allow SemanticCore to be constructed in retrieval-only mode (tests / minimal envs)
        # without requiring an LLM. Extraction/ingestion APIs will safely no-op.
        self.extractor: FactExtractor | None = None
        try:
            if llm is not None:
                self.extractor = FactExtractor(llm, SalienceScorer())
        except Exception:
            logger.exception("SemanticIngestor: failed to initialize FactExtractor; extraction disabled.")
            self.extractor = None
        self.embedder = embedder
        self.semantic_store = semantic_store
        self.threshold = float(salience_threshold)
        logger.debug("SemanticIngestor initialized (threshold=%.2f).", self.threshold)

    async def extract(self, user_id: str, text: str, *, extra_meta: dict | None = None) -> List[Fact]:
        if self.extractor is None:
            logger.debug("SemanticIngestor.extract: extractor unavailable; returning [].")
            return []
        try:
            return await self.extractor.extract_user_facts(
                subject="user",
                text=text,
                owner_type="user",
                owner_id=user_id,
                extra_meta=extra_meta,
            )
        except Exception:
            logger.exception("SemanticIngestor.extract failed.")
            return []

    async def ingest(
        self,
        user_id: str,
        text: str,
        *,
        extra_meta: dict | None = None,
        fact_transform: Optional[Callable[[Fact], None]] = None,
    ) -> List[Fact]:
        if self.embedder is None:
            logger.debug("SemanticIngestor.ingest: embedder unavailable; returning [].")
            return []
        if self.semantic_store is None or not hasattr(self.semantic_store, "upsert_fact"):
            logger.debug("SemanticIngestor.ingest: semantic_store unavailable; returning [].")
            return []

        candidates = await self.extract(user_id, text, extra_meta=extra_meta)
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
                if fact_transform is not None:
                    fact_transform(fact)
                # Enforce owner scoping at write-time so stores can safely filter by owner_id.
                if not getattr(fact, "owner_type", None):
                    fact.owner_type = "user"
                if not getattr(fact, "owner_id", None):
                    fact.owner_id = user_id
                await self.semantic_store.upsert_fact(fact, vec)
                persisted.append(fact)
            except Exception:
                logger.exception("SemanticIngestor: upsert_fact failed (fact_id=%s).", fact.id)

        logger.info("SemanticIngestor: persisted %d facts.", len(persisted))
        return persisted
