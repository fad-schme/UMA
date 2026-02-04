"""
chunk/core.py
=============

ChunkCore — unified interface for UMA document chunks.

Responsibilities
----------------
- Provide a stable API for chunk ingestion and retrieval
- Normalize ownership scoping
- Delegate persistence to ChunkSQLStore
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ...types_chunk import Chunk
from ..utils.identity import ensure_user_subject
from ..utils.dedupe import dedupe_by_id

logger = logging.getLogger(__name__)


class ChunkCore:
    """
    High-level interface for UMA chunk memory.
    """

    def __init__(self, chunk_store: Any) -> None:
        self.store = chunk_store
        logger.info("ChunkCore initialized.")

    # ------------------------------------------------------------------
    # PUBLIC API — ingest / CRUD
    # ------------------------------------------------------------------

    async def upsert_chunk(self, chunk: Chunk, embedding: List[float]) -> bool:
        if self.store is None:
            return False
        try:
            await self.store.upsert_chunk(chunk, embedding)
            return True
        except Exception:
            logger.exception("ChunkCore.upsert_chunk failed for id=%s", getattr(chunk, "id", None))
            return False

    # ------------------------------------------------------------------
    # PUBLIC API — retrieval
    # ------------------------------------------------------------------

    async def search(
        self,
        user_id: str,
        query_embedding: List[float],
        *,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        doc_id: Optional[str] = None,
    ) -> List[Chunk]:
        if self.store is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("ChunkCore.search: invalid subject=%r", user_id)
            return []

        chunks: List[Chunk] = []
        try:
            found = await self.store.search(
                query_embedding=query_embedding,
                doc_id=doc_id,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
            )
            if found:
                chunks.extend(found)
        except Exception:
            logger.exception(
                "ChunkCore.search failed owner=%s:%s",
                owner_type,
                owner_id,
            )
        return dedupe_by_id(chunks)

    async def search_text(
        self,
        user_id: str,
        query_text: str,
        *,
        owner_type: str,
        owner_id: str,
        k: int = 10,
    ) -> List[Chunk]:
        if self.store is None or not hasattr(self.store, "search_text"):
            return []
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("ChunkCore.search_text: invalid subject=%r", user_id)
            return []

        chunks: List[Chunk] = []
        try:
            found = await self.store.search_text(
                query_text,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
            )
            if found:
                chunks.extend(found)
        except Exception:
            logger.exception(
                "ChunkCore.search_text failed owner=%s:%s",
                owner_type,
                owner_id,
            )
        return dedupe_by_id(chunks)
