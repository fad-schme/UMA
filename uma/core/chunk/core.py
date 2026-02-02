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
from typing import Any, List, Optional, Tuple

from ...types_chunk import Chunk
from ..utils.identity import ensure_user_subject

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

    @staticmethod
    def _iter_owner_filters(
        *,
        user_subject: str,
        agent_id: Optional[str],
        project_id: Optional[str],
        owner_scope: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        scope = (owner_scope or "").lower()
        if scope:
            if scope == "user":
                return [("user", user_subject)]
            if scope == "agent" and agent_id:
                return [("agent", agent_id)]
            if scope == "project" and project_id:
                return [("project", f"{user_subject}:{project_id}")]
            return []

        filters: List[Tuple[str, str]] = [("user", user_subject)]
        if agent_id:
            filters.append(("agent", agent_id))
        if project_id:
            filters.append(("project", f"{user_subject}:{project_id}"))
        return filters

    async def search_tiered(
        self,
        user_id: str,
        query_embedding: List[float],
        *,
        k: int = 10,
        doc_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        owner_scope: Optional[str] = None,
    ) -> List[Chunk]:
        if self.store is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("ChunkCore.search_tiered: invalid subject=%r", user_id)
            return []

        chunks: List[Chunk] = []
        for owner_type, owner_id in self._iter_owner_filters(
            user_subject=user_subject,
            agent_id=agent_id,
            project_id=project_id,
            owner_scope=owner_scope,
        ):
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
                    "ChunkCore.search_tiered failed owner=%s:%s",
                    owner_type,
                    owner_id,
                )
        return _dedupe_items(chunks)

    async def search_text_tiered(
        self,
        user_id: str,
        query_text: str,
        *,
        k: int = 10,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        owner_scope: Optional[str] = None,
    ) -> List[Chunk]:
        if self.store is None or not hasattr(self.store, "search_text"):
            return []
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("ChunkCore.search_text_tiered: invalid subject=%r", user_id)
            return []

        chunks: List[Chunk] = []
        for owner_type, owner_id in self._iter_owner_filters(
            user_subject=user_subject,
            agent_id=agent_id,
            project_id=project_id,
            owner_scope=owner_scope,
        ):
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
                    "ChunkCore.search_text_tiered failed owner=%s:%s",
                    owner_type,
                    owner_id,
                )
        return _dedupe_items(chunks)

    async def search(
        self,
        query_embedding: List[float],
        *,
        doc_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 10,
    ) -> List[Chunk]:
        if self.store is None:
            return []
        try:
            return await self.store.search(
                query_embedding=query_embedding,
                doc_id=doc_id,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
            )
        except Exception:
            logger.exception("ChunkCore.search failed")
            return []

    async def search_text(
        self,
        query_text: str,
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 10,
    ) -> List[Chunk]:
        if self.store is None:
            return []
        try:
            if not hasattr(self.store, "search_text"):
                return []
            return await self.store.search_text(
                query_text,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
            )
        except Exception:
            logger.exception("ChunkCore.search_text failed")
            return []


def _dedupe_items(items: List[Any]) -> List[Any]:
    if not items:
        return []
    seen = set()
    out: List[Any] = []
    for it in items:
        key = None
        if isinstance(it, dict):
            key = it.get("id")
        else:
            key = getattr(it, "id", None)
        if key is None:
            key = id(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out
