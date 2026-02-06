"""
chunk/core.py
=============

ChunkCore — unified interface for UMA document chunks.

Responsibilities
----------------
- Provide a stable API for chunk ingestion and retrieval
- Normalize ownership scoping
- Delegate persistence to ChunkSQLStore

Note for maintainers:
- All chunk retrieval must flow through `search_chunks`.
- Low-level helpers are intentionally prefixed with `_` and should not be
  called directly from other modules.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from ...types_chunk import Chunk
from ..utils.dedupe import dedupe_by_id

logger = logging.getLogger(__name__)


class ChunkCore:
    """
    High-level interface for UMA chunk memory.
    """

    def __init__(self, chunk_store: Any, *, memory: Optional[Any] = None) -> None:
        """
        Initialize ChunkCore with a backing store (e.g., ChunkSQLStore).
        """
        self.store = chunk_store
        self._memory = memory
        logger.debug("ChunkCore initialized.")

    # ------------------------------------------------------------------
    # PUBLIC API — ingest / CRUD
    # ------------------------------------------------------------------

    async def upsert_chunk(self, chunk: Chunk, embedding: List[float]) -> bool:
        """
        Persist a chunk and its embedding into the backing store and vector index.
        Returns True on success, False on failure.
        """
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

    async def _search(
        self,
        user_id: str,
        query_embedding: List[float],
        *,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        doc_id: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Internal vector-only chunk search (no lexical fallback or filtering).
        Used by `search_chunks` to build the primary candidate set.
        """
        if self.store is None:
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
            logger.exception("ChunkCore._search failed owner=%s:%s", owner_type, owner_id)
        return dedupe_by_id(chunks)

    async def _search_text(
        self,
        query_text: str,
        *,
        owner_type: str,
        owner_id: str,
        k: int = 10,
    ) -> List[Chunk]:
        """
        Internal lexical-only chunk search (SQL LIKE).
        Used by `search_chunks` as a fallback/merge signal.
        """
        if self.store is None or not hasattr(self.store, "search_text"):
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
            logger.exception("ChunkCore._search_text failed owner=%s:%s", owner_type, owner_id)
        return dedupe_by_id(chunks)

    async def _fetch_by_ids(
        self,
        ids: Sequence[str],
        *,
        owner_type: str,
        owner_id: str,
        log_context: str = "ChunkCore.fetch_by_ids",
    ) -> List[Chunk]:
        """
        Internal ID fetch for chunks (authoritative payload).
        Used by environment/controller for bounded evidence expansion.
        """
        if self.store is None or not hasattr(self.store, "fetch_by_ids"):
            return []
        if not ids:
            return []
        try:
            return await self.store.fetch_by_ids(
                ids=[str(x) for x in ids if x],
                log_context=log_context,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("ChunkCore._fetch_by_ids failed owner=%s:%s", owner_type, owner_id)
            return []

    async def _fetch_ranked_by_ids(
        self,
        ids: Sequence[str],
        *,
        owner_type: str,
        owner_id: str,
        log_context: str = "ChunkCore.fetch_ranked_by_ids",
    ) -> List[Chunk]:
        """
        Internal ranked ID fetch for chunks (store-level ordering).
        Used by retrieval for cited-evidence expansion.
        """
        if self.store is None or not hasattr(self.store, "_fetch_ranked_rows_by_ids"):
            return []
        if not ids:
            return []
        try:
            return await self.store._fetch_ranked_rows_by_ids(
                ids=[str(x) for x in ids if x],
                log_context=log_context,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("ChunkCore._fetch_ranked_by_ids failed owner=%s:%s", owner_type, owner_id)
            return []

    async def search_chunks(
        self,
        *,
        query_embedding: List[float],
        owner_type: str,
        owner_id: str,
        k: int = 10,
        query_text: Optional[str] = None,
        doc_id: Optional[str] = None,
        lexical_k: Optional[int] = None,
        filter_terms: bool = False,
    ) -> List[Chunk]:
        """
        Unified chunk search path for both baseline retrieval and RLM.

        Behavior:
        - Vector search as primary candidate set.
        - Optional lexical fallback + merge (lexical-first ordering).
        - Optional term filtering for strict keyword gating.
        - Deterministic dedupe + metadata tagging.
        """
        if self.store is None:
            return []
        try:
            k = int(k)
        except Exception:
            k = 10
        if k <= 0:
            return []

        chunks = await self._search(
            user_id="",
            query_embedding=query_embedding,
            owner_type=owner_type,
            owner_id=owner_id,
            k=k,
            doc_id=doc_id,
        )

        lexical_ids: set[str] = set()
        if query_text and query_text.strip():
            lk = None
            try:
                lk = int(lexical_k) if lexical_k is not None else None
            except Exception:
                lk = None
            if lk is None:
                lk = k
            if lk > 0:
                found: List[Chunk] = []
                try:
                    from ..utils.user_query_helper import extract_query_terms, expand_query_terms
                except Exception:
                    extract_query_terms = None
                    expand_query_terms = None

                use_fts = False
                store = getattr(self, "store", None)
                try:
                    if store is not None and hasattr(store, "search_fts5"):
                        from ...adapters.db.sqlite_adapter import SQLiteAdapter
                        adapter = getattr(store, "_db_adapter", None)
                        use_fts = isinstance(adapter, SQLiteAdapter)
                        if use_fts:
                            cfg = getattr(getattr(self, "_memory", None), "retrieval_cfg", None)
                            if cfg is not None and getattr(cfg, "fts5_enabled", True) is False:
                                use_fts = False
                except Exception:
                    use_fts = False

                if use_fts:
                    required_terms = extract_query_terms(query_text) if extract_query_terms else []
                    if not required_terms:
                        required_terms = [query_text.strip()]
                    optional_terms = expand_query_terms(query_text) if expand_query_terms else []
                    optional_terms = [t for t in optional_terms if t not in required_terms]
                    found = await store.search_fts5(
                        query_text=query_text,
                        owner_type=owner_type,
                        owner_id=owner_id,
                        k=lk,
                        required_terms=required_terms,
                        optional_terms=optional_terms,
                    )
                else:
                    found = await self._search_text(
                        query_text=query_text,
                        owner_type=owner_type,
                        owner_id=owner_id,
                        k=lk,
                    )
                if found:
                    for it in found:
                        cid = it.get("id") if isinstance(it, dict) else getattr(it, "id", None)
                        if cid:
                            lexical_ids.add(str(cid))
                    merged = list(chunks or []) + list(found or [])
                    chunks = dedupe_by_id(merged)

        if lexical_ids:
            def _chunk_id(x: Any) -> str:
                if isinstance(x, dict):
                    return str(x.get("id") or "")
                return str(getattr(x, "id", "") or "")

            def _apply_meta(ch: Any) -> None:
                try:
                    if isinstance(ch, dict):
                        meta = ch.get("meta") or {}
                        if not isinstance(meta, dict):
                            meta = {}
                        if _chunk_id(ch) in lexical_ids:
                            meta.setdefault("retrieval_method", "lexical")
                            meta.setdefault("lexical_score", 0.2)
                        else:
                            meta.setdefault("retrieval_method", "vector")
                        ch["meta"] = meta
                    else:
                        meta = getattr(ch, "meta", None) or {}
                        if not isinstance(meta, dict):
                            meta = {}
                        if _chunk_id(ch) in lexical_ids:
                            meta.setdefault("retrieval_method", "lexical")
                            meta.setdefault("lexical_score", 0.2)
                        else:
                            meta.setdefault("retrieval_method", "vector")
                        ch.meta = meta
                except Exception:
                    return

            chunks = sorted(
                chunks,
                key=lambda c: (0 if _chunk_id(c) in lexical_ids else 1),
            )
            for ch in chunks:
                _apply_meta(ch)

        if filter_terms and query_text and query_text.strip():
            try:
                from ..utils.user_query_helper import extract_query_terms, expand_query_terms
            except Exception:
                extract_query_terms = None
                expand_query_terms = None
            if expand_query_terms:
                terms = expand_query_terms(query_text)
            elif extract_query_terms:
                terms = extract_query_terms(query_text)
            else:
                terms = []
            terms = [t for t in (terms or []) if t]
            if terms:
                filtered: List[Any] = []
                for ch in chunks:
                    text = ch.get("text") if isinstance(ch, dict) else getattr(ch, "text", "")
                    if not text:
                        continue
                    low = str(text).lower()
                    if any(t.lower() in low for t in terms):
                        filtered.append(ch)
                chunks = filtered

        return chunks

    async def fetch_by_ids(
        self,
        ids: Sequence[str],
        *,
        owner_type: str,
        owner_id: str,
        log_context: str = "ChunkCore.fetch_by_ids",
    ) -> List[Chunk]:
        """
        Deprecated public wrapper for ID fetch (kept for compatibility).
        Prefer `_fetch_by_ids` in internal call sites.
        """
        return await self._fetch_by_ids(
            ids=ids,
            owner_type=owner_type,
            owner_id=owner_id,
            log_context=log_context,
        )
