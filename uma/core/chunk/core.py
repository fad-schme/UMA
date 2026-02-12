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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...types import Chunk
from ..utils.dedupe import dedupe_by_id

logger = logging.getLogger(__name__)


def partition_chunks_by_route(chunks: List[Chunk]) -> tuple[List[Chunk], List[Chunk], List[Chunk]]:
    """
    Partition chunks into (evidence, query_hits, neighbors) based on meta.retrieval_route.
    """
    evidence: List[Chunk] = []
    query_hits: List[Chunk] = []
    neighbors: List[Chunk] = []
    for ch in chunks or []:
        meta = getattr(ch, "meta", None) or {}
        route = meta.get("retrieval_route") if isinstance(meta, dict) else None
        if route == "evidence":
            evidence.append(ch)
        elif route == "neighbor":
            neighbors.append(ch)
        else:
            query_hits.append(ch)
    return evidence, query_hits, neighbors


def merge_chunks_with_precedence(
    evidence: List[Chunk],
    query_hits: List[Chunk],
    neighbors: List[Chunk],
) -> List[Chunk]:
    """
    Merge chunk buckets with deterministic precedence:
    1) evidence, 2) query hits, 3) neighbors.
    """
    return dedupe_by_id(list(evidence or []) + list(query_hits or []) + list(neighbors or []))


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
            raise

    # ------------------------------------------------------------------
    # PUBLIC API — retrieval
    # ------------------------------------------------------------------

    async def search_chunks_for_rlm(
        self,
        *,
        query_embedding: List[float],
        owner_type: str,
        owner_id: str,
        k: int,
        query_text: Optional[str],
    ) -> List[Chunk]:
        """
        RLM-friendly chunk retrieval wrapper.

        Centralizes retrieval_cfg defaults (lexical_k, neighbor expansion, shortlist)
        so RLM controller does not need to know chunk retrieval internals.
        """
        memory = getattr(self, "_memory", None)
        retrieval_cfg = getattr(memory, "retrieval_cfg", None) if memory is not None else None

        try:
            lexical_k = int(getattr(retrieval_cfg, "lexical_chunks_k", 15))
        except Exception:
            lexical_k = 15
        try:
            neighbor_window = int(getattr(retrieval_cfg, "neighbor_window", 1))
        except Exception:
            neighbor_window = 1
        try:
            max_expanded_chunks = int(getattr(retrieval_cfg, "max_expanded_chunks", 24))
        except Exception:
            max_expanded_chunks = 24
        try:
            shortlist_k = int(getattr(retrieval_cfg, "chunk_shortlist_k", 12))
        except Exception:
            shortlist_k = 12
        try:
            shortlist_max_per_doc = int(getattr(retrieval_cfg, "chunk_shortlist_max_per_doc", 3))
        except Exception:
            shortlist_max_per_doc = 3

        return await self.search_chunks(
            query_embedding=list(query_embedding),
            owner_type=owner_type,
            owner_id=owner_id,
            k=int(k),
            query_text=query_text,
            lexical_k=lexical_k,
            filter_terms=bool(query_text and str(query_text).strip()),
            expand_neighbors=True,
            neighbor_window=neighbor_window,
            max_expanded_chunks=max_expanded_chunks,
            shortlist_k=shortlist_k,
            shortlist_max_per_doc=shortlist_max_per_doc,
        )

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
            raise
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
            raise
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
            raise

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
            raise

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
        expand_neighbors: bool = False,
        neighbor_window: int = 1,
        max_expanded_chunks: int = 24,
        shortlist_k: Optional[int] = None,
        shortlist_max_per_doc: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Unified chunk search path for both baseline retrieval and RLM.

        Scoping contract:
        - Facts (semantic memory) may apply *subject* filtering (user-specific vs corpus-wide).
        - Chunks MUST NOT use subject filtering. They are scoped only by:
          `owner_type`/`owner_id` (DAT) and optionally `doc_id`.

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
        if any(isinstance(x, dict) for x in (chunks or [])):
            logger.error("ChunkCore.search_chunks: expected Chunk objects from store.search; got dict(s)")
            raise TypeError(
                "ChunkCore expected Chunk objects from store.search(); got dict(s). "
                "Fix the chunk store to return Chunk only."
            )
        logger.debug(
            "ChunkCore.search_chunks vector owner=%s:%s k=%s results=%d doc_id=%s",
            owner_type,
            owner_id,
            k,
            len(chunks),
            doc_id,
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
                logger.debug(
                    "ChunkCore.search_chunks lexical=like owner=%s:%s lk=%s query_text=%r",
                    owner_type,
                    owner_id,
                    lk,
                    query_text,
                )
                found = await self._search_text(
                    query_text=query_text,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    k=lk,
                )
                logger.debug(
                    "ChunkCore.search_chunks lexical results=%d",
                    len(found),
                )
                if found:
                    for it in found:
                        if isinstance(it, dict):
                            logger.error("ChunkCore.search_chunks: expected Chunk objects from store.search_text; got dict")
                            raise TypeError(
                                "ChunkCore expected Chunk objects from store.search_text(); got dict. "
                                "Fix the chunk store to return Chunk only."
                            )
                        cid = getattr(it, "id", None)
                        if cid:
                            lexical_ids.add(str(cid))
                    merged = list(chunks or []) + list(found or [])
                    chunks = dedupe_by_id(merged)
        logger.debug(
            "ChunkCore.search_chunks merged_results=%d lexical_ids=%d filter_terms=%s",
            len(chunks),
            len(lexical_ids),
            filter_terms,
        )

        if lexical_ids:
            def _chunk_id(ch: Chunk) -> str:
                return str(ch.id or "")

            def _apply_meta(ch: Chunk) -> None:
                meta = getattr(ch, "meta", None) or {}
                if not isinstance(meta, dict):
                    meta = {}
                meta.setdefault("retrieval_route", "query")
                meta.setdefault("retrieval_stage", "search")
                if _chunk_id(ch) in lexical_ids:
                    meta.setdefault("retrieval_method", "lexical")
                    meta.setdefault("lexical_score", 0.2)
                else:
                    meta.setdefault("retrieval_method", "vector")
                ch.meta = meta

            chunks = sorted(
                chunks,
                key=lambda c: (0 if _chunk_id(c) in lexical_ids else 1),
            )
            for ch in chunks:
                _apply_meta(ch)
        else:
            # Vector-only path: still attach minimal provenance for downstream consumers.
            try:
                for ch in chunks:
                    if isinstance(ch, dict):
                        logger.error("ChunkCore.search_chunks: expected Chunk objects from store.search; got dict")
                        raise TypeError(
                            "ChunkCore expected Chunk objects from store.search(); got dict. "
                            "Fix the chunk store to return Chunk only."
                        )
                    meta = getattr(ch, "meta", None) or {}
                    if not isinstance(meta, dict):
                        meta = {}
                    meta.setdefault("retrieval_route", "query")
                    meta.setdefault("retrieval_stage", "search")
                    meta.setdefault("retrieval_method", "vector")
                    ch.meta = meta
            except Exception:
                logger.exception("ChunkCore.search_chunks: failed to attach retrieval metadata")
                raise

            try:
                confs = []
                for ch in chunks:
                    meta = getattr(ch, "meta", None)
                    if isinstance(meta, dict) and meta.get("retrieval_method") == "lexical":
                        c = meta.get("lexical_confidence")
                        if c is not None:
                            confs.append(float(c))
                if confs:
                    logger.info(
                        "ChunkCore.search_chunks lexical_confidence n=%d avg=%.2f max=%.2f",
                        len(confs),
                        sum(confs) / max(1, len(confs)),
                        max(confs),
                    )
            except Exception:
                logger.exception("ChunkCore.search_chunks: failed to summarize lexical_confidence")
                raise

        if filter_terms and query_text and query_text.strip():
            from ..utils.user_query_helper import build_query_term_set
            term_set = build_query_term_set(query_text)
            if term_set and (term_set.terms or term_set.phrases):
                from ..utils.user_query_helper import text_matches_query_terms
                filtered: List[Chunk] = []
                for ch in chunks:
                    if isinstance(ch, dict):
                        logger.error("ChunkCore.search_chunks: expected Chunk objects from store.search; got dict")
                        raise TypeError(
                            "ChunkCore expected Chunk objects from store.search(); got dict. "
                            "Fix the chunk store to return Chunk only."
                        )
                    text = getattr(ch, "text", "")
                    if not text:
                        continue
                    if text_matches_query_terms(text, term_set, min_term_matches=2, max_terms_for_match=6):
                        filtered.append(ch)
                chunks = filtered

        if expand_neighbors and chunks:
            try:
                anchors = self._shortlist_for_neighbor_expansion(
                    chunks,
                    shortlist_k=shortlist_k,
                    shortlist_max_per_doc=shortlist_max_per_doc,
                )
                chunks = await self.expand_neighbors(
                    owner_type=owner_type,
                    owner_id=owner_id,
                    anchors=anchors,
                    window=neighbor_window,
                    max_total=max_expanded_chunks,
                )
            except Exception:
                logger.exception("ChunkCore.search_chunks neighbor expansion failed")
                raise
        return chunks

    @staticmethod
    def _shortlist_for_neighbor_expansion(
        chunks: List[Chunk],
        *,
        shortlist_k: Optional[int],
        shortlist_max_per_doc: Optional[int],
    ) -> List[Chunk]:
        """
        Select a deterministic subset of chunks to serve as neighbor-expansion anchors.

        `chunks` is assumed to already be ranked (lexical-first ordering applied).
        """
        try:
            k = int(shortlist_k) if shortlist_k is not None else None
        except Exception:
            k = None
        try:
            per_doc = int(shortlist_max_per_doc) if shortlist_max_per_doc is not None else None
        except Exception:
            per_doc = None

        if k is None or k <= 0:
            k = len(chunks)
        if per_doc is None or per_doc <= 0:
            per_doc = k

        out: List[Chunk] = []
        per_doc_counts: Dict[str, int] = {}
        for ch in chunks:
            if len(out) >= k:
                break
            doc = str(getattr(ch, "doc_id", "") or "")
            if not doc:
                continue
            n = per_doc_counts.get(doc, 0)
            if n >= per_doc:
                continue
            per_doc_counts[doc] = n + 1
            out.append(ch)
        return out

    async def expand_neighbors(
        self,
        *,
        owner_type: str,
        owner_id: str,
        anchors: List[Chunk],
        window: int = 1,
        max_total: int = 24,
    ) -> List[Chunk]:
        """
        Deterministically expand anchor chunks by doc-local adjacency (position ± window),
        bounded by max_total, with strict owner scope.

        Ordering policy: anchor-first. For each anchor in rank order, append the anchor
        (if not already added) then its neighbors in doc-local position order.
        """
        if not anchors:
            return []
        if self.store is None or not hasattr(self.store, "fetch_by_doc_and_position_range"):
            return list(anchors)[: max(0, int(max_total or 0))]

        try:
            window_i = max(0, int(window))
        except Exception:
            window_i = 1
        try:
            max_total_i = max(0, int(max_total))
        except Exception:
            max_total_i = 24
        if max_total_i <= 0:
            return []

        if any(isinstance(x, dict) for x in anchors):
            logger.error("ChunkCore.expand_neighbors: expected Chunk anchors; got dict(s)")
            raise TypeError("ChunkCore.expand_neighbors expected Chunk anchors; got dict(s).")

        # Compute merged ranges per doc to minimize SQL calls.
        ranges_by_doc: Dict[str, List[Tuple[int, int]]] = {}
        for a in anchors:
            doc_id = str(getattr(a, "doc_id", "") or "")
            if not doc_id:
                continue
            try:
                pos = int(getattr(a, "position", 0) or 0)
            except Exception:
                pos = 0
            start = max(1, pos - window_i)
            end = max(1, pos + window_i)
            ranges_by_doc.setdefault(doc_id, []).append((start, end))

        def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            if not ranges:
                return []
            rs = sorted(ranges, key=lambda x: (x[0], x[1]))
            merged: List[Tuple[int, int]] = []
            cur_s, cur_e = rs[0]
            for s, e in rs[1:]:
                if s <= cur_e + 1:
                    cur_e = max(cur_e, e)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
            return merged

        merged_ranges_by_doc = {doc: _merge_ranges(rgs) for doc, rgs in ranges_by_doc.items()}

        # Fetch rows in deterministic position order.
        fetched_by_doc: Dict[str, List[Chunk]] = {}
        for doc_id, ranges in merged_ranges_by_doc.items():
            fetched: List[Chunk] = []
            for s, e in ranges:
                try:
                    rows = await self.store.fetch_by_doc_and_position_range(
                        owner_type=owner_type,
                        owner_id=owner_id,
                        doc_id=doc_id,
                        pos_start=s,
                        pos_end=e,
                    )
                except Exception:
                    logger.exception("ChunkCore.expand_neighbors fetch failed doc_id=%s", doc_id)
                    rows = []
                if rows:
                    fetched.extend(rows)
            fetched_by_doc[doc_id] = dedupe_by_id(fetched)

        by_doc_pos: Dict[str, Dict[int, Chunk]] = {}
        for doc_id, rows in fetched_by_doc.items():
            pos_map: Dict[int, Chunk] = {}
            for ch in rows:
                try:
                    pos = int(getattr(ch, "position", 0) or 0)
                except Exception:
                    pos = 0
                if pos and getattr(ch, "id", None) and pos not in pos_map:
                    pos_map[pos] = ch
            by_doc_pos[doc_id] = pos_map

        out: List[Chunk] = []
        seen: set[str] = set()

        for a in anchors:
            if len(out) >= max_total_i:
                break
            aid = str(getattr(a, "id", "") or "")
            if aid and aid not in seen:
                meta = getattr(a, "meta", None) or {}
                if not isinstance(meta, dict):
                    meta = {}
                meta.setdefault("retrieval_route", "query")
                meta.setdefault("retrieval_stage", "neighbor_expand")
                a.meta = meta
                out.append(a)
                seen.add(aid)
                if len(out) >= max_total_i:
                    break

            doc_id = str(getattr(a, "doc_id", "") or "")
            pos_map = by_doc_pos.get(doc_id) or {}
            try:
                pos = int(getattr(a, "position", 0) or 0)
            except Exception:
                pos = 0
            for p in range(max(1, pos - window_i), max(1, pos + window_i) + 1):
                if len(out) >= max_total_i:
                    break
                ch = pos_map.get(p)
                if not ch:
                    continue
                cid = str(getattr(ch, "id", "") or "")
                if not cid or cid in seen:
                    continue
                meta = getattr(ch, "meta", None) or {}
                if not isinstance(meta, dict):
                    meta = {}
                meta.setdefault("retrieval_route", "neighbor")
                meta.setdefault("retrieval_stage", "neighbor_expand")
                meta.setdefault("expanded_from", aid)
                ch.meta = meta
                out.append(ch)
                seen.add(cid)

        return out

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
