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
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from uma.common.types.types_scope import DEFAULT_TENANT_ID
from uma.common.types import Chunk
from uma.retrieve.ranking import compute_rerank_score, fuse_candidates, mmr_select
from uma.common.dedupe import dedupe_by_id
from uma.common.text import build_query_term_set, text_matches_query_terms
from uma.retrieve.rlm.intent import QueryIntent, classify_query_intent
from uma.retrieve.rlm.query_decomposition import decompose_query

logger = logging.getLogger(__name__)


def partition_chunks_by_route(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk], list[Chunk]]:
    """
    Partition chunks into (evidence, query_hits, neighbors) based on meta.retrieval_route.
    """
    evidence: list[Chunk] = []
    query_hits: list[Chunk] = []
    neighbors: list[Chunk] = []
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
    evidence: list[Chunk],
    query_hits: list[Chunk],
    neighbors: list[Chunk],
) -> list[Chunk]:
    """
    Merge chunk buckets with deterministic precedence:
    1) evidence, 2) query hits, 3) neighbors.
    """
    return dedupe_by_id(list(evidence or []) + list(query_hits or []) + list(neighbors or []))


@dataclass
class ChunkSearchOptions:
    query_text: Optional[str] = None
    doc_id: Optional[str] = None
    filter_terms: bool = False
    expand_neighbors: bool = False
    neighbor_window: int = 1
    max_expanded_chunks: int = 24
    shortlist_k: Optional[int] = None
    shortlist_max_per_doc: Optional[int] = None


# LoCoMo's sliding-window ingest pairing (and any adapter with similar
# turn-pairing) stores the same turn text twice under different chunk ids —
# once as a user_msg, once as the prior turn's assistant_reply. Uncorrected,
# these exact-duplicate rows fill half of every top-k candidate pool with
# zero added information, halving the pool's effective diversity for
# broad/enumeration-style queries where the answer is scattered across many
# distinct turns. Over-fetching by this multiplier before deduping restores
# the caller's requested k in genuinely distinct candidates.
_DEDUPE_OVERFETCH_MULTIPLIER = 2


def _normalize_chunk_text(text: Optional[str]) -> str:
    return " ".join((text or "").split()).strip().lower()


def dedupe_chunks_by_text(chunks: Sequence[Chunk]) -> list[Chunk]:
    """Drop exact-duplicate-text chunks, keeping the first (highest-ranked) occurrence."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for ch in chunks or []:
        key = _normalize_chunk_text(getattr(ch, "text", None))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(ch)
    return out


def _primary_query_share(query_text: Optional[str], sub_queries: list[str]) -> float:
    """Fraction of the k budget the primary query keeps once decomposition fires.

    A flat 50/50 split (retrieval-ranking-gap ticket 03) taxes the primary
    query even when its decomposed sub-queries barely diverge from it
    lexically -- decomposition adds little new signal in that case, so the
    primary should keep more of its candidate budget. Genuinely divergent
    sub-queries (broad, multi-facet questions) keep close to the original
    even split, which is where decomposition earns its keep.
    """
    if not sub_queries:
        return 1.0
    primary_set = build_query_term_set(query_text or "")
    primary_terms = set(primary_set.terms) | set(primary_set.phrases)
    if not primary_terms:
        return 0.5
    overlaps: list[float] = []
    for sub_q in sub_queries:
        sub_set = build_query_term_set(sub_q or "")
        sub_terms = set(sub_set.terms) | set(sub_set.phrases)
        if not sub_terms:
            # No lexical signal at all -- neither confirms nor contradicts
            # overlap with the primary, so it shouldn't skew the average
            # either way (excluding it, not treating it as full overlap).
            continue
        union = primary_terms | sub_terms
        overlaps.append(len(primary_terms & sub_terms) / len(union) if union else 1.0)
    if not overlaps:
        return 0.5
    avg_overlap = sum(overlaps) / len(overlaps)
    # 0.5 at zero overlap (today's flat split) up to 0.8 at full overlap
    # (sub-queries that restate the primary keep it well-resourced).
    return 0.5 + 0.3 * avg_overlap


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

    async def upsert_chunk(self, chunk: Chunk, embedding: list[float]) -> bool:
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
        query_embedding: list[float],
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        k: int,
        query_text: Optional[str],
        query_scan_severity: Optional[str] = None,
    ) -> list[Chunk]:
        """
        RLM-friendly chunk retrieval wrapper.

        Centralizes retrieval_cfg defaults (neighbor expansion, shortlist)
        so RLM controller does not need to know chunk retrieval internals.
        """
        memory = getattr(self, "_memory", None)
        retrieval_cfg = getattr(memory, "retrieval_cfg", None) if memory is not None else None
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

        k = int(k)
        rlm_cfg = getattr(retrieval_cfg, "rlm", None)
        decomposition_enabled = bool(getattr(rlm_cfg, "query_decomposition_enabled", True))

        sub_queries: list[str] = []
        if (
            decomposition_enabled
            and query_text
            and str(query_text).strip()
            and classify_query_intent(query_text) != QueryIntent.PERSONAL
        ):
            max_sub_queries = int(getattr(rlm_cfg, "query_decomposition_max_sub_queries", 4))
            sub_queries = await decompose_query(
                getattr(memory, "llm", None),
                query_text,
                max_sub_queries=max_sub_queries,
                query_scan_severity=query_scan_severity,
            )

        if not sub_queries:
            return await self.search_chunks(
                query_embedding=list(query_embedding),
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                k=k,
                options=ChunkSearchOptions(
                    query_text=query_text,
                    filter_terms=bool(query_text and str(query_text).strip()),
                    expand_neighbors=True,
                    neighbor_window=neighbor_window,
                    max_expanded_chunks=max_expanded_chunks,
                    shortlist_k=shortlist_k,
                    shortlist_max_per_doc=shortlist_max_per_doc,
                ),
            )

        # Split the caller's k budget between the original query and its
        # decomposed sub-queries rather than searching each at full k: the
        # merge upstream (RLMController._merge_unique) truncates positionally
        # at k, so returning more than k here would just have the tail
        # silently discarded instead of genuinely competing for a slot.
        # A flat 50/50 split taxes the primary query even when the
        # sub-queries barely diverge from it lexically (retrieval-ranking-gap
        # ticket 05: a single-fact query that decomposition fires on anyway
        # loses candidate slots it didn't need to spend). Weight the split by
        # how much the sub-queries actually diverge from the primary instead.
        # int() truncation (not round()) so share=0.5 reproduces the old
        # k // 2 split exactly; capped at k - 1 so decomposition, once fired,
        # always leaves the sub-queries at least one slot to compete for
        # (share's ceiling of 0.8 would otherwise reach k itself at k=2 and
        # silently skip the sub-query search block entirely).
        primary_k = max(1, int(k * _primary_query_share(query_text, sub_queries)))
        if k > 1:
            primary_k = min(primary_k, k - 1)
        primary = await self.search_chunks(
            query_embedding=list(query_embedding),
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            k=primary_k,
            options=ChunkSearchOptions(
                query_text=query_text,
                filter_terms=bool(query_text and str(query_text).strip()),
                expand_neighbors=True,
                neighbor_window=neighbor_window,
                max_expanded_chunks=max_expanded_chunks,
                shortlist_k=shortlist_k,
                shortlist_max_per_doc=shortlist_max_per_doc,
            ),
        )

        remaining = max(0, k - primary_k)
        per_sub_k = max(1, remaining // len(sub_queries)) if remaining > 0 else 0
        extra: list[Chunk] = []
        if per_sub_k > 0:
            embedder = getattr(memory, "embedder", None)
            sub_embeddings: list[list[float]] = []
            if embedder is not None:
                try:
                    sub_embeddings = await embedder.embed(sub_queries)
                except Exception:
                    logger.exception(
                        "ChunkCore.search_chunks_for_rlm: failed embedding decomposed sub-queries"
                    )
                    sub_embeddings = []
            for sub_q, sub_embedding in zip(sub_queries, sub_embeddings):
                try:
                    sub_chunks = await self.search_chunks(
                        query_embedding=list(sub_embedding),
                        tenant_id=tenant_id,
                        owner_type=owner_type,
                        owner_id=owner_id,
                        k=per_sub_k,
                        options=ChunkSearchOptions(
                            query_text=sub_q,
                            filter_terms=False,
                            expand_neighbors=False,
                        ),
                    )
                    extra.extend(sub_chunks)
                except Exception:
                    logger.exception(
                        "ChunkCore.search_chunks_for_rlm: sub-query search failed sub_query=%r", sub_q
                    )

        logger.debug(
            "ChunkCore.search_chunks_for_rlm: query decomposed into %d sub-queries, "
            "primary=%d sub_hits=%d",
            len(sub_queries), len(primary), len(extra),
        )
        combined = dedupe_chunks_by_text(dedupe_by_id(list(primary) + extra))
        return combined[:k]

    async def _search(
        self,
        user_id: str,
        query_embedding: list[float],
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        doc_id: Optional[str] = None,
    ) -> list[Chunk]:
        """
        Internal vector-only chunk search (no lexical fallback or filtering).
        Used by `search_chunks` to build the primary candidate set.
        """
        if self.store is None:
            return []
        chunks: list[Chunk] = []
        try:
            found = await self.store.search(
                query_embedding=query_embedding,
                tenant_id=tenant_id,
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

    async def _lexical_search(
        self,
        query_text: str,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        k: int = 10,
    ) -> list[Chunk]:
        """
        Internal lexical-only chunk search (SQL LIKE).
        Used by `search_chunks` as a fallback/merge signal.
        """
        if self.store is None or not hasattr(self.store, "lexical_search"):
            return []
        chunks: list[Chunk] = []
        try:
            found = await self.store.lexical_search(
                query_text=query_text,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(k),
            )
            if found:
                chunks.extend(found)
        except Exception:
            logger.exception("ChunkCore._lexical_search failed owner=%s:%s", owner_type, owner_id)
            raise
        return dedupe_by_id(chunks)

    async def _fetch_by_ids(
        self,
        ids: Sequence[str],
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        log_context: str = "ChunkCore.fetch_by_ids",
    ) -> list[Chunk]:
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
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
        except Exception:
            logger.exception("ChunkCore._fetch_by_ids failed owner=%s:%s", owner_type, owner_id)
            raise

    async def search_chunks(
        self,
        *,
        query_embedding: list[float],
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        options: Optional[ChunkSearchOptions] = None,
    ) -> list[Chunk]:
        """
        Unified chunk search: vector → hybrid → term-filter → neighbor expansion.

        Scoping: chunks are scoped only by owner_type/owner_id and optionally doc_id.
        Subject filtering must NOT be applied here.
        """
        if self.store is None:
            return []
        try:
            k = int(k)
        except Exception:
            k = 10
        if k <= 0:
            return []
        opts = options or ChunkSearchOptions()

        memory = getattr(self, "_memory", None)
        retrieval_cfg = getattr(memory, "retrieval_cfg", None) if memory is not None else None
        mmr_enabled = bool(getattr(retrieval_cfg, "mmr_enabled", False))
        # MMR needs real headroom to pick diversity from -- the plain top-k
        # path's 2x overfetch (enough margin for exact-duplicate dedup) is
        # too small a pool for MMR to meaningfully diversify within
        # (retrieval-ranking-gap ticket 07: verified regressing the full
        # benchmark at the 2x pool before this was widened). 6x matches the
        # pool/k ratio ticket 06's prototype validated on.
        overfetch_multiplier = (
            int(getattr(retrieval_cfg, "mmr_pool_multiplier", 6)) if mmr_enabled else _DEDUPE_OVERFETCH_MULTIPLIER
        )
        chunks, lexical_ids = await self._execute_hybrid_search(
            query_embedding=query_embedding,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            k=k * overfetch_multiplier,
            query_text=opts.query_text,
            doc_id=opts.doc_id,
        )
        deduped = dedupe_chunks_by_text(chunks)
        dropped = len(chunks) - len(deduped)
        # Tag retrieval metadata (lexical_score, retrieval_method) before the
        # top-k cut, not after: MMR's relevance signal is compute_rerank_score,
        # which reads lexical_score off chunk.meta -- selection must see it.
        self._tag_retrieval_metadata(deduped, lexical_ids)
        chunks = await self._select_chunks(
            deduped, k, query_text=opts.query_text, tenant_id=tenant_id, owner_type=owner_type, owner_id=owner_id,
        )
        logger.debug(
            "ChunkCore.search_chunks merged_results=%d duplicate_text_dropped=%d kept=%d "
            "lexical_ids=%d filter_terms=%s",
            len(deduped) + dropped,
            dropped,
            len(chunks),
            len(lexical_ids),
            opts.filter_terms,
        )
        if opts.filter_terms and opts.query_text and opts.query_text.strip():
            chunks = self._apply_term_filter(chunks, opts.query_text)
        if opts.expand_neighbors and chunks:
            try:
                anchors = self._shortlist_for_neighbor_expansion(
                    chunks,
                    shortlist_k=opts.shortlist_k,
                    shortlist_max_per_doc=opts.shortlist_max_per_doc,
                )
                chunks = await self.expand_neighbors(
                    tenant_id=tenant_id,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    anchors=anchors,
                    window=opts.neighbor_window,
                    max_total=opts.max_expanded_chunks,
                )
            except Exception:
                logger.exception("ChunkCore.search_chunks neighbor expansion failed")
                raise
        return chunks

    async def _select_chunks(
        self,
        candidates: list[Chunk],
        k: int,
        *,
        query_text: Optional[str],
        tenant_id: str,
        owner_type: str,
        owner_id: str,
    ) -> list[Chunk]:
        """
        Top-k cut for a deduped candidate pool.

        Plain score-order slice by default. When `retrieval.mmr_enabled` is
        set, selects via MMR instead (retrieval-ranking-gap ticket 07) --
        picking for topical diversity against what's already selected, not
        just highest score, so a scattered but genuinely distinct candidate
        isn't crowded out by several near-duplicate ones. Falls back to the
        plain slice whenever the pool is already <= k (nothing to select
        between) or the vector backend can't provide vectors for every
        candidate (a partial MMR pass over an incomplete pool would be a
        silent quality regression, not a diversity improvement).

        MMR's relevance signal is `compute_rerank_score` (the same lexical +
        vector formula the downstream `Ranker` uses), not raw `vector_score`.
        An earlier version used vector_score alone and measurably regressed
        the full pipeline: it competed with, rather than complemented, the
        reranker that reorders `search_chunks`'s output afterward -- pruning
        the pool for diversity before the reranker's term/phrase/entity
        signals ever got a vote meant a candidate that formula would have
        surfaced could be gone by the time it ran, with no way back in.

        Known tradeoff, not fixed here: `compute_rerank_score` now runs
        twice for every surviving candidate on an MMR-enabled search --
        once here, again when the RLM controller's `Ranker.rank_chunks`
        reorders `search_chunks`'s output downstream. That second pass
        fully re-sorts by its own score, so MMR only ever determines which
        k candidates survive, never their final order -- a real "two
        scoring passes over the same data" situation `ranking.py`'s module
        docstring says to avoid ("single owner" for ranking math). Verified
        empirically to still produce a net recall gain with zero
        regressions on the 15-question benchmark (ticket 07's Findings),
        so left as a documented cost rather than risking a rearchitecture
        (MMR living inside `Ranker`, or applied by the controller after
        `rank_chunks` instead of inside `ChunkCore`) without dedicated
        verification budget for that redesign.
        """
        if len(candidates) <= k:
            return candidates[:k]

        memory = getattr(self, "_memory", None)
        retrieval_cfg = getattr(memory, "retrieval_cfg", None) if memory is not None else None
        if not bool(getattr(retrieval_cfg, "mmr_enabled", False)):
            return candidates[:k]

        vector_index = getattr(self.store, "vector_index", None)
        vectors: dict[str, list[float]] = {}
        if vector_index is not None:
            try:
                vectors = vector_index.get_vectors(
                    [ch.id for ch in candidates],
                    tenant_id=tenant_id,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            except Exception:  # nosec B110 -- deliberate fallback to plain top-k, not a masked invariant
                logger.debug(
                    "ChunkCore._select_chunks: get_vectors failed; falling back to top-k", exc_info=True
                )
                vectors = {}

        missing = [ch.id for ch in candidates if ch.id not in vectors]
        if missing:
            logger.debug(
                "ChunkCore._select_chunks: vectors unavailable for %d/%d candidates; falling back to top-k",
                len(missing),
                len(candidates),
            )
            return candidates[:k]

        mmr_lambda = float(getattr(retrieval_cfg, "mmr_lambda", 0.6))
        by_id = {ch.id: ch for ch in candidates}
        triples = [
            (ch.id, compute_rerank_score(query_text=query_text or "", candidate=ch), vectors[ch.id])
            for ch in candidates
        ]
        selected_ids = mmr_select(triples, k=k, lambda_diversity=mmr_lambda)
        return [by_id[sid] for sid in selected_ids if sid in by_id]

    async def _execute_hybrid_search(
        self,
        *,
        query_embedding: list[float],
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        k: int,
        query_text: Optional[str],
        doc_id: Optional[str],
    ) -> tuple[list[Chunk], set[str]]:
        memory = getattr(self, "_memory", None)
        retrieval_cfg = getattr(memory, "retrieval_cfg", None) if memory is not None else None
        hybrid_cfg = getattr(retrieval_cfg, "hybrid", None) if retrieval_cfg is not None else None
        hybrid_enabled = bool(getattr(hybrid_cfg, "enabled", True)) if hybrid_cfg is not None else True
        fusion_strategy = str(getattr(hybrid_cfg, "fusion_strategy", "rrf") or "rrf") if hybrid_cfg is not None else "rrf"
        try:
            top_k_dense = int(getattr(hybrid_cfg, "top_k_dense", 0) or 0) if hybrid_cfg is not None else 0
        except Exception:
            top_k_dense = 0
        try:
            top_k_sparse = int(getattr(hybrid_cfg, "top_k_sparse", 15) or 15) if hybrid_cfg is not None else 15
        except Exception:
            top_k_sparse = 15
        dense_k = top_k_dense if top_k_dense > 0 else k

        chunks = await self._search(
            user_id="",
            query_embedding=query_embedding,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            k=dense_k,
            doc_id=doc_id,
        )
        if any(isinstance(x, dict) for x in (chunks or [])):
            logger.error("ChunkCore._execute_hybrid_search: expected Chunk objects from store.search; got dict(s)")
            raise TypeError("ChunkCore expected Chunk objects from store.search(); got dict(s).")
        logger.debug(
            "ChunkCore.search_chunks vector owner=%s:%s k=%s results=%d doc_id=%s",
            owner_type, owner_id, k, len(chunks), doc_id,
        )

        dense_chunks = list(chunks or [])
        lexical_ids: set[str] = set()
        if hybrid_enabled and top_k_sparse > 0 and query_text and query_text.strip():
            logger.debug(
                "ChunkCore.search_chunks lexical=enabled owner=%s:%s k_sparse=%s strategy=%s query_text=%r",
                owner_type, owner_id, top_k_sparse, fusion_strategy, query_text,
            )
            found = await self._lexical_search(
                query_text=query_text,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                k=int(top_k_sparse),
            )
            logger.debug("ChunkCore.search_chunks lexical results=%d", len(found))
            if found:
                for it in found:
                    if isinstance(it, dict):
                        logger.error("ChunkCore._execute_hybrid_search: expected Chunk objects from store.lexical_search; got dict")
                        raise TypeError("ChunkCore expected Chunk objects from store.lexical_search(); got dict.")
                    cid = getattr(it, "id", None)
                    if cid:
                        lexical_ids.add(str(cid))
                chunks = fuse_candidates(dense=dense_chunks, sparse=list(found), strategy=fusion_strategy)
        return chunks, lexical_ids

    def _tag_retrieval_metadata(self, chunks: list[Chunk], lexical_ids: set[str]) -> None:
        if lexical_ids:
            for ch in chunks:
                meta = getattr(ch, "meta", None) or {}
                if not isinstance(meta, dict):
                    meta = {}
                meta.setdefault("retrieval_route", "query")
                meta.setdefault("retrieval_stage", "search")
                if str(ch.id or "") in lexical_ids:
                    meta.setdefault("retrieval_method", "lexical")
                    meta.setdefault("lexical_score", float(meta.get("lexical_confidence", 0.2) or 0.2))
                else:
                    meta.setdefault("retrieval_method", "vector")
                ch.meta = meta
        else:
            try:
                for ch in chunks:
                    if isinstance(ch, dict):
                        logger.error("ChunkCore._tag_retrieval_metadata: expected Chunk objects; got dict")
                        raise TypeError("ChunkCore expected Chunk objects; got dict.")
                    meta = getattr(ch, "meta", None) or {}
                    if not isinstance(meta, dict):
                        meta = {}
                    meta.setdefault("retrieval_route", "query")
                    meta.setdefault("retrieval_stage", "search")
                    meta.setdefault("retrieval_method", "vector")
                    ch.meta = meta
            except Exception:
                logger.exception("ChunkCore._tag_retrieval_metadata: failed to attach retrieval metadata")
                raise
            try:
                confs = [
                    float(ch.meta["lexical_confidence"])
                    for ch in chunks
                    if isinstance(getattr(ch, "meta", None), dict)
                    and ch.meta.get("retrieval_method") == "lexical"
                    and ch.meta.get("lexical_confidence") is not None
                ]
                if confs:
                    logger.info(
                        "ChunkCore.search_chunks lexical_confidence n=%d avg=%.2f max=%.2f",
                        len(confs), sum(confs) / max(1, len(confs)), max(confs),
                    )
            except Exception:
                logger.exception("ChunkCore._tag_retrieval_metadata: failed to summarize lexical_confidence")
                raise

    @staticmethod
    def _apply_term_filter(chunks: list[Chunk], query_text: str) -> list[Chunk]:
        term_set = build_query_term_set(query_text)
        if not term_set or (not term_set.terms and not term_set.phrases):
            return chunks
        filtered: list[Chunk] = []
        for ch in chunks:
            if isinstance(ch, dict):
                logger.error("ChunkCore._apply_term_filter: expected Chunk objects; got dict")
                raise TypeError("ChunkCore expected Chunk objects; got dict.")
            if getattr(ch, "text", "") and text_matches_query_terms(ch.text, term_set, min_term_matches=2, max_terms_for_match=6):
                filtered.append(ch)
        return filtered

    @staticmethod
    def _shortlist_for_neighbor_expansion(
        chunks: list[Chunk],
        *,
        shortlist_k: Optional[int],
        shortlist_max_per_doc: Optional[int],
    ) -> list[Chunk]:
        """
        Select a deterministic subset of chunks to serve as neighbor-expansion anchors.

        `chunks` is assumed to already be ranked (canonical Ranker ordering applied).
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

        out: list[Chunk] = []
        per_doc_counts: dict[str, int] = {}
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
        tenant_id: str = DEFAULT_TENANT_ID,
        owner_type: str,
        owner_id: str,
        anchors: list[Chunk],
        window: int = 1,
        max_total: int = 24,
    ) -> list[Chunk]:
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
        ranges_by_doc: dict[str, list[tuple[int, int]]] = {}
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

        def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
            if not ranges:
                return []
            rs = sorted(ranges, key=lambda x: (x[0], x[1]))
            merged: list[tuple[int, int]] = []
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
        fetched_by_doc: dict[str, list[Chunk]] = {}
        for doc_id, ranges in merged_ranges_by_doc.items():
            fetched: list[Chunk] = []
            for s, e in ranges:
                try:
                    rows = await self.store.fetch_by_doc_and_position_range(
                        tenant_id=tenant_id,
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

        by_doc_pos: dict[str, dict[int, Chunk]] = {}
        for doc_id, rows in fetched_by_doc.items():
            pos_map: dict[int, Chunk] = {}
            for ch in rows:
                try:
                    pos = int(getattr(ch, "position", 0) or 0)
                except Exception:
                    pos = 0
                if pos and getattr(ch, "id", None) and pos not in pos_map:
                    pos_map[pos] = ch
            by_doc_pos[doc_id] = pos_map

        out: list[Chunk] = []
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
