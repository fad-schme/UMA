"""
ChunkSQLStore — SQL + VectorIndex for authoritative document chunks.

Stores raw chunk text and metadata, and maintains embeddings in the vector index.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_vector_sql_store import BaseVectorSQLStore
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from ..core.utils.store_metadata import ensure_store_metadata
from ..types import Chunk

logger = logging.getLogger(__name__)
_DEBUG_LOGGED_PARSE_FAILURES: set[tuple[str, str]] = set()


def _debug_once(mode: str, chunk_id: str, message: str) -> None:
    key = (mode, chunk_id)
    if key in _DEBUG_LOGGED_PARSE_FAILURES:
        return
    _DEBUG_LOGGED_PARSE_FAILURES.add(key)
    logger.debug(message, chunk_id)


class ChunkSQLStore(BaseVectorSQLStore):
    def __init__(self, db_adapter: DBAdapter, vector_index: VectorIndex) -> None:
        super().__init__(db_adapter=db_adapter, vector_index=vector_index)
        self._init_db()
        logger.debug("ChunkSQLStore initialized.")

    # ------------------------------------------------------------------ #
    # SQL Schema
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        conn = self._conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    source_path TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    meta TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_owner ON chunks(owner_type, owner_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_scope_doc_pos ON chunks(owner_type, owner_id, doc_id, position);
                """
            )
            ensure_store_metadata(self, conn, store_name="chunks")
            conn.commit()
        except Exception:
            self._safe_rollback(conn, "init_db")
            logger.exception("ChunkSQLStore: failed initializing schema.")
            raise
        finally:
            conn.close()

    @property
    def _table_name(self) -> str:
        return "chunks"

    @property
    def _id_column(self) -> str:
        return "id"

    # ------------------------------------------------------------------ #
    # Row → Chunk
    # ------------------------------------------------------------------ #

    def _row_to_object(self, row: Any) -> Chunk:
        if hasattr(row, "get"):
            meta_val = row.get("meta")
        else:
            keys = list(row.keys()) if hasattr(row, "keys") else []
            meta_val = row["meta"] if "meta" in keys else None
        chunk_id = row.get("id") if hasattr(row, "get") else row["id"]

        try:
            meta = json.loads(meta_val) if meta_val else {}
        except Exception:
            meta = {}
            _debug_once("meta_json", str(chunk_id), "ChunkSQLStore: failed to parse meta JSON id=%s")

        # Preserve lexical confidence score when present (e.g., computed in search_text CTE).
        try:
            if hasattr(row, "get") and row.get("score") is not None:
                meta["lexical_confidence"] = float(row.get("score"))  # type: ignore[arg-type]
            elif hasattr(row, "keys") and "score" in row.keys() and row["score"] is not None:
                meta["lexical_confidence"] = float(row["score"])
        except Exception:
            _debug_once("lexical_score", str(chunk_id), "ChunkSQLStore: failed to parse lexical score id=%s")

        return Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            text=row["text"],
            page_range=(int(row["page_start"]), int(row["page_end"])),
            position=int(row["position"]),
            source_path=row["source_path"],
            source_hash=row["source_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            owner_type=row["owner_type"],
            owner_id=row["owner_id"],
            meta=meta,
        )

    # ------------------------------------------------------------------ #
    # Insert/Upsert
    # ------------------------------------------------------------------ #

    def _owner_where(self, owner_type: Optional[str], owner_id: Optional[str], params: List[Any]) -> str:
        if not owner_type or not owner_id:
            logger.error("ChunkSQLStore requires owner_type and owner_id")
            raise ValueError("ChunkSQLStore requires owner_type and owner_id")
        params.extend([owner_type, owner_id])
        return "owner_type=? AND owner_id=?"

    async def upsert_chunk(self, chunk: Chunk, embedding: List[float]) -> None:
        conn = self._conn()
        try:
            payload = {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "page_start": chunk.page_range[0],
                "page_end": chunk.page_range[1],
                "position": chunk.position,
                "source_path": chunk.source_path,
                "source_hash": chunk.source_hash,
                "created_at": chunk.created_at.isoformat(),
                "updated_at": chunk.updated_at.isoformat(),
                "owner_type": chunk.owner_type,
                "owner_id": chunk.owner_id,
                "meta": json.dumps(chunk.meta or {}),
            }

            self._execute(
                conn,
                """
                INSERT INTO chunks (
                    id, doc_id, text, page_start, page_end, position,
                    source_path, source_hash, created_at, updated_at,
                    owner_type, owner_id, meta
                ) VALUES (
                    :id, :doc_id, :text, :page_start, :page_end, :position,
                    :source_path, :source_hash, :created_at, :updated_at,
                    :owner_type, :owner_id, :meta
                )
                ON CONFLICT(id) DO UPDATE SET
                    doc_id=excluded.doc_id,
                    text=excluded.text,
                    page_start=excluded.page_start,
                    page_end=excluded.page_end,
                    position=excluded.position,
                    source_path=excluded.source_path,
                    source_hash=excluded.source_hash,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    owner_type=excluded.owner_type,
                    owner_id=excluded.owner_id,
                    meta=excluded.meta
                """,
                params=payload,
                log_context="chunk_upsert",
            )

            # Vector index upsert (projection)
            try:
                vector_meta = {
                    "doc_id": chunk.doc_id,
                    "owner_type": chunk.owner_type,
                    "owner_id": chunk.owner_id,
                    "scope_key": f"{chunk.owner_type}:{chunk.owner_id}",
                }
                self.vector_index.upsert(
                    ids=[chunk.id],
                    vectors=[embedding],
                    metadata=[vector_meta],
                )
            except Exception:
                logger.exception("ChunkSQLStore: vector upsert failed for id=%s", chunk.id)
                self._safe_rollback(conn, "chunk_upsert")
                raise

            try:
                conn.commit()
            except Exception:
                self._safe_rollback(conn, "chunk_upsert_commit")
                try:
                    self.vector_index.delete([chunk.id])
                except Exception:
                    logger.exception(
                        "ChunkSQLStore: vector delete failed after commit error id=%s",
                        chunk.id,
                    )
                raise

        except Exception:
            logger.exception("ChunkSQLStore.upsert_chunk failed for id=%s", chunk.id)
            raise
        finally:
            conn.close()

    async def search(
        self,
        query_embedding: List[float],
        *,
        doc_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 10,
    ) -> List[Chunk]:
        if not owner_type or not owner_id:
            logger.error("ChunkSQLStore.search requires owner_type and owner_id")
            raise ValueError("ChunkSQLStore.search requires owner_type and owner_id")
        filters: Dict[str, Any] = {}
        if doc_id:
            filters["doc_id"] = doc_id
        filters["owner_type"] = owner_type
        filters["owner_id"] = owner_id
        try:
            ids = await self._vector_search_ids(
                query_embedding=query_embedding,
                k=k,
                filters=filters,
                log_context="chunk_search",
                id_prefix="chunk_",
            )
            if not ids:
                logger.debug(
                    "ChunkSQLStore.search: vector candidates=0, sql_fetched=0, owner=%s:%s",
                    owner_type,
                    owner_id,
                )
                return []
            chunks = await self.fetch_by_ids(ids, owner_type=owner_type, owner_id=owner_id)
            logger.debug(
                "ChunkSQLStore.search: vector candidates=%d, sql_fetched=%d, owner=%s:%s",
                len(ids),
                len(chunks),
                owner_type,
                owner_id,
            )
            if ids and not chunks:
                logger.warning(
                    "ChunkSQLStore.search: vector candidates=%d but SQL returned 0 op=search owner=%s:%s ids=%s",
                    len(ids),
                    owner_type,
                    owner_id,
                    ids[:3],
                )
            return chunks
        except Exception:
            logger.exception("ChunkSQLStore.search failed.")
            raise

    async def search_text(
        self,
        query_text: str,
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 10,
    ) -> List[Chunk]:
        """
        Lightweight lexical fallback for chunk retrieval.
        Uses SQL LIKE against chunk text (case-insensitive).
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("ChunkSQLStore.search_text STARTED")

        if not query_text or not isinstance(query_text, str):
            return []
        if not owner_type or not owner_id:
            logger.error("ChunkSQLStore.search_text requires owner_type and owner_id")
            raise ValueError("ChunkSQLStore.search_text requires owner_type and owner_id")

        from ..core.utils.user_query_helper import build_query_term_set

        def _escape_like(term: str) -> str:
            return (term or "").replace("%", "\\%").replace("_", "\\_")

        # Consistent with user_query_helper: use the extracted keywords + phrases when available.
        term_set = build_query_term_set(query_text, max_terms=12, max_phrases=12)
        terms = list(term_set.terms) if term_set else []
        phrases = list(term_set.phrases) if term_set else []
        # No secondary normalization here: build_query_term_set already uses the canonical
        # extractor (extract_keywords_and_phrases) which normalizes internally.

        terms = [t.strip() for t in (terms or []) if t and t.strip()]
        phrases = [p.strip() for p in (phrases or []) if p and p.strip()]
        if not terms and not phrases:
            return []

        # Cap term count to keep SQL small and deterministic.
        terms = list(dict.fromkeys(terms))[:12]
        phrases = list(dict.fromkeys(phrases))[:12]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ChunkSQLStore.search_text terms=%r phrases=%r phrase_primacy=%s",
                terms,
                phrases,
                bool(phrases),
            )

        where = []
        params: dict[str, Any] = {}
        if owner_type:
            where.append("owner_type = :owner_type")
            params["owner_type"] = owner_type
        if owner_id:
            where.append("owner_id = :owner_id")
            params["owner_id"] = owner_id

        # Weighted SQL scoring, using extracted phrases/keywords.
        # Phrase weight > keyword weight to favor coherent multi-word matches.
        # Tuning knobs (keep these as simple constants for now).
        # Baseline values (pre-tuning) were: phrase_weight=5.0, keyword_weight=1.0.
        phrase_weight = 5.0
        keyword_weight = 1.0
        # Pre-tuning threshold behavior: allow keyword-only queries to pass with a lower score.
        min_score = 3.0 if phrases else 2.0
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ChunkSQLStore.search_text weights phrase_weight=%.2f keyword_weight=%.2f min_score=%.2f",
                phrase_weight,
                keyword_weight,
                min_score,
            )

        phrase_terms: List[str] = []
        for i, phrase in enumerate(phrases):
            key = f"p{i}"
            phrase_terms.append(f"CASE WHEN LOWER(text) LIKE :{key} THEN :phrase_weight ELSE 0.0 END")
            params[key] = f"%{_escape_like(phrase.lower())}%"

        keyword_terms: List[str] = []
        for i, term in enumerate(terms):
            key = f"t{i}"
            keyword_terms.append(f"CASE WHEN LOWER(text) LIKE :{key} THEN :keyword_weight ELSE 0.0 END")
            params[key] = f"%{_escape_like(term.lower())}%"

        phrase_expr = " + ".join(phrase_terms) if phrase_terms else "0.0"
        keyword_expr = " + ".join(keyword_terms) if keyword_terms else "0.0"

        # Pre-primacy scoring: always add phrase + keyword scores.
        # Never allow lexical fallback to devolve to an unscoped full-table scan.
        where_sql = " AND ".join(where) if where else "0=1"
        sql = f"""
            WITH scored AS (
                SELECT *,
                       ({phrase_expr}) AS phrase_score,
                       ({keyword_expr}) AS keyword_score
                FROM chunks
                WHERE {where_sql}
                AND LENGTH(text) >= :min_len
            )
            SELECT *,
                   (phrase_score + keyword_score) AS score
            FROM scored
            WHERE score >= :min_score
            ORDER BY score DESC, position ASC
            LIMIT :limit
        """
        params["limit"] = int(k)
        params["min_score"] = float(min_score)
        params["min_len"] = 80
        params["phrase_weight"] = float(phrase_weight)
        params["keyword_weight"] = float(keyword_weight)
        # min_score already accounts for presence/absence of phrases (baseline behavior).

        conn = self._conn()
        try:
            rows = self._query_all(conn, sql, params=params, log_context="chunk_search_text")
            if logger.isEnabledFor(logging.INFO):
                avg_score = 0.0
                try:
                    scores = [float(r.get("score") or 0.0) for r in (rows or []) if hasattr(r, "get")]
                    avg_score = (sum(scores) / len(scores)) if scores else 0.0
                except Exception:
                    avg_score = 0.0
                logger.info(
                    "ChunkSQLStore.search_text query_len=%d terms=%d phrases=%d returned=%d avg_score=%.2f top_terms=%r top_phrases=%r",
                    len(query_text),
                    len(terms),
                    len(phrases),
                    len(rows or []),
                    avg_score,
                    terms[:3],
                    phrases[:2],
                )
            return [self._row_to_object(r) for r in rows]
        except Exception:
            logger.exception("ChunkSQLStore.search_text failed.")
            raise
        finally:
            conn.close()

    async def fetch_by_doc_and_position_range(
        self,
        *,
        owner_type: str,
        owner_id: str,
        doc_id: str,
        pos_start: int,
        pos_end: int,
        log_context: str = "chunk_fetch_by_doc_pos_range",
    ) -> List[Chunk]:
        if not doc_id or not isinstance(doc_id, str):
            return []
        try:
            pos_start_i = int(pos_start)
            pos_end_i = int(pos_end)
        except Exception:
            return []
        if pos_end_i < pos_start_i:
            return []

        conn = self._conn()
        try:
            rows = self._query_all(
                conn,
                """
                SELECT *
                FROM chunks
                WHERE owner_type = ?
                  AND owner_id = ?
                  AND doc_id = ?
                  AND position BETWEEN ? AND ?
                ORDER BY position ASC
                """,
                params=[owner_type, owner_id, doc_id, pos_start_i, pos_end_i],
                log_context=log_context,
            )
            return [self._row_to_object(r) for r in (rows or [])]
        except Exception:
            logger.exception(
                "ChunkSQLStore.fetch_by_doc_and_position_range failed owner=%s:%s doc_id=%s",
                owner_type,
                owner_id,
                doc_id,
            )
            raise
        finally:
            conn.close()

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        owner_type: str,
        owner_id: str,
        log_context: str = "",
    ) -> List[Chunk]:
        """
        Fetch Chunk objects by ID, owner-scoped.
        """
        if not ids:
            return []
        if not owner_type or not owner_id:
            logger.error("ChunkSQLStore.fetch_by_ids requires owner_type and owner_id")
            raise ValueError("ChunkSQLStore.fetch_by_ids requires owner_type and owner_id")

        logger.debug(
            "ChunkSQLStore.fetch_by_ids: ids=%d owner=%s:%s",
            len(ids),
            owner_type,
            owner_id,
        )
        conn = self._conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            params: List[Any] = list(ids)
            owner_clause = self._owner_where(owner_type, owner_id, params)
            sql = f"SELECT * FROM chunks WHERE id IN ({placeholders}) AND {owner_clause}"
            rows = self._query_all(conn, sql, params=params, log_context="fetch_by_ids")
            row_map = {r["id"]: r for r in rows}
            ordered: List[Chunk] = []
            for cid in ids:
                row = row_map.get(cid)
                if row is None:
                    continue
                ordered.append(self._row_to_object(row))
            missing = max(0, len(ids) - len(ordered))
            if missing:
                logger.warning(
                    "ChunkSQLStore.fetch_by_ids: missing=%d owner=%s:%s",
                    missing,
                    owner_type,
                    owner_id,
                )
            return ordered
        except Exception:
            logger.exception("ChunkSQLStore.fetch_by_ids failed.")
            raise
        finally:
            conn.close()
