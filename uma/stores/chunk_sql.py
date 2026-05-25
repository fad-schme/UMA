"""
ChunkSQLStore — SQL + VectorIndex for authoritative document chunks.

Stores raw chunk text and metadata, and maintains embeddings in the vector index.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base_vector_sql_store import BaseVectorSQLStore
from .base_sql_store import DEFAULT_TENANT_ID
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from uma.stores.metadata import ensure_store_metadata
from uma.common.types import Chunk, SCOPE_MODEL_VERSION
from uma.common.storage_metadata import normalize_chunk_metadata

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
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT,
                    origin_agent_id TEXT,
                    origin_user_id TEXT,
                    origin_session_id TEXT,
                    scope_model_version TEXT,
                    meta TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_owner ON chunks(owner_type, owner_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_scope_doc_pos ON chunks(owner_type, owner_id, doc_id, position);
                """
            )
            self._ensure_column(conn, "chunks", "tenant_id", "TEXT NOT NULL DEFAULT 'default'")
            self._ensure_column(conn, "chunks", "workspace_id", "TEXT")
            self._ensure_column(conn, "chunks", "origin_agent_id", "TEXT")
            self._ensure_column(conn, "chunks", "origin_user_id", "TEXT")
            self._ensure_column(conn, "chunks", "origin_session_id", "TEXT")
            self._ensure_column(conn, "chunks", "scope_model_version", "TEXT")
            self._ensure_column(conn, "chunks", "trust_score", "REAL NOT NULL DEFAULT 0.5")
            self._ensure_column(conn, "chunks", "quarantined_at", "DATETIME")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_tenant_owner ON chunks(tenant_id, owner_type, owner_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_tenant_scope_doc_pos ON chunks(tenant_id, owner_type, owner_id, doc_id, position);"
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

        # Preserve lexical confidence score when present (e.g., computed in lexical_search CTE).
        try:
            if hasattr(row, "get") and row.get("score") is not None:
                meta["lexical_confidence"] = float(row.get("score"))  # type: ignore[arg-type]
            elif hasattr(row, "keys") and "score" in row.keys() and row["score"] is not None:
                meta["lexical_confidence"] = float(row["score"])
        except Exception:
            _debug_once("lexical_score", str(chunk_id), "ChunkSQLStore: failed to parse lexical score id=%s")

        normalized_meta = normalize_chunk_metadata(
            meta,
            chunk_id=row["id"],
            doc_id=row["doc_id"],
            owner_type=row["owner_type"],
            owner_id=row["owner_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            page_range=(int(row["page_start"]), int(row["page_end"])),
            position=int(row["position"]),
            source_path=row["source_path"],
            source_hash=row["source_hash"],
        )

        row_keys = row.keys() if hasattr(row, "keys") else []
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
            tenant_id=(row["tenant_id"] if "tenant_id" in row_keys else DEFAULT_TENANT_ID),
            owner_type=row["owner_type"],
            owner_id=row["owner_id"],
            workspace_id=(row["workspace_id"] if "workspace_id" in row_keys else None),
            origin_agent_id=(row["origin_agent_id"] if "origin_agent_id" in row_keys else None),
            origin_user_id=(row["origin_user_id"] if "origin_user_id" in row_keys else None),
            origin_session_id=(row["origin_session_id"] if "origin_session_id" in row_keys else None),
            scope_model_version=(row["scope_model_version"] if "scope_model_version" in row_keys else None),
            trust_score=(float(row["trust_score"]) if "trust_score" in row_keys and row["trust_score"] is not None else 0.5),
            quarantined_at=(
                datetime.fromisoformat(row["quarantined_at"])
                if "quarantined_at" in row_keys and row["quarantined_at"] is not None
                else None
            ),
            meta=normalized_meta,
        )

    # ------------------------------------------------------------------ #
    # Insert/Upsert
    # ------------------------------------------------------------------ #

    def _scope_where(
        self,
        tenant_id: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
        params: List[Any],
    ) -> str:
        if tenant_id:
            params.append(tenant_id)
        else:
            logger.error("ChunkSQLStore requires tenant_id")
            raise ValueError("ChunkSQLStore requires tenant_id")
        if not owner_type or not owner_id:
            logger.error("ChunkSQLStore requires owner_type and owner_id")
            raise ValueError("ChunkSQLStore requires owner_type and owner_id")
        params.extend([owner_type, owner_id])
        return "tenant_id=? AND owner_type=? AND owner_id=?"

    async def upsert_chunk(self, chunk: Chunk, embedding: List[float]) -> None:
        conn = self._conn()
        try:
            normalized_meta = normalize_chunk_metadata(
                chunk.meta,
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                owner_type=chunk.owner_type,
                owner_id=chunk.owner_id,
                created_at=chunk.created_at,
                updated_at=chunk.updated_at,
                page_range=chunk.page_range,
                position=chunk.position,
                source_path=chunk.source_path,
                source_hash=chunk.source_hash,
            )
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
                "tenant_id": getattr(chunk, "tenant_id", None) or DEFAULT_TENANT_ID,
                "owner_type": chunk.owner_type,
                "owner_id": chunk.owner_id,
                "workspace_id": getattr(chunk, "workspace_id", None),
                "origin_agent_id": getattr(chunk, "origin_agent_id", None),
                "origin_user_id": getattr(chunk, "origin_user_id", None),
                "origin_session_id": getattr(chunk, "origin_session_id", None),
                "scope_model_version": getattr(chunk, "scope_model_version", None) or SCOPE_MODEL_VERSION,
                "trust_score": float(_ts if (_ts := getattr(chunk, "trust_score", None)) is not None else 0.5),
                "quarantined_at": (
                    getattr(chunk, "quarantined_at").isoformat()
                    if getattr(chunk, "quarantined_at", None) is not None
                    else None
                ),
                "meta": json.dumps(normalized_meta),
            }

            self._execute(
                conn,
                """
                INSERT INTO chunks (
                    id, doc_id, text, page_start, page_end, position,
                    source_path, source_hash, created_at, updated_at,
                    tenant_id, owner_type, owner_id, workspace_id,
                    origin_agent_id, origin_user_id, origin_session_id,
                    scope_model_version, trust_score, quarantined_at, meta
                ) VALUES (
                    :id, :doc_id, :text, :page_start, :page_end, :position,
                    :source_path, :source_hash, :created_at, :updated_at,
                    :tenant_id, :owner_type, :owner_id, :workspace_id,
                    :origin_agent_id, :origin_user_id, :origin_session_id,
                    :scope_model_version, :trust_score, :quarantined_at, :meta
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
                    tenant_id=excluded.tenant_id,
                    owner_type=excluded.owner_type,
                    owner_id=excluded.owner_id,
                    workspace_id=excluded.workspace_id,
                    origin_agent_id=excluded.origin_agent_id,
                    origin_user_id=excluded.origin_user_id,
                    origin_session_id=excluded.origin_session_id,
                    scope_model_version=excluded.scope_model_version,
                    trust_score=excluded.trust_score,
                    quarantined_at=excluded.quarantined_at,
                    meta=excluded.meta
                """,
                params=payload,
                log_context="chunk_upsert",
            )

            # Vector index upsert (projection)
            try:
                resolved_tenant = getattr(chunk, "tenant_id", None) or DEFAULT_TENANT_ID
                # C1: isolation fields go as explicit parameters; everything
                # else lives in extra_metadata. The vector index promotes
                # tenant_id/owner_type/owner_id into first-class indexable
                # columns so isolation is enforced by construction.
                extra_meta = {
                    "doc_id": chunk.doc_id,
                    "kb_lane": normalized_meta.get("kb_lane"),
                    "position": int(chunk.position),
                    "page_start": int(chunk.page_range[0]),
                    "page_end": int(chunk.page_range[1]),
                    "scope_key": f"{chunk.owner_type}:{chunk.owner_id}",
                }
                self.vector_index.upsert(
                    ids=[chunk.id],
                    vectors=[embedding],
                    tenant_ids=[resolved_tenant],
                    owner_types=[chunk.owner_type],
                    owner_ids=[chunk.owner_id],
                    extra_metadata=[extra_meta],
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
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 10,
    ) -> List[Chunk]:
        if not tenant_id:
            logger.error("ChunkSQLStore.search requires tenant_id")
            raise ValueError("ChunkSQLStore.search requires tenant_id")
        if not owner_type or not owner_id:
            logger.error("ChunkSQLStore.search requires owner_type and owner_id")
            raise ValueError("ChunkSQLStore.search requires owner_type and owner_id")
        # C1: doc_id (when set) is a non-isolation filter — goes through
        # extra_filters. The three isolation keys go as explicit
        # parameters so the vector index pushes them into the backend.
        extra_filters: Dict[str, Any] = {}
        if doc_id:
            extra_filters["doc_id"] = doc_id
        try:
            id_score_pairs = await self._vector_search_ids(
                query_embedding=query_embedding,
                k=k,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                extra_filters=extra_filters or None,
                log_context="chunk_search",
            )
            if not id_score_pairs:
                logger.debug(
                    "ChunkSQLStore.search: vector candidates=0, sql_fetched=0, owner=%s:%s",
                    owner_type,
                    owner_id,
                )
                return []
            ids = [sid for sid, _score in id_score_pairs]
            chunks = await self.fetch_by_ids(
                ids,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
            self._attach_vector_scores(chunks, id_score_pairs)
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

    async def lexical_search(
        self,
        query_text: str,
        *,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 10,
    ) -> List[Chunk]:
        """
        Lightweight lexical fallback for chunk retrieval.
        Uses SQL LIKE against chunk text (case-insensitive).
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("ChunkSQLStore.lexical_search STARTED")

        if not query_text or not isinstance(query_text, str):
            return []
        if not tenant_id:
            logger.error("ChunkSQLStore.lexical_search requires tenant_id")
            raise ValueError("ChunkSQLStore.lexical_search requires tenant_id")
        if not owner_type or not owner_id:
            logger.error("ChunkSQLStore.lexical_search requires owner_type and owner_id")
            raise ValueError("ChunkSQLStore.lexical_search requires owner_type and owner_id")

        from uma.retrieve.user_query_helper import build_query_term_set

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
                "ChunkSQLStore.lexical_search terms=%r phrases=%r phrase_primacy=%s",
                terms,
                phrases,
                bool(phrases),
            )

        where = []
        params: dict[str, Any] = {}
        if tenant_id:
            where.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
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
                "ChunkSQLStore.lexical_search weights phrase_weight=%.2f keyword_weight=%.2f min_score=%.2f",
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
                AND quarantined_at IS NULL
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
            rows = self._query_all(conn, sql, params=params, log_context="chunk_lexical_search")
            if logger.isEnabledFor(logging.INFO):
                avg_score = 0.0
                try:
                    scores = [float(r.get("score") or 0.0) for r in (rows or []) if hasattr(r, "get")]
                    avg_score = (sum(scores) / len(scores)) if scores else 0.0
                except Exception:
                    avg_score = 0.0
                logger.info(
                    "ChunkSQLStore.lexical_search query_len=%d terms=%d phrases=%d returned=%d avg_score=%.2f top_terms=%r top_phrases=%r",
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
            logger.exception("ChunkSQLStore.lexical_search failed.")
            raise
        finally:
            conn.close()

    async def fetch_by_doc_and_position_range(
        self,
        *,
        tenant_id: Optional[str] = None,
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
        if not tenant_id:
            logger.error("ChunkSQLStore.fetch_by_doc_and_position_range requires tenant_id")
            raise ValueError("ChunkSQLStore.fetch_by_doc_and_position_range requires tenant_id")

        conn = self._conn()
        try:
            rows = self._query_all(
                conn,
                """
                SELECT *
                FROM chunks
                WHERE tenant_id = ?
                  AND owner_type = ?
                  AND owner_id = ?
                  AND doc_id = ?
                  AND position BETWEEN ? AND ?
                  AND quarantined_at IS NULL
                ORDER BY position ASC
                """,
                params=[tenant_id, owner_type, owner_id, doc_id, pos_start_i, pos_end_i],
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
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
        log_context: str = "",
    ) -> List[Chunk]:
        """
        Fetch Chunk objects by ID, owner-scoped.
        """
        if not ids:
            return []
        if not tenant_id or not owner_type or not owner_id:
            logger.error("ChunkSQLStore.fetch_by_ids requires tenant_id, owner_type and owner_id")
            raise ValueError("ChunkSQLStore.fetch_by_ids requires tenant_id, owner_type and owner_id")

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
            scope_clause = self._scope_where(tenant_id, owner_type, owner_id, params)
            sql = f"SELECT * FROM chunks WHERE id IN ({placeholders}) AND {scope_clause} AND quarantined_at IS NULL"
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

    async def list_chunks_for_owner(
        self,
        *,
        tenant_id: Optional[str],
        owner_type: str,
        owner_id: str,
        include_quarantined: bool = False,
        limit: Optional[int] = None,
    ) -> List[Chunk]:
        """List chunks for an owner scope. Quarantined excluded by default."""
        if not tenant_id or not owner_type or not owner_id:
            raise ValueError("ChunkSQLStore.list_chunks_for_owner requires scope")
        conn = self._conn()
        try:
            quarantine_clause = "" if include_quarantined else " AND quarantined_at IS NULL"
            sql = f"SELECT * FROM chunks WHERE tenant_id=? AND owner_type=? AND owner_id=?{quarantine_clause} ORDER BY updated_at DESC"
            if limit:
                sql += f" LIMIT {int(limit)}"
            rows = self._query_all(conn, sql, params=[tenant_id, owner_type, owner_id], log_context="list_chunks_owner")
            return [self._row_to_object(r) for r in rows]
        except Exception:
            logger.exception("ChunkSQLStore.list_chunks_for_owner failed.")
            raise
        finally:
            conn.close()

    async def delete_chunk(
        self,
        chunk_id: str,
        *,
        tenant_id: Optional[str],
        owner_type: str,
        owner_id: str,
    ) -> None:
        """Delete a chunk from SQL + vector store."""
        if not tenant_id or not owner_type or not owner_id:
            raise ValueError("ChunkSQLStore.delete_chunk requires scope")
        conn = self._conn()
        try:
            self._execute(
                conn,
                "DELETE FROM chunks WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[chunk_id, tenant_id, owner_type, owner_id],
                log_context="delete_chunk",
            )
            conn.commit()
            try:
                self.vector_index.delete([chunk_id])
            except Exception:
                logger.exception("ChunkSQLStore.delete_chunk: vector delete failed id=%s", chunk_id)
            logger.info("ChunkSQLStore: deleted chunk id=%s owner=%s:%s", chunk_id, owner_type, owner_id)
        except Exception:
            self._safe_rollback(conn, "delete_chunk")
            logger.exception("ChunkSQLStore.delete_chunk failed id=%s", chunk_id)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Quarantine Management
    # ------------------------------------------------------------------ #

    async def reinstate_quarantined_record(
        self,
        record_id: str,
        *,
        tenant_id: Optional[str],
        owner_type: str,
        owner_id: str,
        audit_entry: Dict[str, Any],
    ) -> bool:
        """
        Clear quarantined_at and append an audit log entry to meta.security.audit_log.
        Returns True if a row was updated.
        """
        if not tenant_id or not owner_type or not owner_id:
            raise ValueError("ChunkSQLStore.reinstate_quarantined_record requires scope")
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT meta FROM chunks WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[record_id, tenant_id, owner_type, owner_id],
                log_context="reinstate_chunk_fetch",
            )
            if row is None:
                return False
            try:
                meta = json.loads(row["meta"]) if row["meta"] else {}
            except Exception:
                meta = {}
            security = meta.setdefault("security", {})
            audit_log = security.setdefault("audit_log", [])
            audit_log.append(audit_entry)
            self._execute(
                conn,
                "UPDATE chunks SET quarantined_at=NULL, meta=? WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[json.dumps(meta), record_id, tenant_id, owner_type, owner_id],
                log_context="reinstate_chunk",
            )
            conn.commit()
            logger.info("ChunkSQLStore: reinstated chunk id=%s owner=%s:%s", record_id, owner_type, owner_id)
            return True
        except Exception:
            self._safe_rollback(conn, "reinstate_chunk")
            logger.exception("ChunkSQLStore.reinstate_quarantined_record failed id=%s", record_id)
            raise
        finally:
            conn.close()

    async def get_chunk(
        self,
        chunk_id: str,
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
    ) -> Optional["Chunk"]:
        """Fetch a single chunk by ID (quarantined records included)."""
        if not tenant_id or not owner_type or not owner_id:
            raise ValueError("ChunkSQLStore.get_chunk requires tenant_id, owner_type and owner_id")
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT * FROM chunks WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[chunk_id, tenant_id, owner_type, owner_id],
                log_context="get_chunk",
            )
            return self._row_to_object(row) if row else None
        except Exception:
            logger.exception("ChunkSQLStore.get_chunk failed id=%s", chunk_id)
            raise
        finally:
            conn.close()

    async def quarantine_record(
        self,
        record_id: str,
        *,
        tenant_id: Optional[str],
        owner_type: str,
        owner_id: str,
        quarantined_at: str,
        audit_entry: Dict[str, Any],
    ) -> bool:
        """Set quarantined_at and append an audit log entry. Returns True if updated."""
        if not tenant_id or not owner_type or not owner_id:
            raise ValueError("ChunkSQLStore.quarantine_record requires scope")
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT meta FROM chunks WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[record_id, tenant_id, owner_type, owner_id],
                log_context="quarantine_chunk_fetch",
            )
            if row is None:
                return False
            try:
                meta = json.loads(row["meta"]) if row["meta"] else {}
            except Exception:
                meta = {}
            security = meta.setdefault("security", {})
            security.setdefault("audit_log", []).append(audit_entry)
            self._execute(
                conn,
                "UPDATE chunks SET quarantined_at=?, meta=? WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[quarantined_at, json.dumps(meta), record_id, tenant_id, owner_type, owner_id],
                log_context="quarantine_chunk",
            )
            conn.commit()
            logger.info("ChunkSQLStore: quarantined chunk id=%s owner=%s:%s", record_id, owner_type, owner_id)
            return True
        except Exception:
            self._safe_rollback(conn, "quarantine_chunk")
            logger.exception("ChunkSQLStore.quarantine_record failed id=%s", record_id)
            raise
        finally:
            conn.close()