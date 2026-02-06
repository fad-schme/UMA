"""
ChunkSQLStore — SQL + VectorIndex for authoritative document chunks.

Stores raw chunk text and metadata, and maintains embeddings in the vector index.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, List, Optional

from .base_vector_sql_store import BaseVectorSQLStore
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from ..core.utils.store_metadata import ensure_store_metadata
from ..types_chunk import Chunk

logger = logging.getLogger(__name__)


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
                """
            )
            # Optional SQLite FTS5 index for lexical retrieval.
            # If the runtime doesn't support FTS5, we continue without it.
            try:
                conn.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text, content='chunks', content_rowid='rowid');

                    CREATE TRIGGER IF NOT EXISTS chunks_ai
                    AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                    END;

                    CREATE TRIGGER IF NOT EXISTS chunks_ad
                    AFTER DELETE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
                    END;

                    CREATE TRIGGER IF NOT EXISTS chunks_au
                    AFTER UPDATE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
                        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                    END;
                    """
                )
            except Exception:
                logger.warning("ChunkSQLStore: FTS5 unavailable; lexical search falls back to LIKE.")
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

        try:
            meta = json.loads(meta_val) if meta_val else {}
        except Exception:
            meta = {}

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
        filters = {}
        if doc_id:
            filters["doc_id"] = doc_id
        if owner_type:
            filters["owner_type"] = owner_type
        if owner_id:
            filters["owner_id"] = owner_id
        return await self._semantic_search(
            query_embedding=query_embedding,
            k=k,
            filters=filters or None,
            log_context="chunk_search",
            id_prefix="chunk_",
        )

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
        if not query_text or not isinstance(query_text, str):
            return []

        try:
            from ..core.utils.user_query_helper import extract_query_terms, expand_query_terms
        except Exception:
            extract_query_terms = None
            expand_query_terms = None

        if expand_query_terms:
            terms = expand_query_terms(query_text)
        elif extract_query_terms:
            terms = extract_query_terms(query_text)
        else:
            terms = []
        terms = [t.strip() for t in (terms or []) if t and t.strip()]
        if not terms:
            terms = [query_text.strip()]

        # Cap term count to keep SQL small.
        terms = list(dict.fromkeys(terms))[:6]

        where = []
        params: dict[str, Any] = {}
        if owner_type:
            where.append("owner_type = :owner_type")
            params["owner_type"] = owner_type
        if owner_id:
            where.append("owner_id = :owner_id")
            params["owner_id"] = owner_id

        like_clauses = []
        for i, term in enumerate(terms):
            key = f"t{i}"
            like_clauses.append(f"LOWER(text) LIKE :{key}")
            params[key] = f"%{term.lower()}%"

        if like_clauses:
            where.append("(" + " OR ".join(like_clauses) + ")")

        where_sql = " AND ".join(where) if where else "1=1"
        sql = f"""
            SELECT * FROM chunks
            WHERE {where_sql}
            ORDER BY position ASC
            LIMIT :limit
        """
        params["limit"] = int(k)

        conn = self._conn()
        try:
            rows = self._query_all(conn, sql, params=params, log_context="chunk_search_text")
            return [self._row_to_object(r) for r in rows]
        except Exception:
            logger.exception("ChunkSQLStore.search_text failed.")
            return []
        finally:
            conn.close()

    async def search_fts5(
        self,
        query_text: str,
        *,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 10,
        required_terms: Optional[List[str]] = None,
        optional_terms: Optional[List[str]] = None,
        bm25_weight: float = 1.0,
    ) -> List[Chunk]:
        """
        SQLite FTS5 lexical search with bm25 ranking.

        - required_terms are enforced strictly (AND).
        - optional_terms are included softly (OR), improving recall.
        """
        if not query_text or not isinstance(query_text, str):
            return []

        terms: List[str] = []
        if required_terms:
            terms.extend([t for t in required_terms if isinstance(t, str) and t.strip()])
        if not terms:
            terms = [query_text.strip()]

        soft_terms = [t for t in (optional_terms or []) if isinstance(t, str) and t.strip()]

        # Build FTS query: required terms AND (optional terms OR ...)
        required_clause = " AND ".join(f'"{t}"' for t in terms)
        if soft_terms:
            optional_clause = " OR ".join(f'"{t}"' for t in soft_terms)
            fts_query = f"({required_clause}) AND ({optional_clause})"
        else:
            fts_query = required_clause

        where = ["chunks_fts MATCH :q"]
        params: dict[str, Any] = {"q": fts_query, "limit": int(k), "w": float(bm25_weight)}
        if owner_type:
            where.append("c.owner_type = :owner_type")
            params["owner_type"] = owner_type
        if owner_id:
            where.append("c.owner_id = :owner_id")
            params["owner_id"] = owner_id

        where_sql = " AND ".join(where)
        sql = f"""
            SELECT c.*
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE {where_sql}
            ORDER BY bm25(chunks_fts, :w) ASC
            LIMIT :limit
        """

        conn = self._conn()
        try:
            rows = self._query_all(conn, sql, params=params, log_context="chunk_search_fts5")
            return [self._row_to_object(r) for r in rows]
        except Exception:
            logger.exception("ChunkSQLStore.search_fts5 failed.")
            return []
        finally:
            conn.close()
