"""
ChunkSQLStore — SQL + VectorIndex for authoritative document chunks.

Stores raw chunk text and metadata, and maintains embeddings in the vector index.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any, Optional

from .base_vector_sql_store import BaseVectorSQLStore

# Standard BM25 tuning constants (Robertson/Sparck-Jones defaults, the same
# values Lucene/Elasticsearch ship with) — term-frequency saturation point
# and document-length normalization strength, respectively.
_BM25_K1 = 1.5
_BM25_B = 0.75

# B608: _QUARANTINE_FILTER and _NO_FILTER are the only two values ever
# interpolated into the quarantine toggle position. Both are static SQL
# fragments containing no user data.
_QUARANTINE_FILTER = " AND quarantined_at IS NULL"
_NO_FILTER = ""

from ..common.types.types_scope import DEFAULT_TENANT_ID
from .base_sql_store import LIKE_ESCAPE_SQL, escape_like
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from uma.stores.metadata import ensure_store_metadata
from uma.common.text import build_query_term_set
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
        params: list[Any],
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

    async def upsert_chunk(self, chunk: Chunk, embedding: list[float]) -> None:
        """Insert or update a document chunk. Embeds the chunk and persists to SQL then vector store."""
        def _sync():
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


        return await self._run_sync(_sync)
    async def search(
        self,
        query_embedding: list[float],
        *,
        doc_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        k: int = 10,
    ) -> list[Chunk]:
        """Vector-search chunks within the ownership scope. Returns ranked ``Chunk`` objects."""
        if not tenant_id:
            logger.error("ChunkSQLStore.search requires tenant_id")
            raise ValueError("ChunkSQLStore.search requires tenant_id")
        if not owner_type or not owner_id:
            logger.error("ChunkSQLStore.search requires owner_type and owner_id")
            raise ValueError("ChunkSQLStore.search requires owner_type and owner_id")
        # C1: doc_id (when set) is a non-isolation filter — goes through
        # extra_filters. The three isolation keys go as explicit
        # parameters so the vector index pushes them into the backend.
        extra_filters: dict[str, Any] = {}
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
    ) -> list[Chunk]:
        """
        BM25-ranked lexical search over chunk text, scoped to tenant/owner.

        Two SQL passes (corpus stats, then a broad LIKE-matched candidate
        pool) feed a standard BM25 scorer (Robertson/Sparck-Jones IDF,
        k1=1.5, b=0.75) computed in Python — per-document term frequency
        isn't cheap to express via SQL LIKE the way document-level
        presence/absence is, so exact scoring happens after fetch.
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

        # Consistent with uma.common.text: use the extracted keywords + phrases when available.
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
                "ChunkSQLStore.lexical_search terms=%r phrases=%r",
                terms,
                phrases,
            )

        # Scope is mandatory and already validated above, so it is always the
        # full three-column predicate — never a partial one.
        where = ["tenant_id = :tenant_id", "owner_type = :owner_type", "owner_id = :owner_id"]
        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "owner_type": owner_type,
            "owner_id": owner_id,
        }

        # BM25 treats phrases and single-word terms as one flat set of query
        # tokens — a phrase's rarity (low document frequency) outweighs a
        # common single word via IDF on its own, so no hand-tuned
        # phrase_weight constant is needed the way the old LIKE-scoring
        # required.
        bm25_terms = list(dict.fromkeys([p.lower() for p in phrases] + [t.lower() for t in terms]))[:24]

        where_sql = " AND ".join(where)
        min_len = 80
        params["min_len"] = min_len

        # --- Pass 1: corpus stats in scope (doc count, avg length, per-term
        # document frequency). BM25's IDF is a property of the whole scope,
        # not the candidate pool, so this has to be a separate query.
        df_exprs: list[str] = []
        for i, term in enumerate(bm25_terms):
            key = f"bm{i}"
            df_exprs.append(f"COUNT(CASE WHEN LOWER(text) LIKE :{key}{LIKE_ESCAPE_SQL} THEN 1 END) AS df{i}")
            params[key] = f"%{escape_like(term)}%"
        df_select = (", " + ", ".join(df_exprs)) if df_exprs else ""

        # nosec B608 — df_exprs entries are built only from the fixed
        # template above (`i` is a Python loop index, not user data); every
        # LIKE value is bound as a named :bmN param, and where_sql is the
        # fixed three-element scope predicate with values bound as named
        # params. No user data becomes SQL structure.
        stats_sql = f"""
            SELECT COUNT(*) AS n_docs,
                   AVG(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) AS avg_len
                   {df_select}
            FROM chunks
            WHERE {where_sql}
            AND quarantined_at IS NULL
            AND LENGTH(text) >= :min_len
        """

        # --- Pass 2: broad candidate pool — any chunk matching at least one
        # term/phrase. This only needs to be a superset of the true top-k;
        # exact BM25 scoring happens in Python below, since term-frequency-
        # within-document isn't cheap to express via LIKE-based CASE
        # expressions the way document-level presence/absence is.
        match_exprs: list[str] = []
        for i, term in enumerate(bm25_terms):
            key = f"m{i}"
            match_exprs.append(f"LOWER(text) LIKE :{key}{LIKE_ESCAPE_SQL}")
            params[key] = f"%{escape_like(term)}%"
        match_sql = " OR ".join(match_exprs)
        candidate_limit = max(int(k) * 8, 100)
        params["candidate_limit"] = candidate_limit

        # nosec B608 — match_exprs entries and where_sql follow the same
        # fixed-template / bound-param structure as stats_sql above.
        candidates_sql = f"""
            SELECT * FROM chunks
            WHERE {where_sql}
            AND quarantined_at IS NULL
            AND LENGTH(text) >= :min_len
            AND ({match_sql})
            ORDER BY position ASC
            LIMIT :candidate_limit
        """

        def _sync():
            conn = self._conn()
            try:
                stats_rows = self._query_all(conn, stats_sql, params=params, log_context="chunk_lexical_stats")
                stats = stats_rows[0] if stats_rows else None
                n_docs = int(stats["n_docs"]) if stats and stats["n_docs"] is not None else 0
                if n_docs == 0:
                    return []
                avg_len = float(stats["avg_len"]) if stats["avg_len"] is not None else 1.0
                avg_len = avg_len or 1.0

                # Robertson/Sparck-Jones IDF, the "+1" variant used by
                # Lucene/Elasticsearch — always non-negative, unlike the
                # classic ln((N-df+0.5)/(df+0.5)) which goes negative once a
                # term appears in over half the scope.
                idf: dict[str, float] = {
                    term: math.log(1.0 + (n_docs - int(stats[f"df{i}"] or 0) + 0.5) / (int(stats[f"df{i}"] or 0) + 0.5))
                    for i, term in enumerate(bm25_terms)
                }

                rows = self._query_all(conn, candidates_sql, params=params, log_context="chunk_lexical_search")
                scored: list[tuple[float, dict[str, Any]]] = []
                for row in rows:
                    d = dict(row)
                    text_lower = (d.get("text") or "").lower()
                    # Word count via space-count proxy, matching the avg_len
                    # calculation in stats_sql above — approximate but
                    # consistent on both sides of the ratio.
                    doc_len = max(1, text_lower.count(" ") + 1)
                    score = 0.0
                    for term in bm25_terms:
                        # Substring occurrence count, not word-boundary
                        # matching — consistent with this store's existing
                        # LIKE-based substring semantics.
                        tf = text_lower.count(term)
                        if tf <= 0:
                            continue
                        denom = tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (doc_len / avg_len))
                        score += idf[term] * (tf * (_BM25_K1 + 1.0)) / denom
                    if score > 0.0:
                        d["score"] = score
                        scored.append((score, d))

                scored.sort(key=lambda item: (-item[0], item[1].get("position") or 0))
                top = scored[: int(k)]

                if logger.isEnabledFor(logging.INFO):
                    avg_score = (sum(s for s, _ in top) / len(top)) if top else 0.0
                    logger.info(
                        "ChunkSQLStore.lexical_search query_len=%d terms=%d candidates=%d returned=%d avg_score=%.2f",
                        len(query_text),
                        len(bm25_terms),
                        len(rows or []),
                        len(top),
                        avg_score,
                    )
                return [self._row_to_object(d) for _score, d in top]
            except Exception:
                logger.exception("ChunkSQLStore.lexical_search failed.")
                raise
            finally:
                conn.close()

        return await self._run_sync(_sync)
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
    ) -> list[Chunk]:
        """Fetch chunks belonging to a document within a position range, for neighbour-expansion."""
        if not doc_id or not isinstance(doc_id, str):
            return []
        try:
            pos_start_i = int(pos_start)
            pos_end_i = int(pos_end)
        except Exception as exc:
            logger.debug(
                "ChunkSQLStore.fetch_by_doc_and_position_range: invalid position range: %s",
                exc,
                exc_info=True,
            )
            return []
        if pos_end_i < pos_start_i:
            return []
        if not tenant_id:
            logger.error("ChunkSQLStore.fetch_by_doc_and_position_range requires tenant_id")
            raise ValueError("ChunkSQLStore.fetch_by_doc_and_position_range requires tenant_id")

        def _sync():
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

        return await self._run_sync(_sync)
    async def fetch_by_ids(
        self,
        ids: list[str],
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
        log_context: str = "",
    ) -> list[Chunk]:
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
        def _sync():
            conn = self._conn()
            try:
                # B608: placeholders is "?,?,?" — safe parameterized, no user data interpolated.
                placeholders = ",".join("?" for _ in ids)
                params: list[Any] = list(ids)
                scope_clause = self._scope_where(tenant_id, owner_type, owner_id, params)
                # nosec B608 — placeholders is "?,?,?" only; scope_clause is the fixed
                # string "tenant_id=? AND owner_type=? AND owner_id=?" from _scope_where().
                sql = f"SELECT * FROM chunks WHERE id IN ({placeholders}) AND {scope_clause} AND quarantined_at IS NULL"
                rows = self._query_all(conn, sql, params=params, log_context="fetch_by_ids")
                row_map = {r["id"]: r for r in rows}
                ordered: list[Chunk] = []
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

        return await self._run_sync(_sync)
    async def list_chunks_for_owner(
        self,
        *,
        tenant_id: Optional[str],
        owner_type: str,
        owner_id: str,
        include_quarantined: bool = False,
        limit: Optional[int] = None,
    ) -> list[Chunk]:
        """List chunks for an owner scope. Quarantined excluded by default."""
        if not tenant_id or not owner_type or not owner_id:
            raise ValueError("ChunkSQLStore.list_chunks_for_owner requires scope")
        def _sync():
            conn = self._conn()
            try:
                quarantine_clause = _NO_FILTER if include_quarantined else _QUARANTINE_FILTER
                sql = f"SELECT * FROM chunks WHERE tenant_id=? AND owner_type=? AND owner_id=?{quarantine_clause} ORDER BY updated_at DESC"  # nosec B608 — quarantine_clause is _QUARANTINE_FILTER or _NO_FILTER (module constants)
                chunk_params: list = [tenant_id, owner_type, owner_id]
                if limit:
                    sql += " LIMIT ?"
                    chunk_params.append(int(limit))
                rows = self._query_all(conn, sql, params=chunk_params, log_context="list_chunks_owner")
                return [self._row_to_object(r) for r in rows]
            except Exception:
                logger.exception("ChunkSQLStore.list_chunks_for_owner failed.")
                raise
            finally:
                conn.close()

        return await self._run_sync(_sync)
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
        def _sync():
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

        return await self._run_sync(_sync)
