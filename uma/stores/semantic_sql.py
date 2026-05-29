"""
SemanticSQLStore — SQL + VectorIndex backed semantic memory store for UMA.

This refactored implementation inherits from BaseVectorSQLStore to unify:
- vector search logic,
- ranking preservation,
- SQL connection handling,
- consistent logging and error management.

Semantic memory stores *Facts*, which represent structured knowledge in UMA:
subject, predicate, object triples with metadata and timestamps.

Responsibilities
----------------
- Store Fact objects in a DB-agnostic SQL database.
- Maintain semantic embeddings in a VectorIndex (FAISS, Pinecone, etc.).
- Support upsert with Fact conflict resolution.
- Support semantic fact retrieval with optional subject filters.

Schema (facts table)
--------------------
id TEXT PRIMARY KEY
subject TEXT NOT NULL
predicate TEXT NOT NULL
object TEXT NOT NULL             (JSON)
created_at TEXT NOT NULL         (ISO8601)
updated_at TEXT NOT NULL         (ISO8601)
source_ids TEXT NOT NULL         (JSON list)
confidence REAL NULL
meta TEXT NOT NULL               (JSON dict)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

from .base_vector_sql_store import BaseVectorSQLStore
from .base_sql_store import DEFAULT_TENANT_ID
from uma.retrieve.user_query_helper import extract_keywords_and_phrases
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from uma.common.conflict import FactResolver, LatestWinsFactResolver
from uma.stores.metadata import ensure_store_metadata
from uma.common.types import Fact, SCOPE_MODEL_VERSION
from uma.common.storage_metadata import normalize_fact_metadata
from uma.common.identity import normalize_user_id
from uma.common.types import RuntimeContext

logger = logging.getLogger(__name__)


class SemanticSQLStore(BaseVectorSQLStore):
    """
    SQL + VectorIndex backed semantic fact store.

    Responsibilities:
    - Store Fact objects in SQL
    - Maintain embeddings via VectorIndex
    - Support conflict resolution via FactResolver
    - Support semantic search by embedding + SQL ranking
    """

    def __init__(
        self,
        db_adapter: DBAdapter,
        vector_index: VectorIndex,
        fact_resolver: Optional[FactResolver] = None,
    ) -> None:
        """
        Initialize the SemanticSQLStore.

        Parameters
        ----------
        db_adapter : DBAdapter
            Database backend adapter.
        vector_index : VectorIndex
            Pluggable vector backend for semantic search.
        index : VectorIndex
            Vector index backend (FAISS/etc).
        fact_resolver : Optional[FactResolver]
            Conflict resolution strategy. Defaults to LatestWinsFactResolver.
        """
        super().__init__(db_adapter=db_adapter, vector_index=vector_index)

        self.fact_resolver = fact_resolver or LatestWinsFactResolver()
        self._init_db()

        logger.debug(
            "SemanticSQLStore initialized with resolver=%s, vector_dim=%d",
            type(self.fact_resolver).__name__,
            getattr(vector_index, "dim", getattr(vector_index, "dimension", -1)),
        )

    # ------------------------------------------------------------------ #
    # SQL Schema
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """
        Create semantic facts table if missing.
        """
        conn = self._conn()
        try:
            # Use executescript for multiple DDL statements in sqlite3
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT,
                    session_id TEXT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_ids TEXT NOT NULL,
                    source TEXT,
                    origin_agent_id TEXT,
                    origin_user_id TEXT,
                    origin_session_id TEXT,
                    scope_model_version TEXT,
                    salience REAL NOT NULL,
                    confidence REAL NULL,
                    meta TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_facts_owner ON facts(owner_type, owner_id);
                CREATE INDEX IF NOT EXISTS idx_facts_spo ON facts(subject, predicate);
                """
            )
            self._ensure_column(conn, "facts", "tenant_id", "TEXT NOT NULL DEFAULT 'default'")
            self._ensure_column(conn, "facts", "workspace_id", "TEXT")
            self._ensure_column(conn, "facts", "session_id", "TEXT")
            self._ensure_column(conn, "facts", "origin_agent_id", "TEXT")
            self._ensure_column(conn, "facts", "origin_user_id", "TEXT")
            self._ensure_column(conn, "facts", "origin_session_id", "TEXT")
            self._ensure_column(conn, "facts", "scope_model_version", "TEXT")
            self._ensure_column(conn, "facts", "trust_score", "REAL NOT NULL DEFAULT 0.5")
            self._ensure_column(conn, "facts", "content_hash", "TEXT")
            self._ensure_column(conn, "facts", "quarantined_at", "DATETIME")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_facts_sub_pred
                ON facts(subject, predicate);
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_tenant_owner ON facts(tenant_id, owner_type, owner_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_tenant_sub_pred ON facts(tenant_id, subject, predicate);")
            ensure_store_metadata(self, conn, store_name="semantic")
            conn.commit()
        except Exception:
            self._safe_rollback(conn, "init_db")
            logger.exception("SemanticSQLStore: failed initializing schema.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # BaseVectorSQLStore requirements
    # ------------------------------------------------------------------ #

    @property
    def _table_name(self) -> str:
        return "facts"

    @property
    def _id_column(self) -> str:
        return "id"

    # ------------------------------------------------------------------ #
    # Row → Fact conversion
    # ------------------------------------------------------------------ #

    def _row_to_object(self, row: Any) -> Fact:
        """
        Map a SQL row dict → Fact object.
        """
        # Support both dict-like rows and sqlite3.Row objects (no .get)
        if hasattr(row, "get"):
            owner_type = row.get("owner_type", "user")
            owner_id = row.get("owner_id", "")
            source_ids_val = row.get("source_ids")
            meta_val = row.get("meta")
            object_val = row.get("object")
            salience_val = row.get("salience")
            confidence_val = row.get("confidence")
        else:
            keys = list(row.keys()) if hasattr(row, "keys") else []
            owner_type = row["owner_type"] if "owner_type" in keys else "user"
            owner_id = row["owner_id"] if "owner_id" in keys else ""
            source_ids_val = row["source_ids"] if "source_ids" in keys else None
            meta_val = row["meta"] if "meta" in keys else None
            object_val = row["object"] if "object" in keys else None
            salience_val = row["salience"] if "salience" in keys else 0.0
            confidence_val = row["confidence"] if "confidence" in keys else None

        try:
            obj = json.loads(object_val) if object_val else None
        except Exception:
            logger.exception("SemanticSQLStore: failed to parse object JSON for id=%s", row["id"])
            raise
        try:
            source_ids = json.loads(source_ids_val) if source_ids_val else []
        except Exception:
            logger.exception("SemanticSQLStore: failed to parse source_ids JSON for id=%s", row["id"])
            raise
        try:
            meta = json.loads(meta_val) if meta_val else {}
        except Exception:
            logger.exception("SemanticSQLStore: failed to parse meta JSON for id=%s", row["id"])
            raise

        normalized_meta = normalize_fact_metadata(
            meta,
            fact_id=row["id"],
            owner_type=owner_type,
            owner_id=owner_id,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_ids=source_ids,
            session_id=(row["session_id"] if "session_id" in row.keys() else None),
        )

        row_keys = row.keys() if hasattr(row, "keys") else []
        return Fact(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=obj,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_ids=source_ids,
            confidence=confidence_val,
            meta=normalized_meta,
            salience=salience_val or 0.0,
            tenant_id=(row["tenant_id"] if "tenant_id" in row_keys else DEFAULT_TENANT_ID),
            owner_type=owner_type,
            owner_id=owner_id,
            workspace_id=(row["workspace_id"] if "workspace_id" in row_keys else None),
            session_id=(row["session_id"] if "session_id" in row_keys else None),
            origin_agent_id=(row["origin_agent_id"] if "origin_agent_id" in row_keys else None),
            origin_user_id=(row["origin_user_id"] if "origin_user_id" in row_keys else None),
            origin_session_id=(row["origin_session_id"] if "origin_session_id" in row_keys else None),
            scope_model_version=(row["scope_model_version"] if "scope_model_version" in row_keys else None),
            trust_score=(float(row["trust_score"]) if "trust_score" in row_keys and row["trust_score"] is not None else 0.5),
            content_hash=(row["content_hash"] if "content_hash" in row_keys else None),
            quarantined_at=(
                datetime.fromisoformat(row["quarantined_at"])
                if "quarantined_at" in row_keys and row["quarantined_at"] is not None
                else None
            ),
        )
    # ------------------------------------------------------------------ #
    # Upsert Fact
    # ------------------------------------------------------------------ #

    def _scope_where(
        self,
        tenant_id: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
        params: List[Any],
    ) -> str:
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticSQLStore requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticSQLStore requires tenant_id, owner_type and owner_id")
        params.extend([tenant_id, owner_type, owner_id])
        return "tenant_id=? AND owner_type=? AND owner_id=?"

    async def upsert_fact(self, fact: Fact, embedding: List[float]) -> None:
        """
        Insert or update a Fact and its embedding using conflict resolution.

        Steps:
        1. Fetch existing competing facts via subject + predicate + object.
        2. Resolve conflicts.
        3. Upsert canonical fact.
        4. Upsert embedding into vector index.
        """

        logger.debug(
            "SemanticSQLStore.upsert_fact: id=%s subject=%s pred=%s",
            fact.id,
            fact.subject,
            fact.predicate,
        )

        conn = self._conn()
        try:
            owner_type_in = getattr(fact, "owner_type", "user") or "user"
            owner_id_in = getattr(fact, "owner_id", "") or ""
            tenant_id_in = getattr(fact, "tenant_id", None) or DEFAULT_TENANT_ID
            session_id_in = getattr(fact, "session_id", None)
            if not owner_id_in:
                raise ValueError("SemanticSQLStore.upsert_fact: owner_id must be set")

            # Idempotency guard: avoid duplicating facts on retries when turn_id is present.
            try:
                meta_in = getattr(fact, "meta", None) or {}
                turn_id = meta_in.get("turn_id")
                if turn_id:
                    dup = self._query_one(
                        conn,
                        """
                        SELECT id FROM facts
                        WHERE tenant_id = ? AND owner_type = ? AND owner_id = ?
                          AND subject = ? AND predicate = ?
                          AND object = ?
                          AND ((session_id IS NULL AND ? IS NULL) OR session_id = ?)
                          AND json_extract(meta, '$.turn_id') = ?
                        LIMIT 1
                        """,
                        params=[
                            tenant_id_in,
                            owner_type_in,
                            owner_id_in,
                            fact.subject,
                            fact.predicate,
                            json.dumps(fact.object),
                            session_id_in,
                            session_id_in,
                            str(turn_id),
                        ],
                        log_context="semantic_idempotency",
                    )
                    if dup:
                        logger.info(
                            "SemanticSQLStore.upsert_fact: skipping duplicate (turn_id=%s) id=%s",
                            turn_id,
                            dup["id"] if hasattr(dup, "__getitem__") else None,
                        )
                        return
            except Exception:
                logger.exception("SemanticSQLStore.upsert_fact: idempotency guard failed; continuing.")

            # Fetch competitors: only facts with the SAME (subject, predicate, object) compete.
            # This avoids dropping distinct objects like:
            #   user LIKES sushi
            #   user LIKES pizza
            # which should coexist as separate facts.
            object_json = json.dumps(fact.object)
            rows = self._query_all(
                conn,
                """
                SELECT * FROM facts
                WHERE tenant_id=? AND owner_type=? AND owner_id=? AND subject=? AND predicate=? AND object=?
                  AND ((session_id IS NULL AND ? IS NULL) OR session_id=?)
                """,
                params=[
                    tenant_id_in,
                    owner_type_in,
                    owner_id_in,
                    fact.subject,
                    fact.predicate,
                    object_json,
                    session_id_in,
                    session_id_in,
                ],
                log_context="fetch_conflicts",
            )
            existing = [self._row_to_object(r) for r in rows]

            canonical, _archived = self.fact_resolver.resolve(existing, fact)
            normalized_meta = normalize_fact_metadata(
                canonical.meta,
                fact_id=canonical.id,
                owner_type=canonical.owner_type or owner_type_in,
                owner_id=canonical.owner_id or owner_id_in,
                created_at=canonical.created_at,
                updated_at=canonical.updated_at,
                source_ids=list(canonical.source_ids or []),
                session_id=getattr(canonical, "session_id", None),
            )

            payload = {
                "id": canonical.id,
                "tenant_id": getattr(canonical, "tenant_id", None) or tenant_id_in,
                "subject": canonical.subject,
                "predicate": canonical.predicate,
                "owner_type": canonical.owner_type or owner_type_in,
                "owner_id": canonical.owner_id or owner_id_in,
                "workspace_id": getattr(canonical, "workspace_id", None),
                "session_id": getattr(canonical, "session_id", None),
                "object": json.dumps(canonical.object),
                "created_at": canonical.created_at.isoformat(),
                "updated_at": canonical.updated_at.isoformat(),
                "source_ids": json.dumps(canonical.source_ids),
                "source": getattr(canonical, "source", None),
                "origin_agent_id": getattr(canonical, "origin_agent_id", None),
                "origin_user_id": getattr(canonical, "origin_user_id", None),
                "origin_session_id": getattr(canonical, "origin_session_id", None),
                "scope_model_version": getattr(canonical, "scope_model_version", None) or SCOPE_MODEL_VERSION,
                "salience": canonical.salience,
                "confidence": (
                    float(canonical.confidence)
                    if canonical.confidence is not None
                    else None
                ),
                "trust_score": float(_ts if (_ts := getattr(canonical, "trust_score", None)) is not None else 0.5),
                "content_hash": getattr(canonical, "content_hash", None),
                "quarantined_at": (
                    getattr(canonical, "quarantined_at").isoformat()
                    if getattr(canonical, "quarantined_at", None) is not None
                    else None
                ),
                "meta": json.dumps(normalized_meta),
            }

            # SQL upsert
            self._execute(
                conn,
                """
                INSERT INTO facts (
                    id, tenant_id, subject, predicate, object,
                    created_at, updated_at, source_ids, source,
                    confidence, meta, owner_type, owner_id, workspace_id,
                    session_id, origin_agent_id, origin_user_id,
                    origin_session_id, scope_model_version, salience,
                    trust_score, content_hash, quarantined_at
                ) VALUES (
                    :id, :tenant_id, :subject, :predicate, :object,
                    :created_at, :updated_at, :source_ids, :source,
                    :confidence, :meta, :owner_type, :owner_id, :workspace_id,
                    :session_id, :origin_agent_id, :origin_user_id,
                    :origin_session_id, :scope_model_version, :salience,
                    :trust_score, :content_hash, :quarantined_at
                )
                ON CONFLICT(id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    subject=excluded.subject,
                    predicate=excluded.predicate,
                    object=excluded.object,
                    owner_type=excluded.owner_type,
                    owner_id=excluded.owner_id,
                    workspace_id=excluded.workspace_id,
                    session_id=excluded.session_id,
                    source=excluded.source,
                    origin_agent_id=excluded.origin_agent_id,
                    origin_user_id=excluded.origin_user_id,
                    origin_session_id=excluded.origin_session_id,
                    scope_model_version=excluded.scope_model_version,
                    salience=excluded.salience,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    source_ids=excluded.source_ids,
                    confidence=excluded.confidence,
                    trust_score=excluded.trust_score,
                    content_hash=excluded.content_hash,
                    quarantined_at=excluded.quarantined_at,
                    meta=excluded.meta;
                """,
                params=payload,
                log_context="semantic_upsert",
            )
            # Embedding upsert (commit after vector update)
            try:
                meta = normalized_meta
                topic = meta.get("topic")

                owner_type_out = canonical.owner_type or owner_type_in
                owner_id_out = canonical.owner_id or owner_id_in
                # C1: isolation fields go as explicit parallel-list params.
                # extra_metadata carries everything else.
                extra_meta = {
                    "subject": canonical.subject,
                    "predicate": canonical.predicate,
                    "kb_lane": meta.get("kb_lane"),
                    "scope_key": f"{owner_type_out}:{owner_id_out}",
                }

                if topic:
                    extra_meta["topic"] = topic
                self.vector_index.upsert(
                    ids=[canonical.id],
                    vectors=[embedding],
                    tenant_ids=[tenant_id_in],
                    owner_types=[owner_type_out],
                    owner_ids=[owner_id_out],
                    extra_metadata=[extra_meta],
                )
            except Exception:
                logger.exception(
                    "SemanticSQLStore: vector upsert failed for fact id=%s",
                    canonical.id,
                )
                self._safe_rollback(conn, "upsert_fact")
                raise

            try:
                conn.commit()
            except Exception:
                self._safe_rollback(conn, "upsert_fact_commit")
                try:
                    self.vector_index.delete([canonical.id])
                except Exception:
                    logger.exception(
                        "SemanticSQLStore: vector delete failed after commit error id=%s",
                        canonical.id,
                    )
                raise

            logger.info("SemanticSQLStore: upserted fact id=%s", canonical.id)

        except Exception:
            logger.exception("SemanticSQLStore.upsert_fact failed for fact id=%s", fact.id)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Semantic Search
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query_embedding: List[float],
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        offset: int = 0,
    ) -> List[Fact]:
        """
        Retrieve top-k matching semantic facts.

        Returns ranked list preserving vector search order.
        """
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticSQLStore.search requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticSQLStore.search requires tenant_id, owner_type and owner_id")

        try:
            k_i = max(0, int(k))
        except Exception:
            k_i = 10
        try:
            offset_i = max(0, int(offset))
        except Exception:
            offset_i = 0

        if k_i <= 0:
            return []

        try:
            # Vector indexes do not (yet) support native offset paging; approximate by retrieving
            # a larger window and slicing. This preserves deterministic ordering.
            id_score_pairs = await self._vector_search_ids(
                query_embedding=query_embedding,
                k=k_i + offset_i,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                log_context="semantic_search",
            )
            if not id_score_pairs:
                logger.debug(
                    "SemanticSQLStore.search: vector candidates=0, sql_fetched=0, owner=%s:%s",
                    owner_type,
                    owner_id,
                )
                return []

            windowed_pairs = id_score_pairs[offset_i : offset_i + k_i]
            if not windowed_pairs:
                return []
            windowed_ids = [sid for sid, _score in windowed_pairs]

            facts = await self.fetch_by_ids(
                windowed_ids,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                log_context="semantic_search",
            )
            self._attach_vector_scores(facts, windowed_pairs)
            logger.debug(
                "SemanticSQLStore.search: vector candidates=%d, sql_fetched=%d, owner=%s:%s",
                len(windowed_ids),
                len(facts),
                owner_type,
                owner_id,
            )
            if windowed_ids and not facts:
                logger.warning(
                    "SemanticSQLStore.search: vector candidates=%d but SQL returned 0 op=search owner=%s:%s ids=%s",
                    len(windowed_ids),
                    owner_type,
                    owner_id,
                    windowed_ids[:3],
                )
            return facts
        except Exception:
            logger.exception("SemanticSQLStore.search failed.")
            raise

    async def lexical_search(
        self,
        query_text: str,
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
        k: int = 5,
    ) -> List[Fact]:
        """
        Fallback lexical search over stored document text.
        """
        if not query_text or not isinstance(query_text, str):
            return []
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticSQLStore.lexical_search requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticSQLStore.lexical_search requires tenant_id, owner_type and owner_id")
        try:
            k_i = max(0, int(k))
        except Exception:
            k_i = 5
        if k_i <= 0:
            return []
        terms = []
        if extract_keywords_and_phrases:
            extracted = extract_keywords_and_phrases(query_text)
            terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
        terms = [t for t in terms if isinstance(t, str) and t]
        if not terms:
            return []

        conn = self._conn()
        try:
            where: List[str] = []
            params: List[Any] = []
            term_clauses: List[str] = []
            for term in terms:
                term_clauses.append(
                    "(LOWER(subject) LIKE ? OR LOWER(predicate) LIKE ? OR LOWER(object) LIKE ?)"
                )
                like = f"%{term.lower()}%"
                params.extend([like, like, like])
            if term_clauses:
                where.append(f"({' OR '.join(term_clauses)})")
            where.append(self._scope_where(tenant_id, owner_type, owner_id, params))

            sql = f"""
                SELECT * FROM facts
                WHERE {' AND '.join(where)}
                ORDER BY updated_at DESC
                LIMIT {int(k_i)}
            """
            rows = self._query_all(conn, sql, params=params, log_context="lexical_search")
            return [self._row_to_object(r) for r in rows]
        except Exception:
            logger.exception("SemanticSQLStore.lexical_search failed.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Fact Listing (required by Consolidator + Pruner)
    # ------------------------------------------------------------------ #
    async def list_facts_for_subject(
        self,
        subject: str,
        limit: Optional[int] = None,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        """
        Return all facts for a given subject, ordered by updated_at DESC.

        Parameters
        ----------
        subject : str
            User or entity whose facts should be listed.
        limit : Optional[int]
            Optionally restrict number of returned facts.

        Returns
        -------
        List[Fact]
        """
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticSQLStore.list_facts_for_subject requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticSQLStore.list_facts_for_subject requires tenant_id, owner_type and owner_id")
        conn = self._conn()
        try:
            where_clauses = ["subject=?"]
            params = [subject]
            where_clauses.append("tenant_id=?")
            where_clauses.append("owner_type=?")
            where_clauses.append("owner_id=?")
            where_clauses.append("quarantined_at IS NULL")
            params.extend([tenant_id, owner_type, owner_id])

            sql = f"SELECT * FROM facts WHERE {' AND '.join(where_clauses)} ORDER BY updated_at DESC"
            # Tie-break by id to ensure deterministic paging.
            sql = sql.replace("ORDER BY updated_at DESC", "ORDER BY updated_at DESC, id ASC")
            if limit:
                sql += f" LIMIT {int(limit)}"

            rows = self._query_all(conn, sql, params=params, log_context="list_facts")
            return [self._row_to_object(r) for r in rows]

        except Exception:
            logger.exception("SemanticSQLStore.list_facts_for_subject failed.")
            raise
        finally:
            conn.close()

    async def list_facts_for_owner(
        self,
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
        limit: Optional[int] = None,
        include_quarantined: bool = False,
    ) -> List[Fact]:
        """
        Return all facts for a given owner scope, ordered by updated_at DESC.
        Quarantined facts are excluded by default; pass include_quarantined=True for management use.
        """
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticSQLStore.list_facts_for_owner requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticSQLStore.list_facts_for_owner requires tenant_id, owner_type and owner_id")
        conn = self._conn()
        try:
            quarantine_clause = "" if include_quarantined else " AND quarantined_at IS NULL"
            sql = f"SELECT * FROM facts WHERE tenant_id=? AND owner_type=? AND owner_id=?{quarantine_clause} ORDER BY updated_at DESC, id ASC"
            if limit:
                sql += f" LIMIT {int(limit)}"
            rows = self._query_all(conn, sql, params=[tenant_id, owner_type, owner_id], log_context="list_facts_owner")
            return [self._row_to_object(r) for r in rows]
        except Exception:
            logger.exception("SemanticSQLStore.list_facts_for_owner failed.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Fetch Facts by IDs (snippet-first helpers)
    # ------------------------------------------------------------------ #
    async def fetch_facts_by_ids(
        self,
        ids: List[str],
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        """
        Fetch Fact objects by ID, preserving requested order.
        """
        return await self._fetch_facts_by_ids_sql(
            ids=ids,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            log_context="fetch_facts_by_ids",
        )

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        log_context: str = "",
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        return await self._fetch_facts_by_ids_sql(
            ids=ids,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            log_context=log_context or "fetch_by_ids",
        )

    async def _fetch_facts_by_ids_sql(
        self,
        *,
        ids: List[str],
        tenant_id: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
        log_context: str,
    ) -> List[Fact]:
        if not ids:
            return []
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticSQLStore.fetch_facts_by_ids requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticSQLStore.fetch_facts_by_ids requires tenant_id, owner_type and owner_id")

        conn = self._conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            params = ids[:]
            scope_clause = self._scope_where(tenant_id, owner_type, owner_id, params)
            sql = f"SELECT * FROM facts WHERE id IN ({placeholders}) AND {scope_clause} AND quarantined_at IS NULL"
            rows = self._query_all(
                conn,
                sql,
                params=params,
                log_context=log_context,
            )
            row_map = {r["id"]: r for r in rows}
            ordered: List[Fact] = []
            for fid in ids:
                row = row_map.get(fid)
                if row is None:
                    continue
                ordered.append(self._row_to_object(row))
            return ordered
        except Exception:
            logger.exception("SemanticSQLStore.fetch_facts_by_ids failed.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Fact Deletion (required by Pruner)
    # ------------------------------------------------------------------ #
    async def delete_fact(
        self,
        fact_id: str,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> None:
        """
        Delete a fact from SQL + remove its embedding from VectorIndex.

        Parameters
        ----------
        fact_id : str
            Primary key of the fact to delete.
        """
        if not tenant_id or not owner_type or not owner_id:
            logger.error("SemanticSQLStore.delete_fact requires tenant_id, owner_type and owner_id")
            raise ValueError("SemanticSQLStore.delete_fact requires tenant_id, owner_type and owner_id")
        conn = self._conn()
        try:
            sql = "DELETE FROM facts WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?"
            params = [fact_id, tenant_id, owner_type, owner_id]
            self._execute(conn, sql, params=params, log_context="delete_fact")
            conn.commit()

            # Vector index delete
            try:
                self.vector_index.delete([fact_id])
            except Exception:
                logger.exception(
                    "SemanticSQLStore: vector index deletion failed for fact id=%s",
                    fact_id,
                )

            logger.info(
                "SemanticSQLStore: deleted fact id=%s owner=%s:%s",
                fact_id,
                owner_type,
                owner_id,
            )

        except Exception:
            self._safe_rollback(conn, "delete_fact")
            logger.exception(
                "SemanticSQLStore.delete_fact failed id=%s owner=%s:%s",
                fact_id,
                owner_type,
                owner_id,
            )
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Single-record fetch (bypass quarantine filter for management use)
    # ------------------------------------------------------------------ #

    async def get_fact(
        self,
        fact_id: str,
        *,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Optional[Fact]:
        """Fetch a single fact by ID. Returns None if not found (quarantined records included)."""
        if not tenant_id or not owner_type or not owner_id:
            raise ValueError("SemanticSQLStore.get_fact requires tenant_id, owner_type and owner_id")
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT * FROM facts WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[fact_id, tenant_id, owner_type, owner_id],
                log_context="get_fact",
            )
            return self._row_to_object(row) if row else None
        except Exception:
            logger.exception("SemanticSQLStore.get_fact failed id=%s", fact_id)
            raise
        finally:
            conn.close()

    async def update_trust(
        self,
        fact_id: str,
        new_score: float,
        *,
        reason: str,
        ctx: RuntimeContext,
    ) -> None:
        if not isinstance(ctx, RuntimeContext):
            raise TypeError("SemanticSQLStore.update_trust requires a RuntimeContext")
        if not fact_id or not isinstance(fact_id, str):
            raise ValueError("SemanticSQLStore.update_trust requires fact_id as a non-empty string")

        try:
            normalized_score = float(new_score)
        except Exception as exc:
            raise ValueError("SemanticSQLStore.update_trust: new_score must be a float in [0.0, 1.0]") from exc
        if not (0.0 <= normalized_score <= 1.0):
            raise ValueError("SemanticSQLStore.update_trust: new_score must be a float in [0.0, 1.0]")

        normalized_reason = str(reason or "").strip()
        if not normalized_reason:
            raise ValueError("SemanticSQLStore.update_trust: reason must be a non-empty string")

        owner_refs: list[tuple[str, str]] = [("agent", ctx.agent_id)]
        if ctx.user_id:
            owner_refs.append(("user", normalize_user_id(ctx.user_id)))
        if ctx.workspace_id:
            owner_refs.append(("workspace", ctx.workspace_id))

        conn = self._conn()
        try:
            row = None
            matched_owner: tuple[str, str] | None = None
            for owner_type, owner_id in owner_refs:
                candidate = self._query_one(
                    conn,
                    "SELECT * FROM facts WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                    params=[fact_id, ctx.tenant_id, owner_type, owner_id],
                    log_context="update_trust_fetch",
                )
                if candidate is not None:
                    row = candidate
                    matched_owner = (owner_type, owner_id)
                    break

            if row is None or matched_owner is None:
                raise ValueError(f"SemanticSQLStore.update_trust: fact {fact_id!r} not found")

            fact = self._row_to_object(row)
            meta = dict(getattr(fact, "meta", None) or {})
            trust_updates = list(meta.get("trust_updates") or [])
            trust_updates.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prior_score": float(getattr(fact, "trust_score", 0.5) or 0.5),
                    "new_score": normalized_score,
                    "reason": normalized_reason,
                }
            )
            meta["trust_updates"] = trust_updates
            normalized_meta = normalize_fact_metadata(
                meta,
                fact_id=fact.id,
                owner_type=fact.owner_type,
                owner_id=fact.owner_id,
                created_at=fact.created_at,
                updated_at=fact.updated_at,
                source_ids=list(fact.source_ids or []),
                session_id=getattr(fact, "session_id", None),
            )
            self._execute(
                conn,
                """
                UPDATE facts
                SET trust_score=?, meta=?
                WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?
                """,
                params=[
                    normalized_score,
                    json.dumps(normalized_meta),
                    fact.id,
                    ctx.tenant_id,
                    matched_owner[0],
                    matched_owner[1],
                ],
                log_context="update_trust",
            )
            conn.commit()
            logger.info(
                "SemanticSQLStore.update_trust: fact=%s tenant=%s owner=%s:%s prior=%0.4f new=%0.4f",
                fact.id,
                ctx.tenant_id,
                matched_owner[0],
                matched_owner[1],
                float(getattr(fact, "trust_score", 0.5) or 0.5),
                normalized_score,
            )
        except Exception:
            self._safe_rollback(conn, "update_trust")
            logger.exception("SemanticSQLStore.update_trust failed fact_id=%s tenant=%s", fact_id, getattr(ctx, "tenant_id", None))
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Quarantine Management
    # ------------------------------------------------------------------ #

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
            raise ValueError("SemanticSQLStore.quarantine_record requires scope")
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT meta FROM facts WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[record_id, tenant_id, owner_type, owner_id],
                log_context="quarantine_fact_fetch",
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
                "UPDATE facts SET quarantined_at=?, meta=? WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[quarantined_at, json.dumps(meta), record_id, tenant_id, owner_type, owner_id],
                log_context="quarantine_fact",
            )
            conn.commit()
            logger.info("SemanticSQLStore: quarantined fact id=%s owner=%s:%s", record_id, owner_type, owner_id)
            return True
        except Exception:
            self._safe_rollback(conn, "quarantine_fact")
            logger.exception("SemanticSQLStore.quarantine_record failed id=%s", record_id)
            raise
        finally:
            conn.close()

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
            raise ValueError("SemanticSQLStore.reinstate_quarantined_record requires scope")
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT meta FROM facts WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[record_id, tenant_id, owner_type, owner_id],
                log_context="reinstate_fact_fetch",
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
                "UPDATE facts SET quarantined_at=NULL, meta=? WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[json.dumps(meta), record_id, tenant_id, owner_type, owner_id],
                log_context="reinstate_fact",
            )
            conn.commit()
            logger.info("SemanticSQLStore: reinstated fact id=%s owner=%s:%s", record_id, owner_type, owner_id)
            return True
        except Exception:
            self._safe_rollback(conn, "reinstate_fact")
            logger.exception("SemanticSQLStore.reinstate_quarantined_record failed id=%s", record_id)
            raise
        finally:
            conn.close()
