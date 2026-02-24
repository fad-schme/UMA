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
from datetime import datetime
from typing import List, Optional, Any

from .base_vector_sql_store import BaseVectorSQLStore
from ..core.utils.user_query_helper import extract_keywords_and_phrases
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from ..core.utils.conflict import FactResolver, LatestWinsFactResolver
from ..core.utils.store_metadata import ensure_store_metadata
from ..types import Fact

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
            Vector index backend (FAISS/Qdrant/etc).
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
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_ids TEXT NOT NULL,
                    source TEXT,
                    salience REAL NOT NULL,
                    confidence REAL NULL,
                    meta TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_facts_owner ON facts(owner_type, owner_id);
                CREATE INDEX IF NOT EXISTS idx_facts_spo ON facts(subject, predicate);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_facts_sub_pred
                ON facts(subject, predicate);
                """
            )
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

        return Fact(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=obj,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_ids=source_ids,
            confidence=confidence_val,
            meta=meta,
            salience=salience_val or 0.0,
            owner_type=owner_type,
            owner_id=owner_id,
        )
    # ------------------------------------------------------------------ #
    # Upsert Fact
    # ------------------------------------------------------------------ #

    def _owner_where(self, owner_type: Optional[str], owner_id: Optional[str], params: List[Any]) -> str:
        if not owner_type or not owner_id:
            logger.error("SemanticSQLStore requires owner_type and owner_id")
            raise ValueError("SemanticSQLStore requires owner_type and owner_id")
        params.extend([owner_type, owner_id])
        return "owner_type=? AND owner_id=?"

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
                        WHERE owner_type = ? AND owner_id = ?
                          AND subject = ? AND predicate = ?
                          AND object = ?
                          AND json_extract(meta, '$.turn_id') = ?
                        LIMIT 1
                        """,
                        params=[
                            owner_type_in,
                            owner_id_in,
                            fact.subject,
                            fact.predicate,
                            json.dumps(fact.object),
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
                "SELECT * FROM facts WHERE owner_type=? AND owner_id=? AND subject=? AND predicate=? AND object=?",
                params=[owner_type_in, owner_id_in, fact.subject, fact.predicate, object_json],
                log_context="fetch_conflicts",
            )
            existing = [self._row_to_object(r) for r in rows]

            canonical, _archived = self.fact_resolver.resolve(existing, fact)

            payload = {
                "id": canonical.id,
                "subject": canonical.subject,
                "predicate": canonical.predicate,
                "owner_type": canonical.owner_type or owner_type_in,
                "owner_id": canonical.owner_id or owner_id_in,
                "object": json.dumps(canonical.object),
                "created_at": canonical.created_at.isoformat(),
                "updated_at": canonical.updated_at.isoformat(),
                "source_ids": json.dumps(canonical.source_ids),
                "salience": canonical.salience,
                "confidence": (
                    float(canonical.confidence)
                    if canonical.confidence is not None
                    else None
                ),
                "meta": json.dumps(canonical.meta),
            }

            # SQL upsert
            self._execute(
                conn,
                """
                INSERT INTO facts (
                    id, subject, predicate, object,
                    created_at, updated_at, source_ids,
                    confidence, meta, owner_type, owner_id, salience
                ) VALUES (
                    :id, :subject, :predicate, :object,
                    :created_at, :updated_at, :source_ids,
                    :confidence, :meta, :owner_type, :owner_id, :salience
                )
                ON CONFLICT(id) DO UPDATE SET
                    subject=excluded.subject,
                    predicate=excluded.predicate,
                    object=excluded.object,
                    owner_type=excluded.owner_type,
                    owner_id=excluded.owner_id,
                    salience=excluded.salience,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    source_ids=excluded.source_ids,
                    confidence=excluded.confidence,
                    meta=excluded.meta;
                """,
                params=payload,
                log_context="semantic_upsert",
            )
            # Embedding upsert (commit after vector update)
            try:
                meta = canonical.meta if isinstance(canonical.meta, dict) else {}
                topic = meta.get("topic")

                owner_type_out = canonical.owner_type or owner_type_in
                owner_id_out = canonical.owner_id or owner_id_in
                vector_meta = {
                    "subject": canonical.subject,
                    "predicate": canonical.predicate,
                    "owner_type": owner_type_out,
                    "owner_id": owner_id_out,
                    "scope_key": f"{owner_type_out}:{owner_id_out}",
                }

                if topic:
                    vector_meta["topic"] = topic
                self.vector_index.upsert(
                    ids=[canonical.id],
                    vectors=[embedding],
                    metadata=[vector_meta],
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
        owner_type: str,
        owner_id: str,
        k: int = 10,
        offset: int = 0,
    ) -> List[Fact]:
        """
        Retrieve top-k matching semantic facts.

        Returns ranked list preserving vector search order.
        """
        if not owner_type or not owner_id:
            logger.error("SemanticSQLStore.search requires owner_type and owner_id")
            raise ValueError("SemanticSQLStore.search requires owner_type and owner_id")

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

        filters: dict[str, Any] = {"owner_type": owner_type, "owner_id": owner_id}

        try:
            # Vector indexes do not (yet) support native offset paging; approximate by retrieving
            # a larger window and slicing. This preserves deterministic ordering.
            ids = await self._vector_search_ids(
                query_embedding=query_embedding,
                k=k_i + offset_i,
                filters=filters,
                log_context="semantic_search",
                id_prefix="fact_",
            )
            if not ids:
                logger.debug(
                    "SemanticSQLStore.search: vector candidates=0, sql_fetched=0, owner=%s:%s",
                    owner_type,
                    owner_id,
                )
                return []

            windowed_ids = ids[offset_i : offset_i + k_i]
            if not windowed_ids:
                return []

            facts = await self.fetch_by_ids(
                windowed_ids,
                owner_type=owner_type,
                owner_id=owner_id,
                log_context="semantic_search",
            )
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

    async def search_text(
        self,
        query: str,
        *,
        owner_type: str,
        owner_id: str,
        limit: int = 5,
    ) -> List[Fact]:
        """
        Fallback lexical search over stored document text.
        """
        if not query or not isinstance(query, str):
            return []
        if not owner_type or not owner_id:
            logger.error("SemanticSQLStore.search_text requires owner_type and owner_id")
            raise ValueError("SemanticSQLStore.search_text requires owner_type and owner_id")
        terms = []
        if extract_keywords_and_phrases:
            extracted = extract_keywords_and_phrases(query)
            terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
        terms = [t for t in terms if isinstance(t, str) and t]
        if not terms:
            return []

        conn = self._conn()
        try:
            where = ["predicate='document'"]
            params: List[Any] = []
            for term in terms:
                where.append("LOWER(object) LIKE ?")
                params.append(f"%{term}%")
            where.append(self._owner_where(owner_type, owner_id, params))

            sql = f"""
                SELECT * FROM facts
                WHERE {' AND '.join(where)}
                ORDER BY updated_at DESC
                LIMIT {int(limit)}
            """
            rows = self._query_all(conn, sql, params=params, log_context="search_text")
            return [self._row_to_object(r) for r in rows]
        except Exception:
            logger.exception("SemanticSQLStore.search_text failed.")
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
        if not owner_type or not owner_id:
            logger.error("SemanticSQLStore.list_facts_for_subject requires owner_type and owner_id")
            raise ValueError("SemanticSQLStore.list_facts_for_subject requires owner_type and owner_id")
        conn = self._conn()
        try:
            where_clauses = ["subject=?"]
            params = [subject]
            where_clauses.append("owner_type=?")
            where_clauses.append("owner_id=?")
            params.extend([owner_type, owner_id])

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
        owner_type: str,
        owner_id: str,
        limit: Optional[int] = None,
    ) -> List[Fact]:
        """
        Return all facts for a given owner scope, ordered by updated_at DESC.
        """
        if not owner_type or not owner_id:
            logger.error("SemanticSQLStore.list_facts_for_owner requires owner_type and owner_id")
            raise ValueError("SemanticSQLStore.list_facts_for_owner requires owner_type and owner_id")
        conn = self._conn()
        try:
            sql = "SELECT * FROM facts WHERE owner_type=? AND owner_id=? ORDER BY updated_at DESC, id ASC"
            if limit:
                sql += f" LIMIT {int(limit)}"
            rows = self._query_all(conn, sql, params=[owner_type, owner_id], log_context="list_facts_owner")
            return [self._row_to_object(r) for r in rows]
        except Exception:
            logger.exception("SemanticSQLStore.list_facts_for_owner failed.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Fetch Facts by IDs (snippet-first helpers)
    # ------------------------------------------------------------------ #
    async def fetch_facts_by_ids(self, ids: List[str], owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,) -> List[Fact]:
        """
        Fetch Fact objects by ID, preserving requested order.
        """
        return await self._fetch_facts_by_ids_sql(
            ids=ids,
            owner_type=owner_type,
            owner_id=owner_id,
            log_context="fetch_facts_by_ids",
        )

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        log_context: str = "",
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Fact]:
        return await self._fetch_facts_by_ids_sql(
            ids=ids,
            owner_type=owner_type,
            owner_id=owner_id,
            log_context=log_context or "fetch_by_ids",
        )

    async def _fetch_facts_by_ids_sql(
        self,
        *,
        ids: List[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
        log_context: str,
    ) -> List[Fact]:
        if not ids:
            return []
        if not owner_type or not owner_id:
            logger.error("SemanticSQLStore.fetch_facts_by_ids requires owner_type and owner_id")
            raise ValueError("SemanticSQLStore.fetch_facts_by_ids requires owner_type and owner_id")

        conn = self._conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            params = ids[:]
            owner_clause = self._owner_where(owner_type, owner_id, params)
            sql = f"SELECT * FROM facts WHERE id IN ({placeholders}) AND {owner_clause}"
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
        if not owner_type or not owner_id:
            logger.error("SemanticSQLStore.delete_fact requires owner_type and owner_id")
            raise ValueError("SemanticSQLStore.delete_fact requires owner_type and owner_id")
        conn = self._conn()
        try:
            # SQL delete (conditionally filter by owner if provided)
            sql = "DELETE FROM facts WHERE id=? AND owner_type=? AND owner_id=?"
            params = [fact_id, owner_type, owner_id]
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
