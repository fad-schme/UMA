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
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from ..adapters.vector.faiss_adapter import FaissIndex
from ..core.utils.conflict import FactResolver, LatestWinsFactResolver
from ..types_fact import Fact

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
        index : FaissIndex
            Underlying FAISS index for compatibility/logging.
        fact_resolver : Optional[FactResolver]
            Conflict resolution strategy. Defaults to LatestWinsFactResolver.
        """
        super().__init__(db_adapter=db_adapter, vector_index=vector_index)

        self.fact_resolver = fact_resolver or LatestWinsFactResolver()
        self._init_db()

        logger.info(
            "SemanticSQLStore initialized with resolver=%s, faiss_dim=%d",
            type(self.fact_resolver).__name__,
            getattr(vector_index.index, "dimension", -1),
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_ids TEXT NOT NULL,
                    confidence REAL NULL,
                    meta TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_facts_sub_pred
                ON facts(subject, predicate);
                """
            )
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
        return Fact(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=json.loads(row["object"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source_ids=json.loads(row["source_ids"]),
            confidence=row["confidence"],
            meta=json.loads(row["meta"]),
        )

    # ------------------------------------------------------------------ #
    # Upsert Fact
    # ------------------------------------------------------------------ #

    async def upsert_fact(self, fact: Fact, embedding: List[float]) -> None:
        """
        Insert or update a Fact and its embedding using conflict resolution.

        Steps:
        1. Fetch existing competing facts via subject + predicate.
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
            # Fetch competitors
            rows = self._query_all(
                conn,
                "SELECT * FROM facts WHERE subject=? AND predicate=?",
                params=[fact.subject, fact.predicate],
                log_context="fetch_conflicts",
            )
            existing = [self._row_to_object(r) for r in rows]

            canonical, _archived = self.fact_resolver.resolve(existing, fact)

            payload = {
                "id": canonical.id,
                "subject": canonical.subject,
                "predicate": canonical.predicate,
                "object": json.dumps(canonical.object),
                "created_at": canonical.created_at.isoformat(),
                "updated_at": canonical.updated_at.isoformat(),
                "source_ids": json.dumps(canonical.source_ids),
                "confidence": canonical.confidence,
                "meta": json.dumps(canonical.meta),
            }

            # SQL upsert
            self._execute(
                conn,
                """
                INSERT INTO facts (
                    id, subject, predicate, object,
                    created_at, updated_at, source_ids,
                    confidence, meta
                ) VALUES (
                    :id, :subject, :predicate, :object,
                    :created_at, :updated_at, :source_ids,
                    :confidence, :meta
                )
                ON CONFLICT(id) DO UPDATE SET
                    subject=excluded.subject,
                    predicate=excluded.predicate,
                    object=excluded.object,
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
                self.vector_index.upsert(
                    ids=[canonical.id],
                    vectors=[embedding],
                    metadata=[{"subject": canonical.subject, "predicate": canonical.predicate}],
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
        subject: Optional[str] = None,
        k: int = 10,
    ) -> List[Fact]:
        """
        Retrieve top-k matching semantic facts.

        Returns ranked list preserving vector search order.
        """
        filters = {"subject": subject} if subject else None

        try:
            return await self._semantic_search(
                query_embedding=query_embedding,
                k=k,
                filters=filters,
                log_context="semantic_search",
            )
        except Exception:
            logger.exception("SemanticSQLStore.search failed.")
            raise

    # ------------------------------------------------------------------ #
    # Fact Listing (required by Consolidator + Pruner)
    # ------------------------------------------------------------------ #
    async def list_facts_for_subject(self, subject: str, limit: Optional[int] = None) -> List[Fact]:
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
        conn = self._conn()
        try:
            sql = """
                SELECT * FROM facts
                WHERE subject=?
                ORDER BY updated_at DESC
            """
            if limit:
                sql += f" LIMIT {int(limit)}"

            rows = self._query_all(conn, sql, params=[subject], log_context="list_facts")
            return [self._row_to_object(r) for r in rows]

        except Exception:
            logger.exception("SemanticSQLStore.list_facts_for_subject failed.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Fetch Facts by IDs (snippet-first helpers)
    # ------------------------------------------------------------------ #
    async def fetch_facts_by_ids(self, ids: List[str]) -> List[Fact]:
        """
        Fetch Fact objects by ID, preserving requested order.
        """
        if not ids:
            return []

        conn = self._conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            rows = self._query_all(
                conn,
                f"SELECT * FROM facts WHERE id IN ({placeholders})",
                params=ids,
                log_context="fetch_facts_by_ids",
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
    async def delete_fact(self, fact_id: str) -> None:
        """
        Delete a fact from SQL + remove its embedding from VectorIndex.

        Parameters
        ----------
        fact_id : str
            Primary key of the fact to delete.
        """
        conn = self._conn()
        try:
            # SQL delete
            self._execute(
                conn,
                "DELETE FROM facts WHERE id=?",
                params=[fact_id],
                log_context="delete_fact",
            )
            conn.commit()

            # Vector index delete
            try:
                self.vector_index.delete([fact_id])
            except Exception:
                logger.exception(
                    "SemanticSQLStore: vector index deletion failed for fact id=%s",
                    fact_id,
                )

            logger.info("SemanticSQLStore: deleted fact id=%s", fact_id)

        except Exception:
            self._safe_rollback(conn, "delete_fact")
            logger.exception("SemanticSQLStore.delete_fact failed.")
            raise
        finally:
            conn.close()
