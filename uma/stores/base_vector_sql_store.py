"""
BaseVectorSQLStore — Common foundation for UMA stores requiring both SQL
persistence and vector-index retrieval.

This class provides a consistent implementation of the retrieval pipeline:
    1. Vector ANN search → ranked list of IDs
    2. SQL lookup by those IDs
    3. Row-to-domain-model conversion
    4. Ranking preservation
    5. Robust error handling

Stores using embeddings (SemanticSQLStore, EpisodicSQLStore,
ProceduralSQLStore) must inherit from this class.

Coding agent instructions
-------------------------
- DO NOT put any domain logic here.
- Subclasses MUST implement:
      _row_to_object(row)
      _table_name
      _id_column
- Subclasses MAY override:
      build_metadata()
      _postprocess_row(obj)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base_sql_store import BaseSQLStore
from ..adapters.vector.base import VectorIndex

logger = logging.getLogger(__name__)


class BaseVectorSQLStore(BaseSQLStore):
    """
    Base class for memory stores combining SQL persistence with
    vector similarity search.

    This abstraction ensures:
        • consistent semantic search behavior
        • backend-agnostic vector index usage
        • ranking preserved exactly as returned by ANN search
        • reliable SQL lookups and graceful error handling
    """

    def __init__(self, db_adapter, vector_index: VectorIndex) -> None:
        """
        Initialize the vector store.

        Parameters
        ----------
        db_adapter : DBAdapter
            Abstraction for DB-API compatible connection creation.
        vector_index : VectorIndex
            Backend for embedding upsert and vector similarity queries.
        """
        super().__init__(db_adapter=db_adapter)
        self.vector_index = vector_index

    # ------------------------------------------------------------------ #
    # REQUIRED SUBCLASS METHODS
    # ------------------------------------------------------------------ #

    def _row_to_object(self, row: Any) -> Any:
        """
        Convert a DB row into a domain-specific object.

        Subclasses MUST implement this (e.g., Fact, Episode, Skill).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _row_to_object()"
        )

    # Whitelisted table names and column names for SQL construction.
    # Every value used in f-string SQL templates must appear here.
    # This eliminates the B608 injection vector structurally: subclasses
    # returning anything outside these sets raise at property access, not
    # silently at query execution.
    _ALLOWED_TABLE_NAMES: frozenset = frozenset(
        {"facts", "episodes", "skills", "chunks"}
    )
    _ALLOWED_ID_COLUMNS: frozenset = frozenset({"id"})

    @property
    def _table_name(self) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} must define _table_name"
        )

    @property
    def _id_column(self) -> str:
        """Return the name of the primary key column. Defaults to 'id'."""
        return "id"

    def _validated_table_name(self) -> str:
        """Return _table_name after asserting it is in the allowed whitelist."""
        name = self._table_name
        if name not in self._ALLOWED_TABLE_NAMES:
            raise ValueError(
                f"{self.__class__.__name__}._table_name returned {name!r}, "
                f"which is not in the allowed set {sorted(self._ALLOWED_TABLE_NAMES)}. "
                f"Add it to _ALLOWED_TABLE_NAMES only if it is a known UMA schema table."
            )
        return name

    def _validated_id_column(self) -> str:
        """Return _id_column after asserting it is in the allowed whitelist."""
        col = self._id_column
        if col not in self._ALLOWED_ID_COLUMNS:
            raise ValueError(
                f"{self.__class__.__name__}._id_column returned {col!r}, "
                f"which is not in the allowed set {sorted(self._ALLOWED_ID_COLUMNS)}. "
                f"Add it to _ALLOWED_ID_COLUMNS only if it is a known UMA schema column."
            )
        return col

    # Optional row filtering hook
    def _postprocess_row(self, obj: Any) -> Any:
        return obj

    # ------------------------------------------------------------------ #
    # VECTOR SEARCH (Step 1)
    # ------------------------------------------------------------------ #

    async def _vector_search_ids(
        self,
        query_embedding: List[float],
        k: int,
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        extra_filters: Optional[Dict[str, Any]] = None,
        log_context: str = "",
    ) -> List[Tuple[str, float]]:
        """
        Run a vector search and return a ranked list of (id, score).

        Parameters
        ----------
        query_embedding : List[float]
            Embedding vector used for ANN search.
        k : int
            Number of nearest neighbors to return.
        tenant_id, owner_type, owner_id : str
            Required isolation scope. C1: passed through to the vector
            index, which pushes them into the backend's native predicate
            before the candidate cap is applied. Cross-tenant rows
            cannot leak past this boundary.
        extra_filters : Optional[Dict[str, Any]]
            Optional non-isolation filters (e.g. `{"doc_id": "..."}`).
        log_context : str
            Used to help contextualize logs.

        Returns
        -------
        List[Tuple[str, float]]
            Ordered list of (id, score) pairs, as returned by the vector backend.
        """

        ctx = f" [{log_context}]" if log_context else ""

        if not isinstance(query_embedding, list) or not query_embedding:
            logger.error("%s Vector search aborted: invalid embedding%s",
                         self.__class__.__name__, ctx)
            return []

        # Validate numeric embedding
        if not all(isinstance(x, (float, int)) for x in query_embedding):
            logger.error("%s Embedding contains non-numeric values%s",
                         self.__class__.__name__, ctx)
            return []

        try:
            id_score_pairs = self.vector_index.query(
                vector=query_embedding,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                k=k,
                extra_filters=extra_filters,
            )
        except Exception:
            ctx = f" [{log_context}]" if log_context else ""
            logger.exception(
                "%s BaseVectorSQLStore._vector_search_ids%s: vector backend failed.",
                self.__class__.__name__,
                ctx,
            )
            return []

        if not id_score_pairs:
            logger.debug("%s Vector search returned no results%s",
                         self.__class__.__name__, ctx)
            return []

        # Validate format: [(id, score), ...]
        valid: List[Tuple[str, float]] = []
        seen: set[str] = set()
        for pair in id_score_pairs:
            try:
                sid, raw_score = pair
                if not isinstance(sid, str):
                    logger.warning(
                        "%s Invalid vector search result element=%r%s",
                        self.__class__.__name__, pair, ctx
                    )
                    continue
                if sid in seen:
                    continue
                try:
                    score = float(raw_score)
                except Exception:
                    logger.warning(
                        "%s Invalid vector score element=%r%s",
                        self.__class__.__name__,
                        pair,
                        ctx,
                    )
                    continue
                seen.add(sid)
                valid.append((sid, score))
            except Exception:
                logger.exception(
                    "%s Malformed vector search result element=%r%s",
                    self.__class__.__name__, pair, ctx
                )

        return valid

    def _attach_vector_scores(self, items: Sequence[Any], id_score_pairs: Sequence[Tuple[str, float]]) -> None:
        """
        Attach vector backend scores to each item's `.meta` dict under `vector_score`.

        This preserves dense retrieval scores end-to-end without changing the domain model.
        """
        if not items or not id_score_pairs:
            return

        score_by_id: Dict[str, float] = {sid: float(score) for sid, score in id_score_pairs if sid}
        if not score_by_id:
            return

        for obj in items:
            try:
                sid = getattr(obj, "id", None)
                if not isinstance(sid, str) or not sid:
                    continue
                score = score_by_id.get(sid)
                if score is None:
                    continue
                meta = getattr(obj, "meta", None) or {}
                if not isinstance(meta, dict):
                    meta = {}
                meta["vector_score"] = float(score)
                obj.meta = meta  # type: ignore[attr-defined]
            except Exception:
                logger.exception(
                    "%s Failed attaching vector_score to object=%r",
                    self.__class__.__name__,
                    obj,
                )

    # ------------------------------------------------------------------ #
    # Shared SQL lookup after vector ID retrieval
    # ------------------------------------------------------------------ #

    async def _fetch_ranked_rows_by_ids(
        self,
        ids: List[str],
        log_context: str = "",
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Fetch rows by primary key in caller-supplied order, owner- and
        tenant-scoped, excluding quarantined records.

        Scope contract (CR1, matches typed fetch_by_ids overrides):
        - tenant_id, owner_type, owner_id are all REQUIRED. None / empty
          strings raise ValueError. Earlier versions accepted None for
          tenant_id and silently issued an unscoped SQL fetch — a DAT break
          identical to the one closed in graph_updater (H4).
        - Every fetch includes "AND quarantined_at IS NULL". Quarantined
          rows remain in the table for management API access; they are
          invisible to every normal retrieval path. The base SQL here was
          previously missing this clause, allowing ProceduralSQLStore.search
          (which goes through _semantic_search → _fetch_ranked_rows_by_ids)
          to surface quarantined skills.
        """
        if not ids:
            return []
        if not tenant_id or not owner_type or not owner_id:
            logger.error(
                "%s _fetch_ranked_rows_by_ids requires tenant_id, owner_type and owner_id%s",
                self.__class__.__name__,
                f" [{log_context}]" if log_context else "",
            )
            raise ValueError(
                f"{self.__class__.__name__}._fetch_ranked_rows_by_ids "
                f"requires tenant_id, owner_type and owner_id"
            )

        ctx = f" [{log_context}]" if log_context else ""

        def _sync() -> Dict[str, Any]:
            conn = self._conn()
            try:
                placeholders = ",".join("?" for _ in ids)
                table = self._validated_table_name()
                id_col = self._validated_id_column()
                # nosec B608 — table/_id_col from _validated_table/id_column (frozenset-checked);
                # placeholders is "?,?,?" only — all values bound as ?.
                sql = (
                    f"SELECT * FROM {table} "
                    f"WHERE {id_col} IN ({placeholders}) "
                    f"AND tenant_id=? AND owner_type=? AND owner_id=? "
                    f"AND quarantined_at IS NULL"
                )
                params: List[Any] = list(ids) + [tenant_id, owner_type, owner_id]
                rows = self._query_all(conn, sql, params, log_context)
                return {r[id_col]: r for r in rows}
            finally:
                conn.close()

        try:
            row_map = await self._run_sync(_sync)
        except Exception:
            logger.exception("%s SQL fetch failed%s", self.__class__.__name__, ctx)
            return []

        ordered = []
        for sid in ids:
            row = row_map.get(sid)
            if not row:
                continue
            obj = self._row_to_object(row)
            ordered.append(self._postprocess_row(obj))

        return ordered
    # ------------------------------------------------------------------ #
    # COMPLETE RETRIEVAL PIPELINE
    # ------------------------------------------------------------------ #

    async def _semantic_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        extra_filters: Optional[Dict[str, Any]] = None,
        log_context: str = "",
    ) -> List[Any]:
        """
        Semantic search pipeline:

            vector → ranked IDs → SQL rows → objects

        This is the primary method used by:
            EpisodicSQLStore.search()
            SemanticSQLStore.search()
            ProceduralSQLStore.search()

        Parameters
        ----------
        query_embedding : List[float]
        k : int
        tenant_id, owner_type, owner_id : str
            Required isolation scope (C1).
        extra_filters : Optional[Dict[str, Any]]
            Non-isolation predicates.
        log_context : str

        Returns
        -------
        List[Any]
            Ranked list of objects.
        """
        id_score_pairs = await self._vector_search_ids(
            query_embedding=query_embedding,
            k=k,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            extra_filters=extra_filters,
            log_context=log_context,
        )

        if not id_score_pairs:
            return []

        ids = [sid for sid, _score in id_score_pairs]
        items = await self._fetch_ranked_rows_by_ids(
            ids=ids,
            log_context=log_context,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
        )
        self._attach_vector_scores(items, id_score_pairs)
        return items

    # ------------------------------------------------------------------ #
    # Optional "IDs first" public helpers (for hybrid retrieval)
    # ------------------------------------------------------------------ #

    async def search_ids(
        self,
        query_embedding: List[float],
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        k: int = 10,
        extra_filters: Optional[Dict[str, Any]] = None,
        log_context: str = "",
    ) -> List[Tuple[str, float]]:
        """
        Return ranked (id, score) pairs for a vector query (no SQL fetch).

        This is an optional optimization to enable "IDs+scores first" retrieval.
        Isolation scope (tenant_id, owner_type, owner_id) is mandatory.
        """
        if not (isinstance(tenant_id, str) and tenant_id.strip()):
            raise ValueError(
                f"{self.__class__.__name__}.search_ids requires non-empty tenant_id"
            )
        if not (isinstance(owner_type, str) and owner_type.strip()):
            raise ValueError(
                f"{self.__class__.__name__}.search_ids requires non-empty owner_type"
            )
        if not (isinstance(owner_id, str) and owner_id.strip()):
            raise ValueError(
                f"{self.__class__.__name__}.search_ids requires non-empty owner_id"
            )
        return await self._vector_search_ids(
            query_embedding=query_embedding,
            k=k,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
            extra_filters=extra_filters,
            log_context=log_context,
        )

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        log_context: str = "",
    ) -> List[Any]:
        """
        Fetch rows by IDs in ranked order (SQL authoritative payload).
        """
        if not owner_type or not owner_id:
            logger.error(
                "%s fetch_by_ids requires owner_type and owner_id%s",
                self.__class__.__name__,
                f" [{log_context}]" if log_context else "",
            )
            raise ValueError(f"{self.__class__.__name__} fetch_by_ids requires owner_type and owner_id")
        return await self._fetch_ranked_rows_by_ids(
            ids=ids,
            log_context=log_context,
            tenant_id=tenant_id,
            owner_type=owner_type,
            owner_id=owner_id,
        )
    # ------------------------------------------------------------------ #
    # QUARANTINE MANAGEMENT (shared across all vector-SQL stores)
    # ------------------------------------------------------------------ #

    def _require_scope(
        self,
        tenant_id: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
    ) -> None:
        """Validate that all three scope fields are present; raise ValueError if not."""
        if not tenant_id or not owner_type or not owner_id:
            logger.error(
                "%s requires tenant_id, owner_type and owner_id",
                self.__class__.__name__,
            )
            raise ValueError(
                f"{self.__class__.__name__} requires tenant_id, owner_type and owner_id"
            )

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
        self._require_scope(tenant_id, owner_type, owner_id)

        def _sync() -> bool:
            conn = self._conn()
            try:
                row = self._query_one(
                    conn,
                    f"SELECT meta FROM {self._validated_table_name()} WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",  # nosec B608 — table name from frozenset-validated property
                    params=[record_id, tenant_id, owner_type, owner_id],
                    log_context=f"quarantine_{self._table_name}_fetch",
                )
                if row is None:
                    return False
                try:
                    meta = json.loads(row["meta"]) if row["meta"] else {}
                except Exception:
                    meta = {}
                meta.setdefault("security", {}).setdefault("audit_log", []).append(audit_entry)
                self._execute(
                    conn,
                    f"UPDATE {self._validated_table_name()} SET quarantined_at=?, meta=? WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",  # nosec B608 — table name from frozenset-validated property
                    params=[quarantined_at, json.dumps(meta), record_id, tenant_id, owner_type, owner_id],
                    log_context=f"quarantine_{self._table_name}",
                )
                conn.commit()
                logger.info(
                    "%s: quarantined record id=%s owner=%s:%s",
                    self.__class__.__name__, record_id, owner_type, owner_id,
                )
                return True
            except Exception:
                self._safe_rollback(conn, f"quarantine_{self._table_name}")
                logger.exception("%s.quarantine_record failed id=%s", self.__class__.__name__, record_id)
                raise
            finally:
                conn.close()

        return await self._run_sync(_sync)

    async def reinstate_quarantined_record(
        self,
        record_id: str,
        *,
        tenant_id: Optional[str],
        owner_type: str,
        owner_id: str,
        audit_entry: Dict[str, Any],
    ) -> bool:
        """Clear quarantined_at and append an audit log entry. Returns True if updated."""
        self._require_scope(tenant_id, owner_type, owner_id)

        def _sync() -> bool:
            conn = self._conn()
            try:
                row = self._query_one(
                    conn,
                    f"SELECT meta FROM {self._validated_table_name()} WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",  # nosec B608 — table name from frozenset-validated property
                    params=[record_id, tenant_id, owner_type, owner_id],
                    log_context=f"reinstate_{self._table_name}_fetch",
                )
                if row is None:
                    return False
                try:
                    meta = json.loads(row["meta"]) if row["meta"] else {}
                except Exception:
                    meta = {}
                meta.setdefault("security", {}).setdefault("audit_log", []).append(audit_entry)
                self._execute(
                    conn,
                    f"UPDATE {self._validated_table_name()} SET quarantined_at=NULL, meta=? WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",  # nosec B608 — table name from frozenset-validated property
                    params=[json.dumps(meta), record_id, tenant_id, owner_type, owner_id],
                    log_context=f"reinstate_{self._table_name}",
                )
                conn.commit()
                logger.info(
                    "%s: reinstated record id=%s owner=%s:%s",
                    self.__class__.__name__, record_id, owner_type, owner_id,
                )
                return True
            except Exception:
                self._safe_rollback(conn, f"reinstate_{self._table_name}")
                logger.exception("%s.reinstate_quarantined_record failed id=%s", self.__class__.__name__, record_id)
                raise
            finally:
                conn.close()

        return await self._run_sync(_sync)
