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

    @property
    def _table_name(self) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} must define _table_name"
        )

    @property
    def _id_column(self) -> str:
        """Return the name of the primary key column."""
        return "id"

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
        filters: Optional[Dict[str, Any]] = None,
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
        filters : Optional[Dict[str, Any]]
            Metadata-based filtering supported by the Index.
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
                k=k,
                filters=filters,
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
        if not ids:
            return []
        if not owner_type or not owner_id:
            logger.error(
                "%s _fetch_ranked_rows_by_ids requires owner_type and owner_id%s",
                self.__class__.__name__,
                f" [{log_context}]" if log_context else "",
            )
            raise ValueError(f"{self.__class__.__name__} fetch_by_ids requires owner_type and owner_id")

        ctx = f" [{log_context}]" if log_context else ""
        conn = self._conn()

        try:
            row_map = {}
            placeholders = ",".join("?" for _ in ids)
            sql = f"SELECT * FROM {self._table_name} WHERE {self._id_column} IN ({placeholders})"
            params: List[Any] = list(ids)

            if tenant_id:
                sql += " AND tenant_id=?"
                params.append(tenant_id)
            if owner_type:
                sql += " AND owner_type=?"
                params.append(owner_type)
            if owner_id:
                sql += " AND owner_id=?"
                params.append(owner_id)

            rows = self._query_all(conn, sql, params, log_context)
            row_map.update({r[self._id_column]: r for r in rows})

        except Exception:
            logger.exception("%s SQL fetch failed%s", self.__class__.__name__, ctx)
            return []
        finally:
            conn.close()

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
        filters: Optional[Dict[str, Any]] = None,
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
        filters : Optional[Dict[str, Any]]
        log_context : str

        Returns
        -------
        List[Any]
            Ranked list of objects.
        """
        id_score_pairs = await self._vector_search_ids(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            log_context=log_context,
        )

        if not id_score_pairs:
            return []

        ids = [sid for sid, _score in id_score_pairs]
        items = await self._fetch_ranked_rows_by_ids(
            ids=ids,
            log_context=log_context,
            tenant_id=filters.get("tenant_id") if filters else None,
            owner_type=filters.get("owner_type") if filters else None,
            owner_id=filters.get("owner_id") if filters else None,
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
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        log_context: str = "",
    ) -> List[Tuple[str, float]]:
        """
        Return ranked (id, score) pairs for a vector query (no SQL fetch).

        This is an optional optimization to enable "IDs+scores first" retrieval.
        """
        if (
            not filters
            or not filters.get("tenant_id")
            or not filters.get("owner_type")
            or not filters.get("owner_id")
        ):
            logger.error(
                "%s search_ids requires tenant_id, owner_type and owner_id%s",
                self.__class__.__name__,
                f" [{log_context}]" if log_context else "",
            )
            raise ValueError(
                f"{self.__class__.__name__} search_ids requires tenant_id, owner_type and owner_id"
            )
        return await self._vector_search_ids(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
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
