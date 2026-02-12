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
from typing import Any, Dict, List, Optional, Sequence

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
        *,
        id_prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Run a vector search and return a ranked list of IDs.

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
        List[str]
            Ordered list of object IDs.
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
        valid_ids = []
        for pair in id_score_pairs:
            try:
                sid, _ = pair
                if isinstance(sid, str):
                    if id_prefix is None or sid.startswith(id_prefix):
                        valid_ids.append(sid)
                else:
                    logger.warning(
                        "%s Invalid vector search result element=%r%s",
                        self.__class__.__name__, pair, ctx
                    )
            except Exception:
                logger.exception(
                    "%s Malformed vector search result element=%r%s",
                    self.__class__.__name__, pair, ctx
                )

        return valid_ids

    # ------------------------------------------------------------------ #
    # Shared SQL lookup after vector ID retrieval
    # ------------------------------------------------------------------ #

    async def _fetch_ranked_rows_by_ids(
        self,
        ids: List[str],
        log_context: str = "",
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
        *,
        id_prefix: Optional[str] = None,
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
        ids = await self._vector_search_ids(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            log_context=log_context,
            id_prefix=id_prefix,
        )

        if not ids:
            return []

        return await self._fetch_ranked_rows_by_ids(
            ids=ids,
            log_context=log_context,
            owner_type=filters.get("owner_type") if filters else None,
            owner_id=filters.get("owner_id") if filters else None,
        )

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
        id_prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Return ranked IDs for a vector query (no SQL fetch).

        This is an optional optimization to enable "IDs+scores first" retrieval.
        """
        if not filters or not filters.get("owner_type") or not filters.get("owner_id"):
            logger.error(
                "%s search_ids requires owner_type and owner_id%s",
                self.__class__.__name__,
                f" [{log_context}]" if log_context else "",
            )
            raise ValueError(f"{self.__class__.__name__} search_ids requires owner_type and owner_id")
        return await self._vector_search_ids(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            log_context=log_context,
            id_prefix=id_prefix,
        )

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        log_context: str = "",
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
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
            owner_type=owner_type,
            owner_id=owner_id,
        )
