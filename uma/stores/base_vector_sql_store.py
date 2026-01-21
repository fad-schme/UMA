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
    ) -> List[Any]:
        """
        Fetch SQL rows for the given list of IDs, preserving ranking order.

        Parameters
        ----------
        ids : List[str]
            Ranked list of primary keys, as returned by ANN search.
        log_context : str
            Label used in logs for context.

        Returns
        -------
        List[Any]
            Domain model objects in ANN ranking order.
        """
        if not ids:
            return []

        ctx = f" [{log_context}]" if log_context else ""
        conn = self._conn()

        chunk_size = 500
        if len(ids) > chunk_size:
            logger.warning(
                "%s Large ranked ID list length=%d%s; chunking to %d.",
                self.__class__.__name__,
                len(ids),
                ctx,
                chunk_size,
            )

        try:
            row_map = {}
            for i in range(0, len(ids), chunk_size):
                chunk = ids[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                sql = (
                    f"SELECT * FROM {self._table_name} "
                    f"WHERE {self._id_column} IN ({placeholders})"
                )
                rows = self._query_all(
                    conn,
                    sql,
                    chunk,
                    log_context=log_context,
                )
                row_map.update({r[self._id_column]: r for r in rows})
        except Exception:
            logger.exception(
                "%s SQL fetch by IDs failed%s",
                self.__class__.__name__, ctx
            )
            return []
        finally:
            conn.close()

        # Preserve ranking order, warn on missing rows
        ordered = []
        for sid in ids:
            row = row_map.get(sid)
            if row is None:
                logger.debug(
                    "%s Missing SQL row for id=%s%s (stale vector index?)",
                    self.__class__.__name__, sid, ctx
                )
                continue

            obj = self._row_to_object(row)
            obj = self._postprocess_row(obj)
            ordered.append(obj)

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
        ids = await self._vector_search_ids(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            log_context=log_context,
        )

        if not ids:
            return []

        return await self._fetch_ranked_rows_by_ids(
            ids=ids,
            log_context=log_context,
        )
