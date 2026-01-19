"""
EpisodicSQLStore — Episodic memory store for UMA-3 (SQL + Vector index)

This refactored implementation inherits from BaseVectorSQLStore to avoid
repetition of SQL connection logic, vector retrieval boilerplate, and
ranking preservation logic.

Responsibilities
----------------
- Persist Episode objects in a DB-agnostic SQL database (via DBAdapter).
- Store semantic embeddings in a VectorIndex (FAISS, Pinecone, Weaviate, etc.).
- Support upsert, get-by-id, delete, and semantic search of episodes.

Design notes
------------
- This store contains *only* domain logic specific to episodes.
- DB operations use BaseSQLStore helpers.
- Vector retrieval uses BaseVectorSQLStore._semantic_search().
- The store is 100% backend-agnostic:
  → Any DB supported by a DBAdapter
  → Any vector backend implementing VectorIndex
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

from .base_vector_sql_store import BaseVectorSQLStore
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from ..types_episode import Episode

logger = logging.getLogger(__name__)


class EpisodicSQLStore(BaseVectorSQLStore):
    """
    SQL + VectorIndex episodic memory store.

    Schema (episodes)
    -----------------
    id TEXT PRIMARY KEY
    user_id TEXT NOT NULL
    timestamp TEXT NOT NULL       (ISO8601)
    summary TEXT NOT NULL
    raw TEXT NULL
    tags TEXT NOT NULL            (JSON list)
    meta TEXT NOT NULL            (JSON dict)
    """

    def __init__(self, db_adapter: DBAdapter, vector_index: VectorIndex) -> None:
        """
        Initialize EpisodicSQLStore.

        Parameters
        ----------
        db_adapter : DBAdapter
            Database adapter returning DB-API compatible connections.
        vector_index : VectorIndex
            Pluggable embedding index backend.
        """
        super().__init__(db_adapter=db_adapter, vector_index=vector_index)
        self._init_db()
        logger.info("EpisodicSQLStore initialized with DB=%s", type(db_adapter).__name__)

    # ------------------------------------------------------------------ #
    # SQL Schema
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create the episodes table and indexes if not present."""
        conn = self._conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    raw TEXT,
                    tags TEXT NOT NULL,
                    meta TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);")
            conn.commit()
        except Exception:
            logger.exception("EpisodicSQLStore: failed initializing schema.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Required BaseVectorSQLStore properties
    # ------------------------------------------------------------------ #

    @property
    def _table_name(self) -> str:
        return "episodes"

    @property
    def _id_column(self) -> str:
        return "id"

    # ------------------------------------------------------------------ #
    # Row → Episode conversion
    # ------------------------------------------------------------------ #

    def _row_to_object(self, row) -> Episode:
        """
        Convert a DB row into an Episode instance.

        Parameters
        ----------
        row : DB row (e.g., sqlite3.Row)

        Returns
        -------
        Episode
        """
        return Episode(
            id=row["id"],
            user_id=row["user_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            summary=row["summary"],
            raw=row["raw"],
            tags=json.loads(row["tags"]),
            meta=json.loads(row["meta"]),
        )

    # ------------------------------------------------------------------ #
    # CRUD operations
    # ------------------------------------------------------------------ #

    async def add_episode(self, ep: Episode, embedding: List[float]) -> None:
        """
        Insert or update an episode record + semantic embedding.

        Parameters
        ----------
        ep : Episode
            Episode domain model.
        embedding : List[float]
            Embedding vector used to index and retrieve this episode.
        """
        conn = self._conn()
        try:
            payload = {
                "id": ep.id,
                "user_id": ep.user_id,
                "timestamp": ep.timestamp.isoformat(),
                "summary": ep.summary,
                "raw": ep.raw,
                "tags": json.dumps(ep.tags),
                "meta": json.dumps(ep.meta),
            }

            self._execute(
                conn,
                """
                INSERT INTO episodes (
                    id, user_id, timestamp, summary, raw, tags, meta
                )
                VALUES (
                    :id, :user_id, :timestamp, :summary, :raw, :tags, :meta
                )
                ON CONFLICT(id) DO UPDATE SET
                    user_id=excluded.user_id,
                    timestamp=excluded.timestamp,
                    summary=excluded.summary,
                    raw=excluded.raw,
                    tags=excluded.tags,
                    meta=excluded.meta
                """,
                params=payload,
                log_context="add_episode",
            )
            conn.commit()

            # Insert or update vector embedding
            try:
                self.vector_index.upsert(
                    ids=[ep.id],
                    vectors=[embedding],
                    metadata=[{"user_id": ep.user_id}],
                )
            except Exception:
                logger.exception(
                    "EpisodicSQLStore.add_episode: vector upsert failed for id=%s",
                    ep.id,
                )

            logger.info("EpisodicSQLStore: upserted episode id=%s", ep.id)

        except Exception:
            logger.exception("EpisodicSQLStore.add_episode: failure for id=%s", ep.id)
            raise
        finally:
            conn.close()

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """
        Retrieve a specific episode by ID.

        Parameters
        ----------
        episode_id : str

        Returns
        -------
        Optional[Episode]
        """
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT * FROM episodes WHERE id = ?",
                params=[episode_id],
                log_context="get_episode",
            )
            return self._row_to_object(row) if row else None
        except Exception:
            logger.exception("EpisodicSQLStore.get_episode failed id=%s", episode_id)
            return None
        finally:
            conn.close()

    async def delete_episode(self, episode_id: str) -> None:
        """
        Permanently delete an episode from SQL. Vector index removal
        should also occur in a full implementation (TODO for future).

        Parameters
        ----------
        episode_id : str
        """
        conn = self._conn()
        try:
            # SQL delete
            self._execute(
                conn,
                "DELETE FROM episodes WHERE id = ?",
                params=[episode_id],
                log_context="delete_episode",
            )
            conn.commit()
            logger.info("EpisodicSQLStore: deleted episode id=%s", episode_id)

            # Vector index delete
            try:
                self.vector_index.delete(ids=[episode_id])
            except Exception:
                logger.exception(
                    "EpisodicSQLStore.delete_episode: vector index delete failed id=%s",
                    episode_id,
                )

        except Exception:
            logger.exception("EpisodicSQLStore.delete_episode failed id=%s", episode_id)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Listing / Retention Helpers
    # ------------------------------------------------------------------ #

    async def list_episodes(self, user_id: str) -> List[Episode]:
        """
        Return all episodes for a given user (unsorted).

        Intended for retention policies and consolidation logic.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        List[Episode]
        """
        conn = self._conn()
        try:
            rows = self._query_all(
                conn,
                "SELECT * FROM episodes WHERE user_id = ?",
                params=[user_id],
                log_context="list_episodes",
            )
            return [self._row_to_object(row) for row in rows]
        except Exception:
            logger.exception("EpisodicSQLStore.list_episodes failed user_id=%s", user_id)
            return []
        finally:
            conn.close()

    async def list_recent(self, user_id: str, n: int = 5) -> List[Episode]:
        """
        Return the N most recent episodes for a user, sorted descending by timestamp.

        Parameters
        ----------
        user_id : str
        n : int, default=5

        Returns
        -------
        List[Episode]
        """
        conn = self._conn()
        try:
            rows = self._query_all(
                conn,
                """
                SELECT * FROM episodes
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                params=[user_id, n],
                log_context="list_recent",
            )
            return [self._row_to_object(row) for row in rows]
        except Exception:
            logger.exception("EpisodicSQLStore.list_recent failed user_id=%s", user_id)
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Semantic Search (vector → SQL → Episode)
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query_embedding: List[float],
        user_id: Optional[str] = None,
        k: int = 20,
    ) -> List[Episode]:
        """
        Semantic episodic search.

        Parameters
        ----------
        query_embedding : List[float]
            Embedding for query similarity.
        user_id : Optional[str], default=None
            If provided, filter to episodes belonging to this user.
        k : int, default=20
            Maximum number of results.

        Returns
        -------
        List[Episode]
            Ranked list of retrieved Episode objects.
        """
        filters = {"user_id": user_id} if user_id else None

        try:
            return await self._semantic_search(
                query_embedding=query_embedding,
                k=k,
                filters=filters,
                log_context="episodic_search",
            )
        except Exception:
            logger.exception("EpisodicSQLStore.search failed.")
            return []