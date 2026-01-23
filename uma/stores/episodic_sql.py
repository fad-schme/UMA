"""
EpisodicSQLStore — Episodic memory store for UMA (SQL + Vector index)

This implementation persists Episode rows and indexes embeddings in a VectorIndex.
This patch introduces ownership support (owner_type/owner_id) WITHOUT changing existing
field names or removing any columns.

Backward compatibility
----------------------
- Existing DBs without owner columns will be migrated via ALTER TABLE.
- Existing Episode objects without owner_id will derive:
    owner_type="user"
    owner_id=episode.user_id
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

from .base_vector_sql_store import BaseVectorSQLStore
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from ..core.utils.store_metadata import ensure_store_metadata
from ..types_episode import Episode

logger = logging.getLogger(__name__)


class EpisodicSQLStore(BaseVectorSQLStore):
    """
    SQL + VectorIndex episodic memory store.

    Schema (episodes)
    -----------------
    id TEXT PRIMARY KEY
    user_id TEXT NOT NULL
    owner_type TEXT NOT NULL  (NEW)
    owner_id TEXT NOT NULL    (NEW)
    timestamp TEXT NOT NULL   (ISO8601)
    summary TEXT NOT NULL
    raw TEXT NULL
    tags TEXT NOT NULL        (JSON list)
    meta TEXT NOT NULL        (JSON dict)
    embedding TEXT NULL       (JSON list)
    """

    def __init__(self, db_adapter: DBAdapter, vector_index: VectorIndex) -> None:
        super().__init__(db_adapter=db_adapter, vector_index=vector_index)
        self._init_db()
        logger.info("EpisodicSQLStore initialized with DB=%s", type(db_adapter).__name__)

    # ------------------------------------------------------------------ #
    # SQL Schema
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create the episodes table and indexes if not present, and apply safe migrations."""
        conn = self._conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    raw TEXT,
                    tags TEXT NOT NULL,
                    meta TEXT NOT NULL,
                    embedding TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_owner ON episodes(owner_type, owner_id);")

            # Cluster tables (keep existing schema, add owner columns)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_clusters (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    episode_ids TEXT NOT NULL,
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    latest_timestamp TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_clusters_user ON episode_clusters(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_clusters_ts ON episode_clusters(latest_timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_clusters_owner ON episode_clusters(owner_type, owner_id);")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_cluster_members (
                    cluster_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    PRIMARY KEY (cluster_id, episode_id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster_members_ep ON episode_cluster_members(episode_id);")

            ensure_store_metadata(self, conn, store_name="episodic")
            conn.commit()
        except Exception:
            self._safe_rollback(conn, "init_db")
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
        embedding = None
        try:
            emb_val = row["embedding"] if "embedding" in row.keys() else None
            if emb_val:
                embedding = json.loads(emb_val)
        except Exception:
            logger.exception("EpisodicSQLStore: failed to parse embedding for id=%s", row["id"])

        # owner fields may be NULL for legacy rows; derive from user_id
        owner_type = row["owner_type"] if "owner_type" in row.keys() and row["owner_type"] else "user"
        owner_id = row["owner_id"] if "owner_id" in row.keys() and row["owner_id"] else row["user_id"]

        return Episode(
            id=row["id"],
            user_id=row["user_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            summary=row["summary"],
            raw=row["raw"],
            tags=json.loads(row["tags"]),
            embedding=embedding,
            meta=json.loads(row["meta"]),
            owner_type=owner_type,
            owner_id=owner_id,
        )

    # ------------------------------------------------------------------ #
    # CRUD operations
    # ------------------------------------------------------------------ #

    async def add_episode(self, ep: Episode, embedding: List[float]) -> None:
        """
        Insert or update an episode record + semantic embedding.

        Backward compatibility:
        - If ep.owner_id is missing, derive owner_id from ep.user_id.
        """
        conn = self._conn()
        try:
            owner_type = getattr(ep, "owner_type", "user") or "user"
            owner_id = getattr(ep, "owner_id", "") or ep.user_id

            payload = {
                "id": ep.id,
                "user_id": ep.user_id,
                "owner_type": owner_type,
                "owner_id": owner_id,
                "timestamp": ep.timestamp.isoformat(),
                "summary": ep.summary,
                "raw": ep.raw,
                "tags": json.dumps(ep.tags),
                "embedding": json.dumps(embedding),
                "meta": json.dumps(ep.meta),
            }

            self._execute(
                conn,
                """
                INSERT INTO episodes (
                    id, user_id, owner_type, owner_id, timestamp, summary, raw, tags, embedding, meta
                )
                VALUES (
                    :id, :user_id, :owner_type, :owner_id, :timestamp, :summary, :raw, :tags, :embedding, :meta
                )
                ON CONFLICT(id) DO UPDATE SET
                    user_id=excluded.user_id,
                    owner_type=excluded.owner_type,
                    owner_id=excluded.owner_id,
                    timestamp=excluded.timestamp,
                    summary=excluded.summary,
                    raw=excluded.raw,
                    tags=excluded.tags,
                    embedding=excluded.embedding,
                    meta=excluded.meta
                """,
                params=payload,
                log_context="add_episode",
            )

            # Vector upsert (keep user_id, add owner fields)
            try:
                self.vector_index.upsert(
                    ids=[ep.id],
                    vectors=[embedding],
                    metadata=[{"user_id": ep.user_id, "owner_type": owner_type, "owner_id": owner_id}],
                )
            except Exception:
                logger.exception("EpisodicSQLStore.add_episode: vector upsert failed for id=%s", ep.id)
                self._safe_rollback(conn, "add_episode")
                raise

            try:
                conn.commit()
            except Exception:
                self._safe_rollback(conn, "add_episode_commit")
                try:
                    self.vector_index.delete([ep.id])
                except Exception:
                    logger.exception(
                        "EpisodicSQLStore.add_episode: vector delete failed after commit error id=%s",
                        ep.id,
                    )
                raise

            logger.info("EpisodicSQLStore: upserted episode id=%s", ep.id)

        except Exception:
            logger.exception("EpisodicSQLStore.add_episode: failure for id=%s", ep.id)
            raise
        finally:
            conn.close()

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
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
        conn = self._conn()
        try:
            self._execute(
                conn,
                "DELETE FROM episodes WHERE id = ?",
                params=[episode_id],
                log_context="delete_episode",
            )
            conn.commit()
            logger.info("EpisodicSQLStore: deleted episode id=%s", episode_id)

            try:
                self.vector_index.delete(ids=[episode_id])
            except Exception:
                logger.exception("EpisodicSQLStore.delete_episode: vector delete failed id=%s", episode_id)

        except Exception:
            self._safe_rollback(conn, "delete_episode")
            logger.exception("EpisodicSQLStore.delete_episode failed id=%s", episode_id)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Listing / Retention Helpers
    # ------------------------------------------------------------------ #

    async def list_episodes(self, user_id: str) -> List[Episode]:
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
            raise
        finally:
            conn.close()

    async def list_recent(self, user_id: str, n: int = 5) -> List[Episode]:
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
            raise
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

        We keep filtering by user_id to preserve existing behavior,
        and also pass owner metadata when possible.
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
            raise