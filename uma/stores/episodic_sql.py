"""
EpisodicSQLStore — Episodic memory store for UMA (SQL + Vector index)

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
    timestamp TEXT NOT NULL       (ISO8601)
    summary TEXT NOT NULL
    raw TEXT NULL
    tags TEXT NOT NULL            (JSON list)
    meta TEXT NOT NULL            (JSON dict)
    embedding TEXT NULL           (JSON list)
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
                    meta TEXT NOT NULL,
                    embedding TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);")

            # Backward-compatible migration for existing databases.
            try:
                conn.execute("ALTER TABLE episodes ADD COLUMN embedding TEXT;")
            except Exception:
                # Column likely exists; keep quiet to avoid noisy logs.
                pass

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_clusters (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    episode_ids TEXT NOT NULL,
                    latest_timestamp TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episode_clusters_user ON episode_clusters(user_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episode_clusters_ts ON episode_clusters(latest_timestamp);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_cluster_members (
                    cluster_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    PRIMARY KEY (cluster_id, episode_id)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cluster_members_ep ON episode_cluster_members(episode_id);"
            )
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
        """
        Convert a DB row into an Episode instance.

        Parameters
        ----------
        row : DB row (e.g., sqlite3.Row)

        Returns
        -------
        Episode
        """
        embedding = None
        try:
            if hasattr(row, "get"):
                emb_val = row.get("embedding")
            else:
                emb_val = row["embedding"] if "embedding" in row else None
            if emb_val:
                embedding = json.loads(emb_val)
        except Exception:
            logger.exception("EpisodicSQLStore: failed to parse embedding for id=%s", row["id"])

        return Episode(
            id=row["id"],
            user_id=row["user_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            summary=row["summary"],
            raw=row["raw"],
            tags=json.loads(row["tags"]),
            embedding=embedding,
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
                "embedding": json.dumps(embedding),
                "meta": json.dumps(ep.meta),
            }

            self._execute(
                conn,
                """
                INSERT INTO episodes (
                    id, user_id, timestamp, summary, raw, tags, embedding, meta
                )
                VALUES (
                    :id, :user_id, :timestamp, :summary, :raw, :tags, :embedding, :meta
                )
                ON CONFLICT(id) DO UPDATE SET
                    user_id=excluded.user_id,
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
        Permanently delete an episode from SQL and its vector index entry.

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
            # Prune cluster summaries by removing the episode_id; drop empty clusters.
            try:
                cluster_rows = self._query_all(
                    conn,
                    "SELECT cluster_id FROM episode_cluster_members WHERE episode_id = ?",
                    params=[episode_id],
                    log_context="list_cluster_members_for_prune",
                )
                cluster_ids = [r["cluster_id"] for r in cluster_rows]

                # Remove member rows for the episode.
                self._execute(
                    conn,
                    "DELETE FROM episode_cluster_members WHERE episode_id = ?",
                    params=[episode_id],
                    log_context="delete_cluster_member",
                )

                for cluster_id in cluster_ids:
                    try:
                        member_ids = self._cluster_member_ids(conn, cluster_id)
                        if not member_ids:
                            self._execute(
                                conn,
                                "DELETE FROM episode_clusters WHERE id = ?",
                                params=[cluster_id],
                                log_context="delete_empty_cluster",
                            )
                            continue
                        latest = self._latest_episode_snapshot(conn, member_ids)
                        if latest is None:
                            self._execute(
                                conn,
                                "DELETE FROM episode_clusters WHERE id = ?",
                                params=[cluster_id],
                                log_context="delete_cluster_no_snapshot",
                            )
                            continue
                        self._execute(
                            conn,
                            """
                            UPDATE episode_clusters
                            SET episode_ids = ?, summary = ?, latest_timestamp = ?
                            WHERE id = ?
                            """,
                            params=[
                                json.dumps(member_ids),
                                latest["summary"],
                                latest["timestamp"],
                                cluster_id,
                            ],
                            log_context="update_cluster_after_prune",
                        )
                    except Exception:
                        logger.exception(
                            "EpisodicSQLStore.delete_episode: cluster update failed cluster_id=%s",
                            cluster_id,
                        )
            except Exception:
                logger.exception(
                    "EpisodicSQLStore.delete_episode: cluster prune failed id=%s",
                    episode_id,
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
            self._safe_rollback(conn, "delete_episode")
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
            raise
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
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Fetch summaries / transcripts by IDs (snippet-first helpers)
    # ------------------------------------------------------------------ #

    async def fetch_summaries(self, ids: List[str]) -> List[dict]:
        """
        Fetch episode summaries by ID, preserving requested order.

        Returns a list of dicts: {id, user_id, timestamp, summary}
        """
        if not ids:
            return []

        conn = self._conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            rows = self._query_all(
                conn,
                f"""
                SELECT id, user_id, timestamp, summary
                FROM episodes
                WHERE id IN ({placeholders})
                """,
                params=ids,
                log_context="fetch_summaries",
            )
            row_map = {r["id"]: r for r in rows}
            ordered = []
            for sid in ids:
                row = row_map.get(sid)
                if row is None:
                    continue
                ordered.append(
                    {
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "timestamp": row["timestamp"],
                        "summary": row["summary"],
                    }
                )
            return ordered
        except Exception:
            logger.exception("EpisodicSQLStore.fetch_summaries failed.")
            raise
        finally:
            conn.close()

    async def fetch_transcripts(self, ids: List[str]) -> List[dict]:
        """
        Fetch episode transcripts by ID, preserving requested order.

        Returns a list of dicts: {id, user_id, timestamp, summary, raw}
        """
        if not ids:
            return []

        conn = self._conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            rows = self._query_all(
                conn,
                f"""
                SELECT id, user_id, timestamp, summary, raw
                FROM episodes
                WHERE id IN ({placeholders})
                """,
                params=ids,
                log_context="fetch_transcripts",
            )
            row_map = {r["id"]: r for r in rows}
            ordered = []
            for sid in ids:
                row = row_map.get(sid)
                if row is None:
                    continue
                ordered.append(
                    {
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "timestamp": row["timestamp"],
                        "summary": row["summary"],
                        "raw": row["raw"],
                    }
                )
            return ordered
        except Exception:
            logger.exception("EpisodicSQLStore.fetch_transcripts failed.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Cluster summaries (precomputed during consolidation)
    # ------------------------------------------------------------------ #

    async def upsert_cluster_summary(
        self,
        user_id: str,
        episode_ids: List[str],
        summary: str,
        latest_timestamp: str,
    ) -> None:
        """
        Upsert a cluster summary row keyed by a deterministic cluster id.
        """
        if not episode_ids or not summary:
            return

        cluster_id = self._cluster_id(episode_ids)
        conn = self._conn()
        try:
            payload = {
                "id": cluster_id,
                "user_id": user_id,
                "summary": summary,
                "episode_ids": json.dumps(episode_ids),
                "latest_timestamp": latest_timestamp,
            }
            self._execute(
                conn,
                """
                INSERT INTO episode_clusters (id, user_id, summary, episode_ids, latest_timestamp)
                VALUES (:id, :user_id, :summary, :episode_ids, :latest_timestamp)
                ON CONFLICT(id) DO UPDATE SET
                    user_id=excluded.user_id,
                    summary=excluded.summary,
                    episode_ids=excluded.episode_ids,
                    latest_timestamp=excluded.latest_timestamp
                """,
                params=payload,
                log_context="upsert_cluster_summary",
            )
            conn.commit()
            try:
                self._execute(
                    conn,
                    "DELETE FROM episode_cluster_members WHERE cluster_id = ?",
                    params=[cluster_id],
                    log_context="delete_cluster_members",
                )
                insert_sql = (
                    "INSERT OR IGNORE INTO episode_cluster_members (cluster_id, episode_id) "
                    "VALUES (?, ?)"
                    if getattr(self._db_adapter, "paramstyle", "qmark") == "qmark"
                    else "INSERT INTO episode_cluster_members (cluster_id, episode_id) "
                    "VALUES (?, ?) ON CONFLICT DO NOTHING"
                )
                self._executemany(
                    conn,
                    insert_sql,
                    [(cluster_id, eid) for eid in episode_ids],
                    log_context="insert_cluster_members",
                )
                conn.commit()
            except Exception:
                self._safe_rollback(conn, "upsert_cluster_members")
                logger.exception("EpisodicSQLStore: failed to update cluster members.")
            logger.debug("EpisodicSQLStore: upserted cluster id=%s", cluster_id)
        except Exception:
            self._safe_rollback(conn, "upsert_cluster_summary")
            logger.exception("EpisodicSQLStore.upsert_cluster_summary failed.")
        finally:
            conn.close()

    async def list_cluster_summaries(
        self,
        user_id: str,
        k: int = 5,
        time_range: Optional[dict] = None,
        max_episodes: Optional[int] = None,
    ) -> List[dict]:
        """
        Return precomputed cluster summaries for a user, most recent first.
        """
        conn = self._conn()
        try:
            sql = """
                SELECT * FROM episode_clusters
                WHERE user_id=?
            """
            params = [user_id]
            if time_range:
                start = time_range.get("start")
                end = time_range.get("end")
                if start:
                    sql += " AND latest_timestamp >= ?"
                    params.append(start.isoformat() if hasattr(start, "isoformat") else str(start))
                if end:
                    sql += " AND latest_timestamp <= ?"
                    params.append(end.isoformat() if hasattr(end, "isoformat") else str(end))
            sql += " ORDER BY latest_timestamp DESC LIMIT ?"
            params.append(int(k))

            rows = self._query_all(conn, sql, params=params, log_context="list_cluster_summaries")
            out: List[dict] = []
            for r in rows:
                episode_ids = json.loads(r["episode_ids"])
                trimmed_ids = (
                    episode_ids[: int(max_episodes)] if max_episodes else episode_ids
                )
                out.append(
                    {
                        "id": r["id"],
                        "user_id": r["user_id"],
                        "summary": r["summary"],
                        "episode_ids": trimmed_ids,
                        "latest_timestamp": r["latest_timestamp"],
                        "count": len(episode_ids),
                    }
                )
            return out
        except Exception:
            logger.exception("EpisodicSQLStore.list_cluster_summaries failed.")
            raise
        finally:
            conn.close()

    def _cluster_id(self, episode_ids: List[str]) -> str:
        import hashlib

        payload = "|".join(sorted(episode_ids))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return f"cluster:{digest}"

    def _cluster_member_ids(self, conn, cluster_id: str) -> List[str]:
        try:
            rows = self._query_all(
                conn,
                "SELECT episode_id FROM episode_cluster_members WHERE cluster_id = ?",
                params=[cluster_id],
                log_context="list_cluster_members",
            )
            return [r["episode_id"] for r in rows]
        except Exception:
            logger.exception("EpisodicSQLStore._cluster_member_ids failed.")
            raise

    def _latest_episode_snapshot(self, conn, episode_ids: List[str]) -> Optional[dict]:
        """
        Return the latest episode summary + timestamp for a set of episode IDs.
        """
        if not episode_ids:
            return None
        try:
            placeholders = ",".join("?" for _ in episode_ids)
            row = self._query_one(
                conn,
                f"""
                SELECT summary, timestamp FROM episodes
                WHERE id IN ({placeholders})
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                params=episode_ids,
                log_context="latest_episode_snapshot",
            )
            if not row:
                return None
            return {"summary": row["summary"], "timestamp": row["timestamp"]}
        except Exception:
            logger.exception("EpisodicSQLStore._latest_episode_snapshot failed.")
            raise
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
            raise
