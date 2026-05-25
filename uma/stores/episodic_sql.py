"""
EpisodicSQLStore — Episodic memory store for UMA (SQL + Vector index)

This implementation persists Episode rows and indexes embeddings in a VectorIndex.
This store is scoped by tenant + ownership.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_vector_sql_store import BaseVectorSQLStore
from .base_sql_store import DEFAULT_TENANT_ID
from ..adapters.db.base import DBAdapter
from ..adapters.vector.base import VectorIndex
from uma.stores.metadata import ensure_store_metadata
from uma.common.types import Episode, SCOPE_MODEL_VERSION
from uma.common.storage_metadata import normalize_episode_metadata

logger = logging.getLogger(__name__)


class EpisodicSQLStore(BaseVectorSQLStore):
    """
    SQL + VectorIndex episodic memory store.

    Schema (episodes)
    -----------------
    id TEXT PRIMARY KEY
    owner_type TEXT NOT NULL
    owner_id TEXT NOT NULL
    user_id TEXT NOT NULL
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
        logger.debug("EpisodicSQLStore initialized with DB=%s", type(db_adapter).__name__)

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
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT,
                    session_id TEXT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    raw TEXT,
                    tags TEXT NOT NULL,
                    origin_agent_id TEXT,
                    origin_user_id TEXT,
                    origin_session_id TEXT,
                    scope_model_version TEXT,
                    meta TEXT NOT NULL,
                    embedding TEXT
                );
                """
            )
            self._ensure_column(conn, "episodes", "tenant_id", "TEXT NOT NULL DEFAULT 'default'")
            self._ensure_column(conn, "episodes", "workspace_id", "TEXT")
            self._ensure_column(conn, "episodes", "session_id", "TEXT")
            self._ensure_column(conn, "episodes", "origin_agent_id", "TEXT")
            self._ensure_column(conn, "episodes", "origin_user_id", "TEXT")
            self._ensure_column(conn, "episodes", "origin_session_id", "TEXT")
            self._ensure_column(conn, "episodes", "scope_model_version", "TEXT")
            self._ensure_column(conn, "episodes", "trust_score", "REAL NOT NULL DEFAULT 0.5")
            self._ensure_column(conn, "episodes", "content_hash", "TEXT")
            self._ensure_column(conn, "episodes", "quarantined_at", "DATETIME")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_owner ON episodes(owner_type, owner_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_tenant_owner ON episodes(tenant_id, owner_type, owner_id);")

            # Cluster tables (keep existing schema, add owner columns)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_clusters (
                    id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    episode_ids TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT,
                    session_id TEXT,
                    user_id TEXT NOT NULL,
                    origin_agent_id TEXT,
                    origin_user_id TEXT,
                    origin_session_id TEXT,
                    scope_model_version TEXT,
                    latest_timestamp TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "episode_clusters", "tenant_id", "TEXT NOT NULL DEFAULT 'default'")
            self._ensure_column(conn, "episode_clusters", "workspace_id", "TEXT")
            self._ensure_column(conn, "episode_clusters", "session_id", "TEXT")
            self._ensure_column(conn, "episode_clusters", "origin_agent_id", "TEXT")
            self._ensure_column(conn, "episode_clusters", "origin_user_id", "TEXT")
            self._ensure_column(conn, "episode_clusters", "origin_session_id", "TEXT")
            self._ensure_column(conn, "episode_clusters", "scope_model_version", "TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_clusters_user ON episode_clusters(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_clusters_ts ON episode_clusters(latest_timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_clusters_owner ON episode_clusters(owner_type, owner_id);")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episode_clusters_tenant_owner ON episode_clusters(tenant_id, owner_type, owner_id);"
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

        owner_type = row["owner_type"]
        owner_id = row["owner_id"]
        user_id = row["user_id"] if "user_id" in row.keys() else owner_id

        normalized_meta = normalize_episode_metadata(
            json.loads(row["meta"]),
            episode_id=row["id"],
            owner_type=owner_type,
            owner_id=owner_id,
            timestamp=row["timestamp"],
            session_id=(row["session_id"] if "session_id" in row.keys() else None),
        )

        row_keys = row.keys() if hasattr(row, "keys") else []
        return Episode(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            summary=row["summary"],
            raw=row["raw"],
            tags=json.loads(row["tags"]),
            embedding=embedding,
            meta=normalized_meta,
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
            user_id=user_id,
        )

    # ------------------------------------------------------------------ #
    # CRUD operations
    # ------------------------------------------------------------------ #

    def _require_scope(
        self,
        tenant_id: Optional[str],
        owner_type: Optional[str],
        owner_id: Optional[str],
    ) -> None:
        if not tenant_id or not owner_type or not owner_id:
            logger.error("EpisodicSQLStore requires tenant_id, owner_type and owner_id")
            raise ValueError("EpisodicSQLStore requires tenant_id, owner_type and owner_id")

    async def add_episode(self, ep: Episode, embedding: List[float]) -> None:
        """
        Insert or update an episode record + semantic embedding.

        """
        conn = self._conn()
        try:
            owner_type = getattr(ep, "owner_type", "user") or "user"
            owner_id = getattr(ep, "owner_id", "")
            tenant_id = getattr(ep, "tenant_id", None) or DEFAULT_TENANT_ID
            if not owner_id:
                raise ValueError("EpisodicSQLStore.add_episode: owner_id must be set")
            if not getattr(ep, "user_id", None):
                raise ValueError("EpisodicSQLStore.add_episode: user_id must be set")

            # Idempotency guard: if turn_id is present, avoid duplicating episodes on retries.
            try:
                meta = getattr(ep, "meta", None) or {}
                turn_id = meta.get("turn_id")
                if turn_id:
                    existing = self._query_one(
                        conn,
                        """
                        SELECT id FROM episodes
                        WHERE tenant_id = ? AND owner_type = ? AND owner_id = ?
                          AND json_extract(meta, '$.turn_id') = ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                        """,
                        params=[tenant_id, owner_type, owner_id, str(turn_id)],
                        log_context="add_episode_idempotency",
                    )
                    if existing:
                        logger.info(
                            "EpisodicSQLStore.add_episode: skipping duplicate (turn_id=%s) existing_id=%s",
                            turn_id,
                            (existing["id"] if hasattr(existing, "__getitem__") else None),
                        )
                        return
            except Exception:
                # Never break ingestion due to idempotency guard issues.
                logger.exception("EpisodicSQLStore.add_episode: idempotency guard failed; continuing.")

            normalized_meta = normalize_episode_metadata(
                ep.meta,
                episode_id=ep.id,
                owner_type=owner_type,
                owner_id=owner_id,
                timestamp=ep.timestamp,
                session_id=getattr(ep, "session_id", None),
            )
            payload = {
                "id": ep.id,
                "tenant_id": getattr(ep, "tenant_id", None) or DEFAULT_TENANT_ID,
                "owner_type": owner_type,
                "owner_id": owner_id,
                "workspace_id": getattr(ep, "workspace_id", None),
                "session_id": getattr(ep, "session_id", None),
                "user_id": ep.user_id,
                "timestamp": ep.timestamp.isoformat(),
                "summary": ep.summary,
                "raw": ep.raw,
                "tags": json.dumps(ep.tags),
                "origin_agent_id": getattr(ep, "origin_agent_id", None),
                "origin_user_id": getattr(ep, "origin_user_id", None),
                "origin_session_id": getattr(ep, "origin_session_id", None),
                "scope_model_version": getattr(ep, "scope_model_version", None) or SCOPE_MODEL_VERSION,
                "trust_score": float(_ts if (_ts := getattr(ep, "trust_score", None)) is not None else 0.5),
                "content_hash": getattr(ep, "content_hash", None),
                "quarantined_at": (
                    getattr(ep, "quarantined_at").isoformat()
                    if getattr(ep, "quarantined_at", None) is not None
                    else None
                ),
                "embedding": json.dumps(embedding),
                "meta": json.dumps(normalized_meta),
            }

            self._execute(
                conn,
                """
                INSERT INTO episodes (
                    id, tenant_id, owner_type, owner_id, workspace_id, session_id,
                    user_id, timestamp, summary, raw, tags, origin_agent_id,
                    origin_user_id, origin_session_id, scope_model_version,
                    trust_score, content_hash, quarantined_at, embedding, meta
                )
                VALUES (
                    :id, :tenant_id, :owner_type, :owner_id, :workspace_id, :session_id,
                    :user_id, :timestamp, :summary, :raw, :tags, :origin_agent_id,
                    :origin_user_id, :origin_session_id, :scope_model_version,
                    :trust_score, :content_hash, :quarantined_at, :embedding, :meta
                )
                ON CONFLICT(id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    owner_type=excluded.owner_type,
                    owner_id=excluded.owner_id,
                    workspace_id=excluded.workspace_id,
                    session_id=excluded.session_id,
                    user_id=excluded.user_id,
                    timestamp=excluded.timestamp,
                    summary=excluded.summary,
                    raw=excluded.raw,
                    tags=excluded.tags,
                    origin_agent_id=excluded.origin_agent_id,
                    origin_user_id=excluded.origin_user_id,
                    origin_session_id=excluded.origin_session_id,
                    scope_model_version=excluded.scope_model_version,
                    trust_score=excluded.trust_score,
                    content_hash=excluded.content_hash,
                    quarantined_at=excluded.quarantined_at,
                    embedding=excluded.embedding,
                    meta=excluded.meta
                """,
                params=payload,
                log_context="add_episode",
            )

            # Vector upsert (owner-scoped)
            try:
                self.vector_index.upsert(
                    ids=[ep.id],
                    vectors=[embedding],
                    tenant_ids=[tenant_id],
                    owner_types=[owner_type],
                    owner_ids=[owner_id],
                    extra_metadata=[{
                        "kb_lane": normalized_meta.get("kb_lane"),
                    }],
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

    async def get_episode(
        self,
        episode_id: str,
        *,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Optional[Episode]:
        self._require_scope(tenant_id, owner_type, owner_id)
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT * FROM episodes WHERE id = ? AND tenant_id = ? AND owner_type = ? AND owner_id = ?",
                params=[episode_id, tenant_id, owner_type, owner_id],
                log_context="get_episode",
            )
            return self._row_to_object(row) if row else None
        except Exception:
            logger.exception("EpisodicSQLStore.get_episode failed id=%s", episode_id)
            raise
        finally:
            conn.close()

    async def delete_episode(
        self,
        episode_id: str,
        *,
        tenant_id: Optional[str] = None,
        owner_type: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> None:
        self._require_scope(tenant_id, owner_type, owner_id)
        conn = self._conn()
        try:
            self._execute(
                conn,
                "DELETE FROM episodes WHERE id = ? AND tenant_id = ? AND owner_type = ? AND owner_id = ?",
                params=[episode_id, tenant_id, owner_type, owner_id],
                log_context="delete_episode",
            )
            conn.commit()
            logger.info(
                "EpisodicSQLStore: deleted episode id=%s owner=%s:%s",
                episode_id,
                owner_type,
                owner_id,
            )

            try:
                self.vector_index.delete(ids=[episode_id])
            except Exception:
                logger.exception("EpisodicSQLStore.delete_episode: vector delete failed id=%s", episode_id)

        except Exception:
            self._safe_rollback(conn, "delete_episode")
            logger.exception(
                "EpisodicSQLStore.delete_episode failed id=%s owner=%s:%s",
                episode_id,
                owner_type,
                owner_id,
            )
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Listing / Retention Helpers
    # ------------------------------------------------------------------ #

    async def list_episodes(
        self,
        tenant_id: Optional[str] = None,
        owner_type: str = "",
        owner_id: str = "",
        include_quarantined: bool = False,
    ) -> List[Episode]:
        self._require_scope(tenant_id, owner_type, owner_id)
        conn = self._conn()
        try:
            quarantine_clause = "" if include_quarantined else " AND quarantined_at IS NULL"
            rows = self._query_all(
                conn,
                f"SELECT * FROM episodes WHERE tenant_id = ? AND owner_type = ? AND owner_id = ?{quarantine_clause}",
                params=[tenant_id, owner_type, owner_id],
                log_context="list_episodes",
            )
            logger.debug(
                "EpisodicSQLStore.list_episodes owner=%s:%s count=%d",
                owner_type,
                owner_id,
                len(rows or []),
            )
            return [self._row_to_object(row) for row in rows]
        except Exception:
            logger.exception(
                "EpisodicSQLStore.list_episodes failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            raise
        finally:
            conn.close()

    async def list_recent(self, tenant_id: Optional[str] = None, owner_type: str = "", owner_id: str = "", n: int = 5) -> List[Episode]:
        self._require_scope(tenant_id, owner_type, owner_id)
        conn = self._conn()
        try:
            rows = self._query_all(
                conn,
                """
                SELECT * FROM episodes
                WHERE tenant_id = ? AND owner_type = ? AND owner_id = ?
                  AND quarantined_at IS NULL
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                params=[tenant_id, owner_type, owner_id, n],
                log_context="list_recent",
            )
            logger.debug(
                "EpisodicSQLStore.list_recent owner=%s:%s count=%d",
                owner_type,
                owner_id,
                len(rows or []),
            )
            return [self._row_to_object(row) for row in rows]
        except Exception:
            logger.exception(
                "EpisodicSQLStore.list_recent failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            raise
        finally:
            conn.close()

    async def fetch_summaries(
        self,
        ids: List[str],
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
    ) -> List[dict]:
        """
        Fetch episode summaries by IDs (owner-scoped), preserving requested order.
        """
        if not ids:
            return []
        self._require_scope(tenant_id, owner_type, owner_id)

        conn = self._conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            sql = f"""
                SELECT id, user_id, timestamp, summary
                FROM episodes
                WHERE id IN ({placeholders})
                  AND tenant_id = ?
                  AND owner_type = ?
                  AND owner_id = ?
            """
            params = list(ids) + [tenant_id, owner_type, owner_id]
            rows = self._query_all(conn, sql, params=params, log_context="fetch_episode_summaries")
            row_map = {r["id"]: r for r in rows}
            ordered: List[dict] = []
            for eid in ids:
                row = row_map.get(eid)
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

    async def fetch_transcripts(
        self,
        ids: List[str],
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
    ) -> List[dict]:
        """
        Fetch episode transcripts (raw) by IDs (owner-scoped), preserving requested order.
        """
        if not ids:
            return []
        self._require_scope(tenant_id, owner_type, owner_id)

        conn = self._conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            sql = f"""
                SELECT id, user_id, timestamp, summary, raw
                FROM episodes
                WHERE id IN ({placeholders})
                  AND tenant_id = ?
                  AND owner_type = ?
                  AND owner_id = ?
            """
            params = list(ids) + [tenant_id, owner_type, owner_id]
            rows = self._query_all(conn, sql, params=params, log_context="fetch_episode_transcripts")
            row_map = {r["id"]: r for r in rows}
            ordered: List[dict] = []
            for eid in ids:
                row = row_map.get(eid)
                if row is None:
                    continue
                ordered.append(
                    {
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "timestamp": row["timestamp"],
                        "summary": row["summary"],
                        "raw": (row["raw"] if ("raw" in (row.keys() if hasattr(row, "keys") else [])) else None),
                    }
                )
            return ordered
        except Exception:
            logger.exception("EpisodicSQLStore.fetch_transcripts failed.")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Semantic Search (vector → SQL → Episode)
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query_embedding: List[float],
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
        k: int = 20,
        offset: int = 0,
    ) -> List[Episode]:
        """
        Semantic episodic search.

        Filter by owner_type/owner_id when provided.
        """
        self._require_scope(tenant_id, owner_type, owner_id)
        try:
            k_i = max(0, int(k))
        except Exception:
            k_i = 20
        try:
            offset_i = max(0, int(offset))
        except Exception:
            offset_i = 0
        if k_i <= 0:
            return []
        try:
            id_score_pairs = await self._vector_search_ids(
                query_embedding=query_embedding,
                k=k_i + offset_i,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
                log_context="episodic_search",
            )
            if not id_score_pairs:
                logger.debug(
                    "EpisodicSQLStore.search: vector candidates=0, sql_fetched=0, owner=%s:%s",
                    owner_type,
                    owner_id,
                )
                return []
            windowed_pairs = id_score_pairs[offset_i : offset_i + k_i]
            if not windowed_pairs:
                return []
            windowed_ids = [sid for sid, _score in windowed_pairs]
            episodes = await self.fetch_by_ids(
                windowed_ids,
                tenant_id=tenant_id,
                owner_type=owner_type,
                owner_id=owner_id,
            )
            self._attach_vector_scores(episodes, windowed_pairs)
            logger.debug(
                "EpisodicSQLStore.search: vector candidates=%d, sql_fetched=%d, owner=%s:%s",
                len(windowed_ids),
                len(episodes),
                owner_type,
                owner_id,
            )
            if windowed_ids and not episodes:
                logger.warning(
                    "EpisodicSQLStore.search: vector candidates=%d but SQL returned 0 op=search owner=%s:%s ids=%s",
                    len(windowed_ids),
                    owner_type,
                    owner_id,
                    windowed_ids[:3],
                )
            return episodes
        except Exception:
            logger.exception("EpisodicSQLStore.search failed.")
            raise

    async def fetch_by_ids(
        self,
        ids: List[str],
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
    ) -> List[Episode]:
        if not ids:
            return []
        self._require_scope(tenant_id, owner_type, owner_id)

        conn = self._conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            params: List[str] = list(ids) + [tenant_id, owner_type, owner_id]
            sql = f"SELECT * FROM episodes WHERE id IN ({placeholders}) AND tenant_id=? AND owner_type=? AND owner_id=? AND quarantined_at IS NULL"
            rows = self._query_all(conn, sql, params=params, log_context="fetch_episodes_by_ids")
            row_map = {r["id"]: r for r in rows}
            ordered: List[Episode] = []
            for eid in ids:
                row = row_map.get(eid)
                if row is None:
                    continue
                ordered.append(self._row_to_object(row))
            missing = max(0, len(ids) - len(ordered))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "EpisodicSQLStore.fetch_by_ids ids=%d fetched=%d owner=%s:%s",
                    len(ids),
                    len(ordered),
                    owner_type,
                    owner_id,
                )
            if missing:
                logger.warning(
                    "EpisodicSQLStore.fetch_by_ids missing=%d owner=%s:%s",
                    missing,
                    owner_type,
                    owner_id,
                )
            return ordered
        except Exception:
            logger.exception("EpisodicSQLStore.fetch_by_ids failed")
            raise
        finally:
            conn.close()

    async def upsert_cluster_summary(
        self,
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
        user_id: str,
        episode_ids: List[str],
        summary: str,
        latest_timestamp: str,
    ) -> None:
        self._require_scope(tenant_id, owner_type, owner_id)
        if not user_id:
            raise ValueError("EpisodicSQLStore.upsert_cluster_summary requires user_id")
        conn = self._conn()
        try:
            payload = {
                "id": f"cluster:{owner_type}:{owner_id}:{user_id}:{latest_timestamp}",
                "summary": summary,
                "episode_ids": json.dumps(episode_ids or []),
                "tenant_id": tenant_id,
                "owner_type": owner_type,
                "owner_id": owner_id,
                "workspace_id": None,
                "session_id": None,
                "user_id": user_id,
                "origin_agent_id": None,
                "origin_user_id": user_id,
                "origin_session_id": None,
                "scope_model_version": SCOPE_MODEL_VERSION,
                "latest_timestamp": latest_timestamp,
            }
            self._execute(
                conn,
                """
                INSERT INTO episode_clusters (
                    id, summary, episode_ids, tenant_id, owner_type, owner_id,
                    workspace_id, session_id, user_id, origin_agent_id, origin_user_id,
                    origin_session_id, scope_model_version, latest_timestamp
                ) VALUES (
                    :id, :summary, :episode_ids, :tenant_id, :owner_type, :owner_id,
                    :workspace_id, :session_id, :user_id, :origin_agent_id, :origin_user_id,
                    :origin_session_id, :scope_model_version, :latest_timestamp
                )
                ON CONFLICT(id) DO UPDATE SET
                    summary=excluded.summary,
                    episode_ids=excluded.episode_ids,
                    tenant_id=excluded.tenant_id,
                    owner_type=excluded.owner_type,
                    owner_id=excluded.owner_id,
                    workspace_id=excluded.workspace_id,
                    session_id=excluded.session_id,
                    user_id=excluded.user_id,
                    origin_agent_id=excluded.origin_agent_id,
                    origin_user_id=excluded.origin_user_id,
                    origin_session_id=excluded.origin_session_id,
                    scope_model_version=excluded.scope_model_version,
                    latest_timestamp=excluded.latest_timestamp
                """,
                params=payload,
                log_context="upsert_cluster_summary",
            )
            conn.commit()
            logger.debug(
                "EpisodicSQLStore.upsert_cluster_summary owner=%s:%s user_id=%s",
                owner_type,
                owner_id,
                user_id,
            )
        except Exception:
            self._safe_rollback(conn, "upsert_cluster_summary")
            logger.exception(
                "EpisodicSQLStore.upsert_cluster_summary failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            raise
        finally:
            conn.close()

    async def list_cluster_summaries(
        self,
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
        k: int = 5,
        max_episodes: Optional[int] = None,
        time_range: Optional[dict] = None,
    ) -> List[dict]:
        self._require_scope(tenant_id, owner_type, owner_id)
        conn = self._conn()
        try:
            where = ["tenant_id = ?", "owner_type = ?", "owner_id = ?"]
            params: List[Any] = [tenant_id, owner_type, owner_id]
            if isinstance(time_range, dict):
                start = time_range.get("start")
                end = time_range.get("end")
                if start is not None:
                    where.append("latest_timestamp >= ?")
                    params.append(str(start))
                if end is not None:
                    where.append("latest_timestamp <= ?")
                    params.append(str(end))
            sql = f"""
                SELECT * FROM episode_clusters
                WHERE {' AND '.join(where)}
                ORDER BY latest_timestamp DESC
                LIMIT ?
            """
            params.append(int(k))
            rows = self._query_all(conn, sql, params=params, log_context="list_cluster_summaries")
            out: List[dict] = []
            for row in rows or []:
                try:
                    episode_ids = json.loads(row["episode_ids"]) if row.get("episode_ids") else []
                except Exception:
                    episode_ids = []
                out.append(
                    {
                        "id": row["id"],
                        "owner_type": row["owner_type"],
                        "owner_id": row["owner_id"],
                        "user_id": row["user_id"],
                        "summary": row["summary"],
                        "episode_ids": episode_ids,
                        "latest_timestamp": row["latest_timestamp"],
                        "count": len(episode_ids),
                    }
                )
            logger.debug(
                "EpisodicSQLStore.list_cluster_summaries owner=%s:%s count=%d",
                owner_type,
                owner_id,
                len(out),
            )
            return out
        except Exception:
            logger.exception(
                "EpisodicSQLStore.list_cluster_summaries failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            raise
        finally:
            conn.close()

    async def get_cluster_members(
        self,
        *,
        tenant_id: Optional[str] = None,
        owner_type: str,
        owner_id: str,
        cluster_id: str,
    ) -> List[Episode]:
        self._require_scope(tenant_id, owner_type, owner_id)
        if not cluster_id:
            return []
        conn = self._conn()
        try:
            cluster = self._query_one(
                conn,
                """
                SELECT id FROM episode_clusters
                WHERE id = ? AND tenant_id = ? AND owner_type = ? AND owner_id = ?
                """,
                params=[cluster_id, tenant_id, owner_type, owner_id],
                log_context="get_cluster_members",
            )
            if not cluster:
                logger.warning(
                    "EpisodicSQLStore.get_cluster_members: cluster not found owner=%s:%s id=%s",
                    owner_type,
                    owner_id,
                    cluster_id,
                )
                return []
            rows = self._query_all(
                conn,
                """
                SELECT e.*
                FROM episode_cluster_members m
                JOIN episodes e ON e.id = m.episode_id
                WHERE m.cluster_id = ? AND e.tenant_id = ? AND e.owner_type = ? AND e.owner_id = ?
                ORDER BY e.timestamp DESC
                """,
                params=[cluster_id, tenant_id, owner_type, owner_id],
                log_context="get_cluster_members",
            )
            episodes = [self._row_to_object(r) for r in rows]
            logger.debug(
                "EpisodicSQLStore.get_cluster_members owner=%s:%s cluster=%s count=%d",
                owner_type,
                owner_id,
                cluster_id,
                len(episodes),
            )
            return episodes
        except Exception:
            logger.exception(
                "EpisodicSQLStore.get_cluster_members failed owner=%s:%s cluster=%s",
                owner_type,
                owner_id,
                cluster_id,
            )
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Quarantine Management
    # ------------------------------------------------------------------ #

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
            raise ValueError("EpisodicSQLStore.reinstate_quarantined_record requires scope")
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT meta FROM episodes WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[record_id, tenant_id, owner_type, owner_id],
                log_context="reinstate_episode_fetch",
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
                "UPDATE episodes SET quarantined_at=NULL, meta=? WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[json.dumps(meta), record_id, tenant_id, owner_type, owner_id],
                log_context="reinstate_episode",
            )
            conn.commit()
            logger.info("EpisodicSQLStore: reinstated episode id=%s owner=%s:%s", record_id, owner_type, owner_id)
            return True
        except Exception:
            self._safe_rollback(conn, "reinstate_episode")
            logger.exception("EpisodicSQLStore.reinstate_quarantined_record failed id=%s", record_id)
            raise
        finally:
            conn.close()

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
            raise ValueError("EpisodicSQLStore.quarantine_record requires scope")
        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                "SELECT meta FROM episodes WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[record_id, tenant_id, owner_type, owner_id],
                log_context="quarantine_episode_fetch",
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
                "UPDATE episodes SET quarantined_at=?, meta=? WHERE id=? AND tenant_id=? AND owner_type=? AND owner_id=?",
                params=[quarantined_at, json.dumps(meta), record_id, tenant_id, owner_type, owner_id],
                log_context="quarantine_episode",
            )
            conn.commit()
            logger.info("EpisodicSQLStore: quarantined episode id=%s owner=%s:%s", record_id, owner_type, owner_id)
            return True
        except Exception:
            self._safe_rollback(conn, "quarantine_episode")
            logger.exception("EpisodicSQLStore.quarantine_record failed id=%s", record_id)
            raise
        finally:
            conn.close()