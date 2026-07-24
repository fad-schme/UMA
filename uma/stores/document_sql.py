"""
DocumentSQLStore — SQL store for document manifests.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .base_sql_store import BaseSQLStore, DEFAULT_TENANT_ID
from ..adapters.db.base import DBAdapter
from uma.stores.metadata import ensure_store_metadata
from uma.common.types import SCOPE_MODEL_VERSION
from uma.common.storage_metadata import normalize_document_metadata

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    doc_id: str
    source_path: str
    source_hash: str
    ingested_at: datetime
    owner_type: str
    owner_id: str
    meta: dict
    tenant_id: str = DEFAULT_TENANT_ID
    workspace_id: Optional[str] = None
    origin_agent_id: Optional[str] = None
    origin_user_id: Optional[str] = None
    origin_session_id: Optional[str] = None
    scope_model_version: Optional[str] = None


class DocumentSQLStore(BaseSQLStore):
    def __init__(self, db_adapter: DBAdapter) -> None:
        super().__init__(db_adapter=db_adapter)
        self._init_db()
        logger.debug("DocumentSQLStore initialized.")

    def _init_db(self) -> None:
        conn = self._conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT,
                    origin_agent_id TEXT,
                    origin_user_id TEXT,
                    origin_session_id TEXT,
                    scope_model_version TEXT,
                    meta TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_type, owner_id);
                CREATE INDEX IF NOT EXISTS idx_documents_owner_hash ON documents(owner_type, owner_id, source_hash);
                """
            )
            self._ensure_column(conn, "documents", "tenant_id", "TEXT NOT NULL DEFAULT 'default'")
            self._ensure_column(conn, "documents", "workspace_id", "TEXT")
            self._ensure_column(conn, "documents", "origin_agent_id", "TEXT")
            self._ensure_column(conn, "documents", "origin_user_id", "TEXT")
            self._ensure_column(conn, "documents", "origin_session_id", "TEXT")
            self._ensure_column(conn, "documents", "scope_model_version", "TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_tenant_owner ON documents(tenant_id, owner_type, owner_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_tenant_owner_hash ON documents(tenant_id, owner_type, owner_id, source_hash);"
            )
            ensure_store_metadata(self, conn, store_name="documents")
            conn.commit()
        except Exception:
            self._safe_rollback(conn, "init_db")
            logger.exception("DocumentSQLStore: failed initializing schema.")
            raise
        finally:
            conn.close()

    async def get_by_owner_and_hash(
        self,
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        source_hash: str,
    ) -> DocumentRecord | None:
        """
        Return the most recently ingested document record for a tenant + owner + hash.

        tenant_id is required (DAT invariant). The lookup is scoped to a single
        tenant: two tenants with overlapping owner_id values will never see
        each other's manifests through this method. Earlier versions of this
        function omitted the tenant predicate; calling code that relied on
        that behavior was unsafe.

        This is used for idempotent ingestion gates.
        """
        if not tenant_id or not owner_type or not owner_id or not source_hash:
            return None

        def _sync():
            conn = self._conn()
            try:
                row = self._query_one(
                    conn,
                    """
                    SELECT
                        doc_id,
                        source_path,
                        source_hash,
                        ingested_at,
                        tenant_id,
                        owner_type,
                        owner_id,
                        workspace_id,
                        origin_agent_id,
                        origin_user_id,
                        origin_session_id,
                        scope_model_version,
                        meta
                    FROM documents
                    WHERE tenant_id = ? AND owner_type = ? AND owner_id = ? AND source_hash = ?
                    ORDER BY ingested_at DESC
                    LIMIT 1
                    """,
                    params=[tenant_id, owner_type, owner_id, source_hash],
                    log_context="documents_get_by_tenant_owner_hash",
                )
                if not row:
                    return None

                meta_raw = row.get("meta") if hasattr(row, "get") else row["meta"]
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw else {}
                except Exception:
                    meta = {}

                ingested_at_raw = row.get("ingested_at") if hasattr(row, "get") else row["ingested_at"]
                try:
                    ingested_at = (
                        datetime.fromisoformat(ingested_at_raw)
                        if isinstance(ingested_at_raw, str) and ingested_at_raw
                        else datetime.utcnow()
                    )
                except Exception:
                    ingested_at = datetime.utcnow()

                return DocumentRecord(
                    doc_id=str((row.get("doc_id") if hasattr(row, "get") else row["doc_id"]) or ""),
                    source_path=str((row.get("source_path") if hasattr(row, "get") else row["source_path"]) or ""),
                    source_hash=str((row.get("source_hash") if hasattr(row, "get") else row["source_hash"]) or ""),
                    ingested_at=ingested_at,
                    tenant_id=str((row.get("tenant_id") if hasattr(row, "get") else row["tenant_id"]) or DEFAULT_TENANT_ID),
                    owner_type=str((row.get("owner_type") if hasattr(row, "get") else row["owner_type"]) or ""),
                    owner_id=str((row.get("owner_id") if hasattr(row, "get") else row["owner_id"]) or ""),
                    workspace_id=(row.get("workspace_id") if hasattr(row, "get") else row["workspace_id"]),
                    origin_agent_id=(row.get("origin_agent_id") if hasattr(row, "get") else row["origin_agent_id"]),
                    origin_user_id=(row.get("origin_user_id") if hasattr(row, "get") else row["origin_user_id"]),
                    origin_session_id=(row.get("origin_session_id") if hasattr(row, "get") else row["origin_session_id"]),
                    scope_model_version=(row.get("scope_model_version") if hasattr(row, "get") else row["scope_model_version"]),
                    meta=normalize_document_metadata(
                        meta,
                        doc_id=str((row.get("doc_id") if hasattr(row, "get") else row["doc_id"]) or ""),
                        owner_type=str((row.get("owner_type") if hasattr(row, "get") else row["owner_type"]) or ""),
                        owner_id=str((row.get("owner_id") if hasattr(row, "get") else row["owner_id"]) or ""),
                        ingested_at=ingested_at,
                        source_path=str((row.get("source_path") if hasattr(row, "get") else row["source_path"]) or ""),
                        source_hash=str((row.get("source_hash") if hasattr(row, "get") else row["source_hash"]) or ""),
                    ),
                )
            except Exception:
                logger.exception(
                    "DocumentSQLStore.get_by_owner_and_hash failed tenant=%s owner=%s:%s",
                    tenant_id,
                    owner_type,
                    owner_id,
                )
                raise
            finally:
                conn.close()

        return await self._run_sync(_sync)
    async def get_latest_by_source_path(
        self,
        *,
        tenant_id: str,
        owner_type: str,
        owner_id: str,
        source_path: str,
    ) -> DocumentRecord | None:
        """
        Return the most recently ingested document record for a tenant + owner + source path.

        Source-path identity is used by the ingest manifest gate to detect a new
        version of an already-known source without broadening outside the
        current tenant/owner scope.
        """
        if not tenant_id or not owner_type or not owner_id or not source_path:
            return None

        def _sync():
            conn = self._conn()
            try:
                row = self._query_one(
                    conn,
                    """
                    SELECT
                        doc_id,
                        source_path,
                        source_hash,
                        ingested_at,
                        tenant_id,
                        owner_type,
                        owner_id,
                        workspace_id,
                        origin_agent_id,
                        origin_user_id,
                        origin_session_id,
                        scope_model_version,
                        meta
                    FROM documents
                    WHERE tenant_id = ? AND owner_type = ? AND owner_id = ? AND source_path = ?
                    ORDER BY ingested_at DESC
                    LIMIT 1
                    """,
                    params=[tenant_id, owner_type, owner_id, source_path],
                    log_context="documents_get_latest_by_source_path",
                )
                if not row:
                    return None

                meta_raw = row.get("meta") if hasattr(row, "get") else row["meta"]
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw else {}
                except Exception:
                    meta = {}

                ingested_at_raw = row.get("ingested_at") if hasattr(row, "get") else row["ingested_at"]
                try:
                    ingested_at = (
                        datetime.fromisoformat(ingested_at_raw)
                        if isinstance(ingested_at_raw, str) and ingested_at_raw
                        else datetime.utcnow()
                    )
                except Exception:
                    ingested_at = datetime.utcnow()

                return DocumentRecord(
                    doc_id=str((row.get("doc_id") if hasattr(row, "get") else row["doc_id"]) or ""),
                    source_path=str((row.get("source_path") if hasattr(row, "get") else row["source_path"]) or ""),
                    source_hash=str((row.get("source_hash") if hasattr(row, "get") else row["source_hash"]) or ""),
                    ingested_at=ingested_at,
                    tenant_id=str((row.get("tenant_id") if hasattr(row, "get") else row["tenant_id"]) or DEFAULT_TENANT_ID),
                    owner_type=str((row.get("owner_type") if hasattr(row, "get") else row["owner_type"]) or ""),
                    owner_id=str((row.get("owner_id") if hasattr(row, "get") else row["owner_id"]) or ""),
                    workspace_id=(row.get("workspace_id") if hasattr(row, "get") else row["workspace_id"]),
                    origin_agent_id=(row.get("origin_agent_id") if hasattr(row, "get") else row["origin_agent_id"]),
                    origin_user_id=(row.get("origin_user_id") if hasattr(row, "get") else row["origin_user_id"]),
                    origin_session_id=(row.get("origin_session_id") if hasattr(row, "get") else row["origin_session_id"]),
                    scope_model_version=(row.get("scope_model_version") if hasattr(row, "get") else row["scope_model_version"]),
                    meta=normalize_document_metadata(
                        meta,
                        doc_id=str((row.get("doc_id") if hasattr(row, "get") else row["doc_id"]) or ""),
                        owner_type=str((row.get("owner_type") if hasattr(row, "get") else row["owner_type"]) or ""),
                        owner_id=str((row.get("owner_id") if hasattr(row, "get") else row["owner_id"]) or ""),
                        ingested_at=ingested_at,
                        source_path=str((row.get("source_path") if hasattr(row, "get") else row["source_path"]) or ""),
                        source_hash=str((row.get("source_hash") if hasattr(row, "get") else row["source_hash"]) or ""),
                    ),
                )
            except Exception:
                logger.exception(
                    "DocumentSQLStore.get_latest_by_source_path failed tenant=%s owner=%s:%s path=%s",
                    tenant_id,
                    owner_type,
                    owner_id,
                    source_path,
                )
                raise
            finally:
                conn.close()

        return await self._run_sync(_sync)
    async def upsert_document(self, record: DocumentRecord) -> None:
        """Insert or update a document manifest record."""
        def _sync():
            conn = self._conn()
            try:
                normalized_meta = normalize_document_metadata(
                    record.meta,
                    doc_id=record.doc_id,
                    owner_type=record.owner_type,
                    owner_id=record.owner_id,
                    ingested_at=record.ingested_at,
                    source_path=record.source_path,
                    source_hash=record.source_hash,
                )
                payload = {
                    "doc_id": record.doc_id,
                    "source_path": record.source_path,
                    "source_hash": record.source_hash,
                    "ingested_at": record.ingested_at.isoformat(),
                    "tenant_id": getattr(record, "tenant_id", None) or DEFAULT_TENANT_ID,
                    "owner_type": record.owner_type,
                    "owner_id": record.owner_id,
                    "workspace_id": getattr(record, "workspace_id", None),
                    "origin_agent_id": getattr(record, "origin_agent_id", None),
                    "origin_user_id": getattr(record, "origin_user_id", None),
                    "origin_session_id": getattr(record, "origin_session_id", None),
                    "scope_model_version": getattr(record, "scope_model_version", None) or SCOPE_MODEL_VERSION,
                    "meta": json.dumps(normalized_meta),
                }
                self._execute(
                    conn,
                    """
                    INSERT INTO documents (
                        doc_id, source_path, source_hash, ingested_at, tenant_id,
                        owner_type, owner_id, workspace_id, origin_agent_id,
                        origin_user_id, origin_session_id, scope_model_version, meta
                    ) VALUES (
                        :doc_id, :source_path, :source_hash, :ingested_at, :tenant_id,
                        :owner_type, :owner_id, :workspace_id, :origin_agent_id,
                        :origin_user_id, :origin_session_id, :scope_model_version, :meta
                    )
                    ON CONFLICT(doc_id) DO UPDATE SET
                        source_path=excluded.source_path,
                        source_hash=excluded.source_hash,
                        ingested_at=excluded.ingested_at,
                        tenant_id=excluded.tenant_id,
                        owner_type=excluded.owner_type,
                        owner_id=excluded.owner_id,
                        workspace_id=excluded.workspace_id,
                        origin_agent_id=excluded.origin_agent_id,
                        origin_user_id=excluded.origin_user_id,
                        origin_session_id=excluded.origin_session_id,
                        scope_model_version=excluded.scope_model_version,
                        meta=excluded.meta
                    """,
                    params=payload,
                    log_context="document_upsert",
                )
                conn.commit()
            except Exception:
                self._safe_rollback(conn, "document_upsert")
                logger.exception("DocumentSQLStore.upsert_document failed for doc_id=%s", record.doc_id)
                raise
            finally:
                conn.close()

        return await self._run_sync(_sync)
