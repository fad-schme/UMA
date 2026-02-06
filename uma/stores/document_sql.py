"""
DocumentSQLStore — SQL store for document manifests.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from .base_sql_store import BaseSQLStore
from ..adapters.db.base import DBAdapter
from ..core.utils.store_metadata import ensure_store_metadata

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
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    meta TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_type, owner_id);
                CREATE INDEX IF NOT EXISTS idx_documents_owner_hash ON documents(owner_type, owner_id, source_hash);
                """
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
        owner_type: str,
        owner_id: str,
        source_hash: str,
    ) -> DocumentRecord | None:
        """
        Return the most recently ingested document record for an owner+hash.

        This is used for idempotent ingestion gates.
        """
        if not owner_type or not owner_id or not source_hash:
            return None

        conn = self._conn()
        try:
            row = self._query_one(
                conn,
                """
                SELECT doc_id, source_path, source_hash, ingested_at, owner_type, owner_id, meta
                FROM documents
                WHERE owner_type = ? AND owner_id = ? AND source_hash = ?
                ORDER BY ingested_at DESC
                LIMIT 1
                """,
                params=[owner_type, owner_id, source_hash],
                log_context="documents_get_by_owner_hash",
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
                owner_type=str((row.get("owner_type") if hasattr(row, "get") else row["owner_type"]) or ""),
                owner_id=str((row.get("owner_id") if hasattr(row, "get") else row["owner_id"]) or ""),
                meta=meta,
            )
        except Exception:
            logger.exception(
                "DocumentSQLStore.get_by_owner_and_hash failed owner=%s:%s",
                owner_type,
                owner_id,
            )
            return None
        finally:
            conn.close()

    async def upsert_document(self, record: DocumentRecord) -> None:
        conn = self._conn()
        try:
            payload = {
                "doc_id": record.doc_id,
                "source_path": record.source_path,
                "source_hash": record.source_hash,
                "ingested_at": record.ingested_at.isoformat(),
                "owner_type": record.owner_type,
                "owner_id": record.owner_id,
                "meta": json.dumps(record.meta or {}),
            }
            self._execute(
                conn,
                """
                INSERT INTO documents (
                    doc_id, source_path, source_hash, ingested_at, owner_type, owner_id, meta
                ) VALUES (
                    :doc_id, :source_path, :source_hash, :ingested_at, :owner_type, :owner_id, :meta
                )
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    source_hash=excluded.source_hash,
                    ingested_at=excluded.ingested_at,
                    owner_type=excluded.owner_type,
                    owner_id=excluded.owner_id,
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
