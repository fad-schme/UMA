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
        logger.info("DocumentSQLStore initialized.")

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
