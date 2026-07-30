"""
RetrievalAuditStore — structured append-only audit log for retrieval calls.

Purpose
-------
The retrieval audit log records one row per `retrieve_context` /
`retrieve_memory` call. It is intentionally separate from the per-record
`meta.security.audit_log` (which tracks quarantine actions on individual
records). The two systems answer different operator questions:

- record audit log → "who quarantined / reinstated this fact?"
- retrieval audit log → "who queried for what; what severity; what came back?"

Query text is NEVER stored in clear. A SHA-256 hash (first 16 hex chars)
plus a short preview (first 80 chars) gives operators enough information
to correlate logs and inspect suspicious patterns without persisting
arbitrary user payloads.

Schema
------
    request_id      TEXT PRIMARY KEY   -- runtime_context.request_id
    tenant_id       TEXT
    user_id         TEXT
    agent_id        TEXT
    query_hash      TEXT               -- sha256_hex(query)[:16]
    query_preview   TEXT               -- query[:80]
    scan_severity   TEXT               -- "none" / "low" / "medium" / "high"
    lanes           TEXT               -- JSON list of participating lanes
    result_count    INTEGER            -- count of chunks + facts returned
    refined_via_llm INTEGER            -- 1 if snippet refinement ran, else 0
    pruned_via_llm  INTEGER            -- 1 if fact pruning ran, else 0
    created_at      DATETIME           -- ISO-8601 UTC

The store is opt-out via config — enabled by default at the standard
embedded profile path. Disable by setting
`security.retrieval_audit_enabled: false` in your config YAML.

The store is also fail-soft. Any error during write is logged at WARNING
and swallowed; retrieval results are returned regardless. The audit
log is observability, not a correctness dependency.

Read API
--------
The management module exposes `list_retrieval_audit(memory, ...)` for
operators to query this log; see `uma.api.management`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS retrieval_audit (
    request_id      TEXT PRIMARY KEY,
    tenant_id       TEXT,
    user_id         TEXT,
    agent_id        TEXT,
    query_hash      TEXT,
    query_preview   TEXT,
    scan_severity   TEXT,
    lanes           TEXT,
    result_count    INTEGER,
    refined_via_llm INTEGER,
    pruned_via_llm  INTEGER,
    created_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_retrieval_audit_tenant_user_created
    ON retrieval_audit(tenant_id, user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_audit_severity_created
    ON retrieval_audit(scan_severity, created_at DESC);
"""


@dataclass(frozen=True)
class RetrievalAuditRow:
    """Single row of the retrieval audit log."""
    request_id: str
    tenant_id: str
    user_id: str
    agent_id: Optional[str]
    query_hash: str
    query_preview: str
    scan_severity: str
    lanes: list[str] = field(default_factory=list)
    result_count: int = 0
    refined_via_llm: bool = False
    pruned_via_llm: bool = False
    created_at: str = ""


class RetrievalAuditStore:
    """Append-only sqlite-backed retrieval audit log.

    Single-writer: append() opens a short-lived connection, writes one
    row, closes. No connection pooling, no caching — keeps the
    failure mode simple. Acceptable performance for the expected write
    volume (one row per retrieval call).
    """

    def __init__(self, db_path: str) -> None:
        if not isinstance(db_path, str) or not db_path.strip():
            raise ValueError("RetrievalAuditStore: db_path required")
        self._db_path = db_path.strip()
        # Ensure parent directory exists; the store may be the first thing
        # to write to .uma/db/ on a fresh install.
        try:
            parent = os.path.dirname(self._db_path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
        except OSError:
            logger.exception(
                "RetrievalAuditStore: failed to create parent dir for %s; subsequent writes may fail",
                self._db_path,
            )
        # Initialize schema on construction. Idempotent.
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.executescript(_SCHEMA_SQL)
        except sqlite3.Error:
            logger.exception(
                "RetrievalAuditStore: failed to initialize schema at %s",
                self._db_path,
            )

    def append(self, row: RetrievalAuditRow) -> bool:
        """Insert one audit row. Returns True on success, False on failure.

        Fail-soft: any database error is logged at WARNING and swallowed.
        Retrieval calls do not wait for or fail on audit-log writes.
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO retrieval_audit (
                        request_id, tenant_id, user_id, agent_id,
                        query_hash, query_preview, scan_severity,
                        lanes, result_count, refined_via_llm,
                        pruned_via_llm, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.request_id,
                        row.tenant_id,
                        row.user_id,
                        row.agent_id,
                        row.query_hash,
                        row.query_preview,
                        row.scan_severity,
                        json.dumps(list(row.lanes or [])),
                        int(row.result_count or 0),
                        1 if row.refined_via_llm else 0,
                        1 if row.pruned_via_llm else 0,
                        row.created_at or datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
            return True
        except sqlite3.Error:
            logger.warning(
                "RetrievalAuditStore.append failed for request_id=%s (continuing without audit row)",
                row.request_id,
                exc_info=True,
            )
            return False

    def list_rows(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        severity_min: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit rows. Used by `uma.api.management.list_retrieval_audit`.

        - tenant_id / user_id filter exact match if provided.
        - severity_min returns rows at or above the given severity tier
          (ordering: none < low < medium < high). None or "none" returns all.
        - limit is capped at 1000 to keep the response bounded.
        """
        limit = max(1, min(int(limit or 100), 1000))
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        severity_floor = severity_order.get((severity_min or "").lower(), 0)

        # B608: _AUDIT_FILTER_COLS maps filter names to their exact SQL fragment.
        # Only these two column names are ever appended to where_clause; both are
        # hardcoded string constants, not derived from request parameters.
        _AUDIT_FILTER_COLS = {
            "tenant_id": "tenant_id = ?",
            "user_id": "user_id = ?",
        }
        where: list[str] = []
        params: list[Any] = []
        if tenant_id:
            where.append(_AUDIT_FILTER_COLS["tenant_id"])
            params.append(tenant_id)
        if user_id:
            where.append(_AUDIT_FILTER_COLS["user_id"])
            params.append(user_id)
        where_clause = (" WHERE " + " AND ".join(where)) if where else ""
        # B608: the only variable part of the SELECT is where_clause, which
        # is built exclusively from the two whitelisted fragments above.
        sql = (
            "SELECT request_id, tenant_id, user_id, agent_id, query_hash, "
            "query_preview, scan_severity, lanes, result_count, "
            "refined_via_llm, pruned_via_llm, created_at "
            f"FROM retrieval_audit{where_clause} ORDER BY created_at DESC LIMIT ?"
        )
        params.append(limit)

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            logger.warning(
                "RetrievalAuditStore.list_rows failed; returning empty list",
                exc_info=True,
            )
            return []

        out: list[dict[str, Any]] = []
        for r in rows:
            sev = (r["scan_severity"] or "").lower()
            if severity_order.get(sev, 0) < severity_floor:
                continue
            try:
                lanes = json.loads(r["lanes"] or "[]")
            except Exception:
                lanes = []
            out.append({
                "request_id": r["request_id"],
                "tenant_id": r["tenant_id"],
                "user_id": r["user_id"],
                "agent_id": r["agent_id"],
                "query_hash": r["query_hash"],
                "query_preview": r["query_preview"],
                "scan_severity": r["scan_severity"],
                "lanes": lanes,
                "result_count": r["result_count"],
                "refined_via_llm": bool(r["refined_via_llm"]),
                "pruned_via_llm": bool(r["pruned_via_llm"]),
                "created_at": r["created_at"],
            })
        return out
