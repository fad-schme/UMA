"""
Base SQL Store for UMA.

This module defines a reusable base class for UMA stores that persist
data in a relational database via a DBAdapter abstraction.

It centralizes:
- Connection acquisition via DBAdapter.
- Safe execution of SQL statements with logging and error handling.
- Small convenience helpers for common query patterns.

Concrete stores such as EpisodicSQLStore, SemanticSQLStore, and
ProceduralSQLStore should inherit from this base class (or from a more
specialized base class like BaseVectorSQLStore) rather than duplicating
connection and error-handling logic.

Coding agent instructions
-------------------------
- Do NOT put domain-specific logic in this file.
- Use this base class for cross-cutting concerns:
  - getting connections
  - executing SQL with parameters
  - fetching rows
- Concrete stores are responsible for:
  - defining schemas (CREATE TABLE statements)
  - mapping rows to domain objects
  - implementing domain-specific methods (add_episode, upsert_fact, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Optional, Sequence, TypeVar

from ..adapters.db.base import DBAdapter, DBConnection

_T = TypeVar("_T")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL identifier whitelists — Pattern A fix (Bandit B608)
# ---------------------------------------------------------------------------
# SQL identifiers (table names, column names, column type definitions) cannot
# be bound as DB-API 2.0 parameters; the parser requires them at parse time.
# The defence is a whitelist: _ensure_column and _table_columns only accept
# identifiers from these frozen sets, so no external value can reach the DDL
# f-strings even via an errant future caller.
# ---------------------------------------------------------------------------
_ALLOWED_SCHEMA_TABLES: frozenset = frozenset({
    "facts", "episodes", "episode_clusters", "skills", "chunks",
    "documents", "uma_store_meta",
})
_ALLOWED_SCHEMA_COLUMNS: frozenset = frozenset({
    "id", "tenant_id", "workspace_id", "owner_type", "owner_id",
    "session_id", "origin_agent_id", "origin_user_id", "origin_session_id",
    "scope_model_version", "trust_score", "content_hash", "quarantined_at",
    "doc_id", "source_path", "source_hash", "ingested_at", "meta",
    "user_id", "agent_id", "request_id", "subject", "predicate", "object",
    "salience", "confidence", "source_ids", "source", "created_at",
    "updated_at", "summary", "raw", "tags", "embedding", "name",
    "description", "trigger_phrases", "trigger_patterns", "plan", "tools",
    "example", "text", "position", "page_start", "page_end",
    "supersedes", "superseded_by", "superseded_at",
    # Skill discriminator + agent-profile fields. `kind` distinguishes
    # ordinary procedural rows from `agent_profile` rows (one per agent,
    # consulted by PromotionPolicy.qualifies_for_agent_kb). `focus_areas`
    # is the JSON-encoded keyword list the qualifier's deterministic
    # branch matches against fact text. `profile_embedding` holds the
    # agent-profile embedding as a BLOB — agent_profile rows never enter
    # the vector index, so procedural search cannot leak them.
    "kind", "focus_areas", "profile_embedding",
})
# Column type definitions used in ALTER TABLE ADD COLUMN statements.
_ALLOWED_COLUMN_DEFS: frozenset = frozenset({
    "TEXT",
    "TEXT NOT NULL DEFAULT 'default'",
    "TEXT NOT NULL DEFAULT 'procedural'",
    "TEXT NOT NULL DEFAULT ''",
    "BLOB",
    "DATETIME",
    "REAL NOT NULL DEFAULT 0.5",
})


class BaseSQLStore:
    """
    Improved BaseSQLStore with safer DB patterns, context-managed cursor
    usage, and consistent commit behavior.

    Enhancements:
    - Use context managers for cursor operations.
    - Ensure commits only occur when explicitly requested by caller.
    - Better error messages with SQL + parameters.
    - Ensure cursor is always closed.

    This class handles:
    - Connection acquisition via DBAdapter.
    - Common execution helpers for SQL operations.
    - Structured logging of errors and SQL statements on failure.

    It does NOT:
    - Define any schema.
    - Encode knowledge of specific tables or domain models.
    """

    def __init__(self, db_adapter: DBAdapter) -> None:
        """
        Initialize the base SQL store.

        Parameters
        ----------
        db_adapter : DBAdapter
            Database adapter used to create DB connections.
        """
        self._db_adapter = db_adapter

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    def _conn(self) -> DBConnection:
        """
        Obtain a new database connection from the adapter.

        Returns
        -------
        DBConnection
            A DB-API compatible connection.

        Notes
        -----
        - Callers are responsible for closing the connection.
        - Use try/finally:

              conn = self._conn()
              try:
                  ...
                  conn.commit()
              finally:
                  conn.close()
        """
        return self._db_adapter.get_connection()

    def _adapt_sql(self, sql: str) -> str:
        """
        Convert SQL parameter placeholders to the adapter's paramstyle.

        Supported:
        - qmark: "?"
        - format: "%s"
        - pyformat: "%(name)s" for named params
        """
        style = getattr(self._db_adapter, "paramstyle", "qmark")
        if style == "qmark":
            return sql
        if style == "format":
            return sql.replace("?", "%s")
        if style == "pyformat":
            sql = re.sub(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)", r"%(\1)s", sql)
            return sql.replace("?", "%s")
        return sql

    def _safe_rollback(self, conn: DBConnection, log_context: str = "") -> None:
        """Attempt a rollback, logging any failures without raising."""
        ctx = f" [{log_context}]" if log_context else ""
        try:
            conn.rollback()
        except Exception:
            logger.exception("BaseSQLStore._safe_rollback%s: rollback failed.", ctx)

    # ------------------------------------------------------------------ #
    # Async offload for blocking SQL work
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _run_sync(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """Run a blocking DB callable on the default thread executor.

        Sqlite3 is a synchronous C API. Every store method that touches the
        database must go through this helper so the event loop is not
        blocked for the duration of the query. Concurrent async callers can
        then genuinely overlap: readers run in parallel under WAL, and
        writers serialize at the SQLite level rather than at the event
        loop.

        The whole per-method sync body is wrapped in one `to_thread` call so
        that the connection is created, used, committed, and closed on the
        same worker thread (sqlite3 default ``check_same_thread=True``).
        """
        return await asyncio.to_thread(fn, *args, **kwargs)

    # ------------------------------------------------------------------ #
    # Execution helpers
    # ------------------------------------------------------------------ #

    def _execute(
        self,
        conn: DBConnection,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        log_context: str = "",
    ) -> None:
        """
        Execute a single SQL statement (INSERT/UPDATE/DELETE).

        Parameters
        ----------
        conn : DBConnection
            Open DB connection.
        sql : str
            SQL statement to execute.
        params : Optional[Sequence[Any]]
            Positional parameters for the SQL statement.
        log_context : str
            Short label for log messages.

        Raises
        ------
        Exception
            Any DB-API exception raised by the driver will be logged and
            re-raised.
        """
        ctx = f" [{log_context}]" if log_context else ""
        try:
            # Many DB-API cursors do NOT support context manager,
            # so we fallback if needed.
            cursor = conn.cursor()
            try:
                adapted_sql = self._adapt_sql(sql)
                if params is None:
                    cursor.execute(adapted_sql)
                else:
                    cursor.execute(adapted_sql, params)
            finally:
                cursor.close()
        except Exception as e:
            logger.exception(
                "BaseSQLStore._execute%s: SQL failed: %s ; params=%s",
                ctx,
                sql,
                params,
            )
            # Re-raise so tests (and callers) can handle the failure
            raise e

    def _query_all(
        self,
        conn: DBConnection,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        log_context: str = "",
    ) -> list[Any]:
        """
        Execute a SELECT statement and return all rows.

        Parameters
        ----------
        conn : DBConnection
            Open DB connection.
        sql : str
            SELECT statement to execute.
        params : Optional[Sequence[Any]]
            Positional parameters for the SQL statement.
        log_context : str
            Short label for log messages.

        Returns
        -------
        List[Any]
            The list of rows returned by the driver (often sqlite3.Row).

        Raises
        ------
        Exception
            Any DB-API exception raised by the driver will be logged and
            re-raised.
        """
        ctx = f" [{log_context}]" if log_context else ""
        try:
            cursor = conn.cursor()
            try:
                adapted_sql = self._adapt_sql(sql)
                if params is None:
                    cursor.execute(adapted_sql)
                else:
                    cursor.execute(adapted_sql, params)
                return list(cursor.fetchall())
            finally:
                cursor.close()
        except Exception:
            logger.exception(
                "BaseSQLStore._query_all%s: SQL failed: %s ; params=%s",
                ctx,
                sql,
                params,
            )
            raise

    # ------------------------------------------------------------------ #
    # Lightweight schema helpers
    # ------------------------------------------------------------------ #

    def _table_columns(self, conn: DBConnection, table_name: str) -> set[str]:
        if table_name not in _ALLOWED_SCHEMA_TABLES:
            raise ValueError(
                f"_table_columns: table {table_name!r} is not in the allowed schema table set."
            )
        rows = self._query_all(
            conn,
            f"PRAGMA table_info({table_name})",  # nosec B608 — table_name validated against _ALLOWED_SCHEMA_TABLES
            log_context=f"schema_info_{table_name}",
        )
        columns: set[str] = set()
        for row in rows:
            if hasattr(row, "get"):
                name = row.get("name")
            else:
                name = row["name"] if "name" in row.keys() else None
            if isinstance(name, str) and name:
                columns.add(name)
        return columns

    def _ensure_column(
        self,
        conn: DBConnection,
        table_name: str,
        column_name: str,
        column_sql: str,
    ) -> None:
        if table_name not in _ALLOWED_SCHEMA_TABLES:
            raise ValueError(
                f"_ensure_column: table {table_name!r} is not in the allowed schema table set."
            )
        if column_name not in _ALLOWED_SCHEMA_COLUMNS:
            raise ValueError(
                f"_ensure_column: column {column_name!r} is not in the allowed schema column set."
            )
        if column_sql not in _ALLOWED_COLUMN_DEFS:
            raise ValueError(
                f"_ensure_column: column definition {column_sql!r} is not in the allowed column def set."
            )
        if column_name in self._table_columns(conn, table_name):
            return
        self._execute(
            conn,
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}",  # nosec B608 — all three identifiers validated against frozenset whitelists above
            log_context=f"schema_add_column_{table_name}_{column_name}",
        )

    def _query_one(
        self,
        conn: DBConnection,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        log_context: str = "",
    ) -> Optional[Any]:
        """
        Execute a SELECT statement and return a single row or None.

        Parameters
        ----------
        conn : DBConnection
            Open DB connection.
        sql : str
            SELECT statement to execute.
        params : Optional[Sequence[Any]]
            Positional parameters for the SQL statement.
        log_context : str
            Short label for log messages.

        Returns
        -------
        Optional[Any]
            The first row, or None if no rows are found.

        Raises
        ------
        Exception
            Any DB-API exception raised by the driver will be logged and
            re-raised.
        """
        ctx = f" [{log_context}]" if log_context else ""
        try:
            cursor = conn.cursor()
            try:
                adapted_sql = self._adapt_sql(sql)
                if params is None:
                    cursor.execute(adapted_sql)
                else:
                    cursor.execute(adapted_sql, params)
                return cursor.fetchone()
            finally:
                cursor.close()
        except Exception:
            logger.exception(
                "BaseSQLStore._query_one%s: SQL failed: %s ; params=%s",
                ctx,
                sql,
                params,
            )
            raise
