"""
SQLiteAdapter — SQLite database adapter for UMA.

This adapter implements the DBAdapter interface defined in
`uma.adapters.db.base` and provides a clean, production-grade
connection layer for SQLite-based UMA stores.

Design goals
------------
- Keep SQLite usage isolated from store logic.
- Provide dict-like row access via sqlite3.Row.
- Ensure that each call to `get_connection()` returns a fresh connection.
- Support UMA’s store patterns (explicit commit/close in try/finally).

Coding agent instructions
-------------------------
- Do not add business logic here.
- Stores should call `adapter.get_connection()` each time they need to
  access the DB.
- Future DB backends (PostgresAdapter, DuckDBAdapter) must follow the
  same DBAdapter interface.

Error handling
--------------
- Any failure to create a SQLite connection should log the error and
  propagate the exception upward.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from .base import DBAdapter, DBConnection

logger = logging.getLogger(__name__)


class SQLiteAdapter(DBAdapter):
    """
    SQLite DBAdapter implementation for UMA.

    Parameters
    ----------
    db_path : str
        Filesystem path to the SQLite database file. UMA stores use
        this adapter to create and manage DB connections for all DB
        operations.
    pragmas : dict | None
        Optional PRAGMA overrides applied per connection.

    Notes
    -----
    - Each call to `get_connection()` returns a **new** sqlite3 connection.
    - Connections set `row_factory = sqlite3.Row` to provide dict-like
      access for store query results.
    - This adapter applies conservative PRAGMA defaults by default
      (WAL, foreign_keys, synchronous). You may override or disable
      these via the `pragmas` parameter for portability/performance.
    """

    def __init__(self, db_path: str, pragmas: dict | None = None) -> None:
      # Basic validation
      if not isinstance(db_path, str):
        raise TypeError(f"db_path must be a string, got {type(db_path)}")
      if not db_path:
        raise ValueError("db_path cannot be empty for SQLiteAdapter")

      # Connection tuning defaults
      self.db_path = db_path
      self.timeout = 5.0  # seconds
      self.detect_types = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
      # Use URI mode when db_path begins with 'file:' allowing query params
      self.uri = db_path.startswith("file:")
      # Pragmas configurable per-connection (can be adjusted if needed)
      self._pragmas = pragmas if pragmas is not None else {
        "journal_mode": "WAL",
        "foreign_keys": 1,
        "synchronous": "NORMAL",
      }

      # Ensure parent directory exists for file-backed DBs
      parent = None
      if not self.uri:
        import os

        parent = os.path.dirname(self.db_path)
        if parent and not os.path.exists(parent):
          try:
            os.makedirs(parent, exist_ok=True)
            logger.debug("SQLiteAdapter: created parent directory %s", parent)
          except Exception as exc:
              logger.exception(
                "SQLiteAdapter: failed to create parent directory %s", parent
              )
              raise RuntimeError(
                f"SQLiteAdapter failed to create parent directory: {parent}. "
                "Ensure the path is writable and the parent exists."
              ) from exc

      logger.debug("SQLiteAdapter initialized with db_path=%s uri=%s", db_path, self.uri)

    def get_connection(self) -> DBConnection:
        """
        Return a new SQLite connection with sqlite3.Row as row_factory.

        Returns
        -------
        DBConnection
            A DB-API compatible connection object.

        Raises
        ------
        Exception
            If the connection cannot be established.
        """
        try:
          conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            detect_types=self.detect_types,
            uri=self.uri,
          )

          # Provide dict-like row access to match other adapters' semantics
          conn.row_factory = sqlite3.Row

          # Apply pragmatic connection tuning for durability/performance
          try:
            cur = conn.cursor()
            # Enable foreign key constraints when requested
            cur.execute("PRAGMA foreign_keys = %d" % int(self._pragmas.get("foreign_keys", 1)))
            # Set journal mode (WAL is typically better for concurrency)
            jm = self._pragmas.get("journal_mode")
            if jm:
              cur.execute("PRAGMA journal_mode = %s" % (jm,))
            # Set synchronous mode
            sync = self._pragmas.get("synchronous")
            if sync:
              cur.execute("PRAGMA synchronous = %s" % (sync,))
            cur.close()
          except Exception:
            # If pragmas fail, log but continue — pragmas are optimizations
            logger.exception("SQLiteAdapter: failed to apply PRAGMA settings")

          if logger.isEnabledFor(logging.DEBUG):
            def _trace(stmt: str) -> None:
              logger.debug("SQLiteAdapter: sql db=%s stmt=%s", self.db_path, stmt)

            conn.set_trace_callback(_trace)

          return conn  # type: ignore[return-value]
        except Exception:
          logger.exception(
            "SQLiteAdapter: failed to open database at %s", self.db_path
          )
          raise
