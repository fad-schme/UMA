"""
Database adapter abstractions for UMA-3.

This module defines a minimal but extensible database abstraction that
decouples UMA-3's stores from a specific DB backend such as SQLite or
PostgreSQL.

The key idea:
- Stores depend on DBAdapter, not on a concrete driver like sqlite3.
- DBAdapter returns a DBConnection, which is a thin protocol around a
  DB-API compatible connection.
- Concrete implementations (e.g. SQLiteAdapter, PostgresAdapter) live in
  `uma3.adapters.db.*` and implement DBAdapter.

Coding agent instructions
-------------------------
- Do NOT put application logic in this file.
- DBAdapter implementations MUST:
  - Return a valid DB-API connection object from `get_connection()`.
  - Ensure connections can be used in a `try/finally` block (i.e. support
    `.commit()` and `.close()` methods).
  - Configure row factories (e.g. sqlite3.Row) at the connection level
    when needed by stores.
- When adding new backends (PostgreSQL, DuckDB, etc.), create a new
  module under `uma3.adapters.db` that subclasses DBAdapter and fully
  documents any backend-specific behavior and configuration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DBConnection(Protocol):
    """
    Protocol representing a DB-API compatible connection.

    This protocol intentionally abstracts away the concrete driver
    (sqlite3, psycopg2, duckdb, etc.) and focuses on the minimum
    functionality required by UMA-3 stores.

    Expected capabilities
    ---------------------
    - `cursor() -> Any`: returns a DB-API compatible cursor.
    - `commit() -> None`: commits the current transaction.
    - `close() -> None`: closes the connection.

    Optional but common capabilities
    --------------------------------
    Many DB-API implementations also support:
    - Context manager protocol (`__enter__`, `__exit__`)
    - `rollback() -> None`

    UMA-3 stores explicitly manage commit/close in try/finally blocks
    and do not rely on context manager support.
    """

    def cursor(self) -> Any:
        """
        Return a DB-API compatible cursor.

        The exact type is driver-specific (sqlite3.Cursor, psycopg2 cursor,
        etc.), so this method returns `Any`. Stores typically call
        `cursor.execute(...)` and `cursor.fetchall()`.
        """
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def close(self) -> None:
        """Close the connection and free underlying resources."""
        ...


class DBAdapter(ABC):
    """
    Abstract base class for UMA-3 database adapters.

    A DBAdapter is responsible for:
    - Creating connections to a specific database backend.
    - Optionally configuring connection-level settings (e.g. row
      factories, PRAGMA settings for SQLite, connection options).

    Stores receive a DBAdapter instance in their constructor and call
    `get_connection()` whenever they need to perform DB operations.

    Example
    -------
    For SQLite:

        adapter = SQLiteAdapter(db_path=\"data/episodic.db\")
        store = EpisodicSQLStore(db_adapter=adapter, vector_index=index)

    For PostgreSQL (future):

        adapter = PostgresAdapter(dsn=os.environ[\"UMA_PG_DSN\"])
        store = SemanticSQLStore(db_adapter=adapter, vector_index=index, ...)

    Coding agent instructions
    -------------------------
    - When you implement a new DBAdapter, subclass this class and implement
      `get_connection()` to return a DBConnection.
    - Configure any driver-specific row factory (such as `sqlite3.Row`)
      inside `get_connection()` so stores can rely on dict-like row access.
    - Do not cache connections inside DBAdapter unless you also implement
      safe pooling behavior; UMA-3 stores assume each call to
      `get_connection()` returns a fresh, usable connection.
    """

    @abstractmethod
    def get_connection(self) -> DBConnection:
        """
        Return a new DBConnection instance.

        Implementations must:
        - Create a new physical or pooled connection.
        - Configure any required connection settings (row_factory, encoding).
        - Return an object that satisfies the DBConnection protocol.

        Error handling
        --------------
        - Implementations should raise a clear exception (e.g. a driver
          error) if a connection cannot be established. Callers are
          expected to catch, log, and handle these errors at a higher
          level (e.g. during UMA-3 initialization).
        """
        raise NotImplementedError("DBAdapter.get_connection() must be implemented")