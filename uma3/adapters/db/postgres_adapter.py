"""
PostgresAdapter — PostgreSQL database adapter for UMA-3.

This adapter implements the DBAdapter interface defined in
`uma3.adapters.db.base` and provides a robust, production-oriented
connection layer for PostgreSQL-backed UMA-3 stores.

Features
--------
- Optional connection pooling via `psycopg2.pool.SimpleConnectionPool`.
- Returns DB-API compatible connections where `cursor()` defaults to
  `RealDictCursor` (dict-like rows) so stores can treat results like
  dictionaries similar to `sqlite3.Row`.
- Safe wrapper around pooled connections that returns connections to
  the pool when `close()` is called (stores may call `close()` as usual).

Coding agent instructions
-------------------------
- Do not embed business logic in this adapter.
- Keep pooling optional (controlled by constructor flags).
- If `psycopg2` is not installed, raise a clear ImportError with an
  actionable message.
- Configure default cursor factory to `RealDictCursor` so stores get
  dict-like rows consistently across backends.

Error handling
--------------
- Any failure to create or obtain a connection should log the error and
  propagate the exception so UMA-3 initialization fails visibly.

Security / Notes
----------------
- DSN values may contain secrets (passwords). Callers should ensure
  configuration is stored securely (environment variables, secret
  manager) and not checked into source control.

"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..db.base import DBAdapter, DBConnection

logger = logging.getLogger(__name__)


class _PooledConnWrapper:
    """Wrap a pooled psycopg2 connection so `close()` returns it to pool.

    The wrapper delegates `cursor()`, `commit()`, `rollback()`, and
    other common methods to the underlying connection object. Calling
    `close()` will return the connection to the pool via
    `pool.putconn(conn)` instead of closing the physical connection.
    """

    def __init__(self, conn: Any, pool: Any, real_dict_cursor: Any) -> None:
        self._conn = conn
        self._pool = pool
        self._real_dict_cursor = real_dict_cursor

    def cursor(self, *args, **kwargs):
        # Default to RealDictCursor for dict-like rows unless caller
        # explicitly provided a different cursor_factory.
        if "cursor_factory" not in kwargs and self._real_dict_cursor is not None:
            kwargs["cursor_factory"] = self._real_dict_cursor
        return self._conn.cursor(*args, **kwargs)

    def commit(self) -> None:
        return self._conn.commit()

    def rollback(self) -> None:
        return self._conn.rollback()

    def close(self) -> None:
        try:
            # Return connection to the pool for reuse
            self._pool.putconn(self._conn)
        except Exception:
            # Fallback: try to close the underlying connection
            try:
                self._conn.close()
            except Exception:
                logger.exception("_PooledConnWrapper: failed to return/close pooled connection")

    # Provide attribute access to underlying connection for uncommon APIs
    def __getattr__(self, name: str):
        return getattr(self._conn, name)


class _DirectConnWrapper:
    """Wrap a direct psycopg2 connection and ensure cursor() uses RealDictCursor.

    This wrapper delegates all common DB-API calls to the underlying
    connection but ensures that `cursor()` uses `RealDictCursor` by
    default so stores get dict-like rows consistently.
    """

    def __init__(self, conn: Any, real_dict_cursor: Any) -> None:
        self._conn = conn
        self._real_dict_cursor = real_dict_cursor

    def cursor(self, *args, **kwargs):
        if "cursor_factory" not in kwargs and self._real_dict_cursor is not None:
            kwargs["cursor_factory"] = self._real_dict_cursor
        return self._conn.cursor(*args, **kwargs)

    def commit(self) -> None:
        return self._conn.commit()

    def rollback(self) -> None:
        return self._conn.rollback()

    def close(self) -> None:
        return self._conn.close()

    def __getattr__(self, name: str):
        return getattr(self._conn, name)


class PostgresAdapter(DBAdapter):
    """PostgreSQL DBAdapter implementation for UMA-3.

    Parameters
    ----------
    dsn: str
        A libpq-style DSN string or keyword arguments accepted by
        `psycopg2.connect`. Example: "postgresql://user:pass@host:5432/db".

    use_pool: bool
        If True, create a `SimpleConnectionPool` and reuse connections.
        Pooling is recommended in production but optional for tests.

    minconn, maxconn: int
        Pool size bounds used when `use_pool=True`.

    connect_timeout: Optional[int]
        Optional connect timeout (seconds) passed to the driver.

    Notes
    -----
    - This adapter lazily imports `psycopg2` to avoid hard dependency
      at module import time for environments that do not require Postgres.
    - If `psycopg2` is not installed, a helpful ImportError is raised.
    """

    def __init__(
        self,
        dsn: str,
        use_pool: bool = False,
        minconn: int = 1,
        maxconn: int = 5,
        connect_timeout: Optional[int] = None,
    ) -> None:
        if not isinstance(dsn, str) or not dsn:
            raise TypeError("PostgresAdapter requires a non-empty DSN string")

        self.dsn = dsn
        self.use_pool = bool(use_pool)
        self.minconn = int(minconn)
        self.maxconn = int(maxconn)
        self.connect_timeout = connect_timeout

        # Lazy imports - import driver only when adapter is constructed so
        # environments that don't use Postgres are unaffected.
        try:
            import psycopg2  # type: ignore
            import psycopg2.extras as _extras  # type: ignore
            from psycopg2 import pool as _pool  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency error path
            logger.exception("PostgresAdapter: missing psycopg2 dependency")
            raise ImportError(
                "psycopg2 is required for PostgresAdapter. Install with: pip install psycopg2-binary"
            ) from exc

        self._psycopg2 = psycopg2
        self._extras = _extras
        self._pool_module = _pool

        self._pool = None

        if self.use_pool:
            try:
                # Create a simple-threaded connection pool
                self._pool = self._pool_module.SimpleConnectionPool(
                    self.minconn,
                    self.maxconn,
                    dsn=self.dsn,
                    connect_timeout=self.connect_timeout,
                )
                logger.info(
                    "PostgresAdapter: initialized connection pool (min=%d max=%d)",
                    self.minconn,
                    self.maxconn,
                )
            except Exception:
                logger.exception("PostgresAdapter: failed to create connection pool")
                raise
        else:
            logger.info("PostgresAdapter initialized (pooling disabled)")

    def get_connection(self) -> DBConnection:
        """Return a DB-API compatible connection.

        If pooling is enabled, obtain a connection from the pool and wrap it
        so that calling `close()` returns the connection to the pool instead
        of closing the physical connection.
        """
        try:
            if self._pool is not None:
                raw_conn = self._pool.getconn()
                # Wrap pooled connection so `close()` returns it to pool.
                return _PooledConnWrapper(raw_conn, self._pool, self._extras.RealDictCursor)

            # Direct connection path
            conn = self._psycopg2.connect(dsn=self.dsn, connect_timeout=self.connect_timeout)
            return _DirectConnWrapper(conn, self._extras.RealDictCursor)
        except Exception:
            logger.exception("PostgresAdapter: failed to obtain a database connection")
            raise

    def close_pool(self) -> None:
        """Close and dispose the connection pool if one was created.

        Call this during application shutdown to release pooled resources.
        If pooling was not enabled this is a no-op.
        """
        if self._pool is None:
            return
        try:
            # Close all pooled connections
            self._pool.closeall()
            logger.info("PostgresAdapter: connection pool closed")
        except Exception:
            logger.exception("PostgresAdapter: failed to close connection pool")

