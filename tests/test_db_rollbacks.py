from uma.stores.base_sql_store import BaseSQLStore
from uma.adapters.db.sqlite_adapter import SQLiteAdapter


class _FailingConn:
    def rollback(self):
        raise RuntimeError("rollback failed")


class _OkConn:
    def __init__(self):
        self.called = False

    def rollback(self):
        self.called = True


def test_safe_rollback_swallows_errors(tmp_path):
    store = BaseSQLStore(SQLiteAdapter(str(tmp_path / "t.db")))
    store._safe_rollback(_FailingConn(), "test")


def test_safe_rollback_calls_connection(tmp_path):
    store = BaseSQLStore(SQLiteAdapter(str(tmp_path / "t.db")))
    conn = _OkConn()
    store._safe_rollback(conn, "test")
    assert conn.called is True
