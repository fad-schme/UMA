from uma.stores.base_sql_store import BaseSQLStore


class _DummyAdapter:
    def get_connection(self):
        raise RuntimeError("not used")


class _FailingConn:
    def rollback(self):
        raise RuntimeError("rollback failed")


class _OkConn:
    def __init__(self):
        self.called = False

    def rollback(self):
        self.called = True


def test_safe_rollback_swallows_errors():
    store = BaseSQLStore(_DummyAdapter())
    store._safe_rollback(_FailingConn(), "test")


def test_safe_rollback_calls_connection():
    store = BaseSQLStore(_DummyAdapter())
    conn = _OkConn()
    store._safe_rollback(conn, "test")
    assert conn.called is True
