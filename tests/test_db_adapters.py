from uma.adapters.db.sqlite_adapter import SQLiteAdapter


def test_sqlite_adapter_row_access_and_rollback(tmp_path):
    db_path = tmp_path / "test.db"
    adapter = SQLiteAdapter(str(db_path))

    conn = adapter.get_connection()
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        cur.execute("INSERT INTO items (name) VALUES (?)", ("alpha",))
        conn.commit()

        cur.execute("SELECT id, name FROM items")
        row = cur.fetchone()
        assert row["name"] == "alpha"

        conn.rollback()
    finally:
        conn.close()
