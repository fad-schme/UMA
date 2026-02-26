from datetime import datetime, timezone

import pytest

from uma.stores.document_sql import DocumentRecord


@pytest.mark.asyncio
async def test_document_manifest_persistence(uma_memory):
    mem = uma_memory
    record = DocumentRecord(
        doc_id="doc_test",
        source_path="/tmp/doc.txt",
        source_hash="hash123",
        ingested_at=datetime.now(timezone.utc),
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )

    await mem.document_store.upsert_document(record)

    # Verify record exists by querying directly
    conn = mem.document_store._conn()
    try:
        rows = mem.document_store._query_all(
            conn,
            "SELECT * FROM documents WHERE doc_id=?",
            params=["doc_test"],
            log_context="test_document_manifest",
        )
        assert rows and rows[0]["source_hash"] == "hash123"
    finally:
        conn.close()
