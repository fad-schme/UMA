from __future__ import annotations

from datetime import datetime

import pytest

from uma.core.semantic.core import SemanticCore
from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.stores.document_sql import DocumentRecord, DocumentSQLStore
from uma.types import Fact


@pytest.mark.asyncio
async def test_memory_sanity_doc_manifest_idempotency_and_versions():
    # Use a single in-memory SQLite connection across store calls.
    # DocumentSQLStore opens/closes connections per operation; ":memory:" would create a fresh DB each time.
    db = SQLiteAdapter("/tmp/uma_memory_sanity_documents.sqlite")
    docs = DocumentSQLStore(db_adapter=db)

    now = datetime.utcnow()
    r1 = DocumentRecord(
        doc_id="doc1",
        owner_type="user",
        owner_id="user:u1",
        source_path="p",
        source_hash="h1",
        ingested_at=now,
        meta={"ingest_signature": {"chunk_size_tokens": 128}},
    )
    await docs.upsert_document(r1)

    found = await docs.get_by_owner_and_hash(owner_type="user", owner_id="user:u1", source_hash="h1")
    assert found is not None
    assert found.meta and isinstance(found.meta, dict)
    assert "ingest_signature" in found.meta


@pytest.mark.asyncio
async def test_memory_sanity_fetch_more_facts_offset_is_deterministic():
    now = datetime.utcnow()
    facts = [
        Fact(id="1", subject="user:u1", predicate="P", object="a", created_at=now, updated_at=now, owner_type="user", owner_id="user:u1"),
        Fact(id="2", subject="user:u1", predicate="P", object="b", created_at=now, updated_at=now, owner_type="user", owner_id="user:u1"),
        Fact(id="3", subject="user:u1", predicate="Q", object="c", created_at=now, updated_at=now, owner_type="user", owner_id="user:u1"),
        Fact(id="4", subject="user:u1", predicate="P", object="d", created_at=now, updated_at=now, owner_type="user", owner_id="user:u1"),
    ]

    class Store:
        async def list_facts_for_owner(self, *, owner_type: str, owner_id: str, limit=None):
            return facts

    core = SemanticCore(llm=None, embedder=None, semantic_store=Store())
    core.store = Store()

    page1 = await core.fetch_more_facts("P", owner_type="user", owner_id="user:u1", k=2, offset=0)
    page2 = await core.fetch_more_facts("P", owner_type="user", owner_id="user:u1", k=2, offset=2)
    assert [f.id for f in page1] == ["1", "2"]
    assert [f.id for f in page2] == ["4"]
