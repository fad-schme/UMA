from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.adapters.vector.base import VectorIndex
from uma.stores.chunk_sql import ChunkSQLStore
from uma.common.types import Chunk


class _SpyVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self.last_ids = None
        self.last_vectors = None
        self.last_tenant_ids = None
        self.last_owner_types = None
        self.last_owner_ids = None
        self.last_extra_metadata = None

    def upsert(self, ids, vectors, *, tenant_ids, owner_types, owner_ids, extra_metadata=None) -> None:
        self.last_ids = list(ids or [])
        self.last_vectors = list(vectors or [])
        self.last_tenant_ids = list(tenant_ids or [])
        self.last_owner_types = list(owner_types or [])
        self.last_owner_ids = list(owner_ids or [])
        self.last_extra_metadata = list(extra_metadata or [])

    def query(self, vector, *, tenant_id, owner_type, owner_id, k=10, extra_filters=None):
        return []

    def delete(self, ids) -> None:
        return None


@pytest.mark.asyncio
async def test_chunk_vector_payload_is_minimal_and_excludes_text(tmp_path) -> None:
    db = SQLiteAdapter(str(tmp_path / "uma_test_payload.sqlite"))
    spy = _SpyVectorIndex()
    store = ChunkSQLStore(db_adapter=db, vector_index=spy)

    now = datetime.now(timezone.utc)
    chunk = Chunk(
        id="chunk_1",
        doc_id="doc_1",
        text="This is the canonical chunk text stored in SQL.",
        page_range=(3, 4),
        position=7,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )

    await store.upsert_chunk(chunk, embedding=[0.0, 0.0, 0.0])

    assert spy.last_ids == ["chunk_1"]
    assert spy.last_tenant_ids == ["default"]
    assert spy.last_owner_types == ["user"]
    assert spy.last_owner_ids == ["user:u1"]
    assert spy.last_extra_metadata and isinstance(spy.last_extra_metadata[0], dict)
    meta = spy.last_extra_metadata[0]

    # Minimal, filterable fields only.
    assert meta.get("doc_id") == "doc_1"
    assert meta.get("kb_lane") == "raw"
    assert meta.get("position") == 7
    assert meta.get("page_start") == 3
    assert meta.get("page_end") == 4

    # Never duplicate full chunk text in vector payload.
    assert "text" not in meta
    assert "__text" not in meta
