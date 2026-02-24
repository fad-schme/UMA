from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.adapters.vector.inmemory import InMemoryVectorIndex
from uma.core.chunk.core import ChunkCore
from uma.stores.chunk_sql import ChunkSQLStore
from uma.types import Chunk


def test_vector_index_query_returns_id_and_score() -> None:
    idx = InMemoryVectorIndex(dim=3)
    idx.upsert(
        ids=["a", "b"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadata=[{"owner_type": "user"}, {"owner_type": "user"}],
    )
    res = idx.query(vector=[1.0, 0.0, 0.0], k=2, filters={"owner_type": "user"})
    assert res, "expected non-empty vector results"
    assert all(isinstance(t, tuple) and len(t) == 2 for t in res)
    assert all(isinstance(t[0], str) and isinstance(t[1], float) for t in res)


@pytest.mark.asyncio
async def test_chunk_core_preserves_vector_score(tmp_path) -> None:
    db = SQLiteAdapter(str(tmp_path / "uma_test_vector_scores.sqlite"))
    vec = InMemoryVectorIndex(dim=3)
    store = ChunkSQLStore(db_adapter=db, vector_index=vec)
    core = ChunkCore(store)

    now = datetime.now(timezone.utc)
    c1 = Chunk(
        id="chunk_1",
        doc_id="doc1",
        text="hello.",
        page_range=(1, 1),
        position=1,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )
    c2 = Chunk(
        id="chunk_2",
        doc_id="doc1",
        text="world.",
        page_range=(1, 1),
        position=2,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )

    await store.upsert_chunk(c1, embedding=[1.0, 0.0, 0.0])
    await store.upsert_chunk(c2, embedding=[0.0, 1.0, 0.0])

    out = await core.search_chunks(query_embedding=[1.0, 0.0, 0.0], owner_type="user", owner_id="user:u1", k=2)
    assert [c.id for c in out][:2] == ["chunk_1", "chunk_2"]
    assert "vector_score" in (out[0].meta or {})
    assert float(out[0].meta["vector_score"]) >= float(out[1].meta.get("vector_score", -1.0))

