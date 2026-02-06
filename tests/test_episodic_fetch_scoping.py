from __future__ import annotations

from datetime import datetime

import pytest

from uma.adapters.db.sqlite_adapter import SQLiteAdapter
from uma.adapters.vector.faiss_adapter import FaissIndex
from uma.stores.episodic_sql import EpisodicSQLStore
from uma.types_episode import Episode


@pytest.mark.asyncio
async def test_episodic_fetch_summaries_owner_scoped():
    db = SQLiteAdapter("/tmp/uma_test_episodic_scoping.sqlite")
    vec = FaissIndex(dim=3)
    store = EpisodicSQLStore(db_adapter=db, vector_index=vec)

    now = datetime.utcnow()
    e1 = Episode(
        id="e1",
        user_id="user:u1",
        timestamp=now,
        summary="s1",
        raw="r1",
        meta={},
        owner_type="user",
        owner_id="user:u1",
    )
    # Same id format, different owner.
    e2 = Episode(
        id="e2",
        user_id="user:u2",
        timestamp=now,
        summary="s2",
        raw="r2",
        meta={},
        owner_type="user",
        owner_id="user:u2",
    )

    await store.add_episode(e1, embedding=[0.0, 0.0, 0.0])
    await store.add_episode(e2, embedding=[0.0, 0.0, 0.0])

    rows = await store.fetch_summaries(["e1", "e2"], owner_type="user", owner_id="user:u1")
    assert [r["id"] for r in rows] == ["e1"]

    rows = await store.fetch_transcripts(["e1", "e2"], owner_type="user", owner_id="user:u2")
    assert [r["id"] for r in rows] == ["e2"]
