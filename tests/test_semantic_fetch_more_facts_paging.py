from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from uma.types import Fact


@pytest.mark.asyncio
async def test_fetch_more_facts_pages_deterministically_by_offset(uma_memory):
    memory = uma_memory

    owner_type = "user"
    owner_id = "user:u1"

    now = datetime.now(timezone.utc)
    emb = (await memory.embedder.embed(["shared"]))[0]

    # Ensure deterministic ordering: SemanticSQLStore orders by updated_at DESC, then id ASC.
    facts = [
        Fact(
            id="fact_1",
            subject=owner_id,
            predicate="P",
            object="a",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            meta={},
            salience=0.1,
            owner_type=owner_type,
            owner_id=owner_id,
        ),
        Fact(
            id="fact_2",
            subject=owner_id,
            predicate="P",
            object="b",
            created_at=now,
            updated_at=now - timedelta(seconds=1),
            source_ids=[],
            confidence=0.9,
            meta={},
            salience=0.1,
            owner_type=owner_type,
            owner_id=owner_id,
        ),
        Fact(
            id="fact_3",
            subject=owner_id,
            predicate="Q",
            object="c",
            created_at=now,
            updated_at=now - timedelta(seconds=2),
            source_ids=[],
            confidence=0.9,
            meta={},
            salience=0.1,
            owner_type=owner_type,
            owner_id=owner_id,
        ),
        Fact(
            id="fact_4",
            subject=owner_id,
            predicate="P",
            object="d",
            created_at=now,
            updated_at=now - timedelta(seconds=3),
            source_ids=[],
            confidence=0.9,
            meta={},
            salience=0.1,
            owner_type=owner_type,
            owner_id=owner_id,
        ),
    ]

    for f in facts:
        await memory.semantic_core.upsert_fact(f, emb)

    page1 = await memory.semantic_core.fetch_more_facts("P", owner_type=owner_type, owner_id=owner_id, k=2, offset=0)
    page2 = await memory.semantic_core.fetch_more_facts("P", owner_type=owner_type, owner_id=owner_id, k=2, offset=2)

    assert [f.id for f in page1] == ["fact_1", "fact_2"]
    assert [f.id for f in page2] == ["fact_4"]
