from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.common.types import Fact


@pytest.mark.asyncio
async def test_semantic_search_subject_optional(uma_memory):
    """
    Semantic retrieval is ownership-only; subject is not a gating filter.
    """
    memory = uma_memory
    owner_type = "agent"
    owner_id = memory.agent_id or "agent-default"

    now = datetime.now(timezone.utc)
    emb = (await memory.embedder.embed(["shared"]))[0]

    facts = [
        Fact(
            id="fact_zt",
            subject="entity:zero_trust",
            predicate="PRINCIPLE",
            object="least privilege",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            salience=0.9,
            owner_type=owner_type,
            owner_id=owner_id,
            meta={},
        ),
        Fact(
            id="fact_cloud",
            subject="entity:cloud_security",
            predicate="PRINCIPLE",
            object="segmentation",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            salience=0.9,
            owner_type=owner_type,
            owner_id=owner_id,
            meta={},
        ),
        Fact(
            id="fact_userish",
            subject="user:local",
            predicate="REMEMBERED",
            object="note",
            created_at=now,
            updated_at=now,
            source_ids=[],
            confidence=0.9,
            salience=0.9,
            owner_type=owner_type,
            owner_id=owner_id,
            meta={},
        ),
    ]

    for f in facts:
        await memory.semantic_core.upsert_fact(f, emb)

    all_facts = await memory.semantic_core.search(
        query_embedding=emb,
        owner_type=owner_type,
        owner_id=owner_id,
        k=10,
        filters=None,
        query_text=None,
    )
    assert len(all_facts) == 3

    # Subject filters are ignored (ownership-only retrieval).
    filtered = await memory.semantic_core.search(
        query_embedding=emb,
        owner_type=owner_type,
        owner_id=owner_id,
        k=10,
        filters={"subject": "user:local"},
        query_text=None,
    )
    assert len(filtered) == 3

