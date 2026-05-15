from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.common.types import OwnershipRef, Skill
from uma.memory.chunk.core import ChunkSearchOptions


@pytest.mark.asyncio
async def test_chunk_search_does_not_require_subject(uma_memory, tmp_path):
    memory = uma_memory
    owner_type = "agent"
    owner_id = memory.agent_id or "agent-default"

    doc = tmp_path / "doc.txt"
    doc.write_text(
        "This is a test document used for UMA retrieval. "
        "It contains the phrase hello world in a longer passage so lexical search can match it reliably. "
        "The rest of this sentence is padding to ensure the stored chunk is long enough for LIKE-based lexical search.\n"
    )
    await memory.ingest_document(str(doc), owner_type=owner_type, owner_id=owner_id)

    q = "hello world"
    query_embedding = (await memory.embedder.embed([q]))[0]

    res = await memory.chunk_core.search_chunks(
        query_embedding=query_embedding,
        owner_type=owner_type,
        owner_id=owner_id,
        k=5,
    )
    assert res, "Expected dense chunk retrieval to return at least one result"

    res2 = await memory.chunk_core.search_chunks(
        query_embedding=query_embedding,
        owner_type=owner_type,
        owner_id=owner_id,
        k=5,
        options=ChunkSearchOptions(query_text=q, filter_terms=False),
    )
    assert res2, "Expected hybrid chunk retrieval to return at least one result"

    if hasattr(memory.chunk_core.store, "lexical_search"):
        assert any(
            (getattr(ch, "meta", None) or {}).get("retrieval_method") == "lexical"
            for ch in res2
        ), "Expected lexical capability to tag at least one chunk as lexical"
    else:
        assert all(
            (getattr(ch, "meta", None) or {}).get("retrieval_method") == "vector"
            for ch in res2
        ), "Expected vector-only path when lexical capability is absent"


@pytest.mark.asyncio
async def test_procedural_search_does_not_require_subject(uma_memory):
    memory = uma_memory
    owner_type = "agent"
    owner_id = memory.agent_id or "agent-default"

    skill = Skill(
        id="skill_s1",
        name="Test skill",
        description="How to do the hello world procedure safely and deterministically.",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        owner_type=owner_type,
        owner_id=owner_id,
    )
    emb = (await memory.embedder.embed([skill.description]))[0]
    persisted = await memory.procedural_core.add_skill(skill, emb)
    assert persisted is not None

    query_embedding = (await memory.embedder.embed(["hello world procedure"]))[0]
    res = await memory.procedural_core.search(
        query_embedding=query_embedding,
        owner=OwnershipRef(tenant_id="default", owner_type=owner_type, owner_id=owner_id),
        k=5,
    )
    assert res and res[0].id == "skill_s1"
