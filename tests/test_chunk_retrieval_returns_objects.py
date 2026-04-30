from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.retrieve.rlm.snippet_refiner import SnippetRefiner
from uma.common.types import Chunk, Fact


@pytest.mark.asyncio
async def test_chunk_retrieval_returns_chunk_objects(uma_memory, tmp_path) -> None:
    memory = uma_memory
    owner_type = "agent"
    owner_id = memory.agent_id or "agent-default"

    doc = tmp_path / "doc.txt"
    doc.write_text(
        "This document contains hello world and enough text to be chunked and retrieved.\n"
        "Second sentence for stability.\n",
        encoding="utf-8",
    )
    await memory.ingest_document(str(doc), owner_type=owner_type, owner_id=owner_id)

    q = "hello world"
    query_embedding = (await memory.embedder.embed([q]))[0]
    res = await memory.chunk_core.search_chunks(
        query_embedding=query_embedding,
        owner_type=owner_type,
        owner_id=owner_id,
        k=5,
        query_text=q,
        filter_terms=False,
    )
    assert res
    assert all(isinstance(c, Chunk) for c in res)


@pytest.mark.asyncio
async def test_snippet_refiner_accepts_object_facts_and_chunks(uma_memory) -> None:
    memory = uma_memory

    class _Cfg:
        snippet_refiner_top_k = 3
        max_chunks = 2
        snippet_max_chars = 120

    now = datetime.now(timezone.utc)
    fact = Fact(
        id="fact_1",
        subject="user",
        predicate="STATES",
        object="Something happened.",
        created_at=now,
        updated_at=now,
        source_ids=[],
        confidence=0.9,
        salience=0.5,
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )
    chunks = [
        Chunk(
            id="chunk_1",
            doc_id="doc_1",
            text="Something happened. More context here.",
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
    ]

    refiner = SnippetRefiner(llm=memory.llm, cfg=_Cfg())
    out = await refiner.refine(query_text="something", facts=[fact], chunks=chunks)
    assert isinstance(out, list)
    assert out and isinstance(out[0], dict)

