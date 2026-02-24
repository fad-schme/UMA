from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from uma.core.retrieval.rlm.snippet_refiner import SnippetRefiner
from uma.types import Chunk


def test_snippet_refiner_presentation_only_does_not_filter_by_relevance() -> None:
    class _Cfg:
        max_chunks = 3
        snippet_max_chars = 60

    now = datetime.now(timezone.utc)
    chunks = [
        Chunk(
            id="chunk_1",
            doc_id="doc1",
            text="This chunk is about apples. It should not be dropped.",
            page_range=(1, 1),
            position=1,
            source_path="/tmp/x",
            source_hash="h",
            created_at=now,
            updated_at=now,
            owner_type="user",
            owner_id="user:u1",
            meta={},
        ),
        Chunk(
            id="chunk_2",
            doc_id="doc2",
            text="This chunk is about bananas. It should not be dropped either.",
            page_range=(1, 1),
            position=1,
            source_path="/tmp/y",
            source_hash="h2",
            created_at=now,
            updated_at=now,
            owner_type="user",
            owner_id="user:u1",
            meta={},
        ),
    ]

    refiner = SnippetRefiner(llm=None, cfg=_Cfg())

    async def run():
        return await refiner.refine(query_text="zzz-not-present", facts=[], chunks=chunks)

    out = asyncio.run(run())
    assert len(out) == 2
    assert all(isinstance(s, dict) for s in out)
    assert all(s.get("text") for s in out)
    assert all(len(s["text"]) <= _Cfg.snippet_max_chars for s in out)
    assert out[0]["source"]["doc_id"] == "doc1"
    assert out[1]["source"]["doc_id"] == "doc2"

