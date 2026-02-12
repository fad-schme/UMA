from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from uma.core.chunk.core import ChunkCore
from uma.types import Chunk


class _DictChunkStore:
    async def search(self, **_kwargs):
        return [{"id": "chunk_1", "text": "bad", "meta": {}}]

    async def search_text(self, *_args, **_kwargs):
        return [{"id": "chunk_1", "text": "bad", "meta": {}}]


class _ObjChunkStore:
    async def search(self, **_kwargs):
        return [
            Chunk(
                id="chunk_1",
                doc_id="doc_1",
                text="hello.",
                page_range=(1, 1),
                position=1,
                source_path="/tmp/x",
                source_hash="h",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                owner_type="user",
                owner_id="user:u1",
                meta={},
            )
        ]

    async def search_text(self, *_args, **_kwargs):
        return await self.search()


def test_chunkcore_raises_on_dict_results() -> None:
    core = ChunkCore(_DictChunkStore())

    async def run():
        return await core.search_chunks(query_embedding=[0.1], owner_type="user", owner_id="user:u1", k=1)

    try:
        asyncio.run(run())
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_chunkcore_returns_chunk_objects() -> None:
    core = ChunkCore(_ObjChunkStore())

    async def run():
        return await core.search_chunks(query_embedding=[0.1], owner_type="user", owner_id="user:u1", k=1)

    res = asyncio.run(run())
    assert res and not isinstance(res[0], dict)
    assert isinstance(res[0], Chunk)


def test_snippet_refiner_accepts_object_facts_and_chunks() -> None:
    from uma.core.retrieval.rlm.snippet_refiner import SnippetRefiner

    class _Cfg:
        snippet_refiner_top_k = 3
        max_chunks = 2

    class _FactObj:
        def __init__(self):
            self.subject = "S"
            self.predicate = "STATES"
            self.object = "Something happened."
            self.meta = {}

    # chunks can be Chunk objects already; SnippetRefiner must normalize them.
    store = _ObjChunkStore()
    core = ChunkCore(store)

    async def run():
        return await core.search_chunks(query_embedding=[0.1], owner_type="user", owner_id="user:u1", k=1)

    chunks = asyncio.run(run())

    refiner = SnippetRefiner(llm=None, cfg=_Cfg())
    out = asyncio.run(refiner.refine(query_text="something", facts=[_FactObj()], chunks=chunks))
    assert isinstance(out, list)
