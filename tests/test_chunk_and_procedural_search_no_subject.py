import pytest

from uma.core.chunk.core import ChunkCore
from uma.core.procedural.core import ProceduralCore


class DummyChunkStore:
    async def search(self, **kwargs):
        return [{"id": "c1", "text": "t"}]

    async def search_text(self, query_text, **kwargs):
        return [{"id": "c2", "text": query_text}]


class DummyProceduralStore:
    async def search(self, **kwargs):
        return [{"id": "s1", "name": "skill"}]


@pytest.mark.asyncio
async def test_chunk_search_does_not_require_subject():
    core = ChunkCore(DummyChunkStore())
    res = await core.search_chunks(
        query_embedding=[0.0],
        owner_type="agent",
        owner_id="agent-default",
        k=5,
    )
    assert res and res[0]["id"] == "c1"

    res = await core.search_chunks(
        query_embedding=[0.0],
        owner_type="agent",
        owner_id="agent-default",
        k=5,
        query_text="hello",
        lexical_k=5,
        filter_terms=False,
    )
    assert res and res[0]["id"] == "c2"


@pytest.mark.asyncio
async def test_procedural_search_does_not_require_subject():
    core = ProceduralCore(DummyProceduralStore())
    res = await core.search(
        user_id="u1",
        query_embedding=[0.0],
        owner_type="agent",
        owner_id="agent-default",
        k=5,
    )
    assert res and res[0]["id"] == "s1"
