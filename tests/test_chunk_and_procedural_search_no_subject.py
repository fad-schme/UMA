import pytest

from uma.core.chunk.core import ChunkCore
from uma.types_chunk import Chunk
from uma.core.procedural.core import ProceduralCore
from datetime import datetime, timezone


class DummyChunkStore:
    async def search(self, **kwargs):
        return [
            Chunk(
                id="c1",
                doc_id="d1",
                text="t",
                page_range=(1, 1),
                position=1,
                source_path="",
                source_hash="",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                owner_type=kwargs.get("owner_type", "agent"),
                owner_id=kwargs.get("owner_id", "agent-default"),
                meta={},
            )
        ]

    async def search_text(self, query_text, **kwargs):
        return [
            Chunk(
                id="c2",
                doc_id="d1",
                text=query_text,
                page_range=(1, 1),
                position=1,
                source_path="",
                source_hash="",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                owner_type=kwargs.get("owner_type", "agent"),
                owner_id=kwargs.get("owner_id", "agent-default"),
                meta={},
            )
        ]


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
    assert res and res[0].id == "c1"

    res = await core.search_chunks(
        query_embedding=[0.0],
        owner_type="agent",
        owner_id="agent-default",
        k=5,
        query_text="hello",
        lexical_k=5,
        filter_terms=False,
    )
    assert res and res[0].id == "c2"


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
