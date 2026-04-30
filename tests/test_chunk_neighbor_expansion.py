from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.common.types import Chunk


def _mk(doc_id: str, pos: int, *, owner_type: str, owner_id: str) -> Chunk:
    now = datetime.now(timezone.utc)
    return Chunk(
        id=f"chunk_{doc_id}_{pos}",
        doc_id=doc_id,
        text=f"text {doc_id} {pos}.",
        page_range=(1, 1),
        position=pos,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type=owner_type,
        owner_id=owner_id,
        meta={},
    )


@pytest.mark.asyncio
async def test_expand_neighbors_single_anchor_window_1(uma_memory) -> None:
    memory = uma_memory
    owner_type = "user"
    owner_id = "user:u1"

    chunks = [_mk("d1", p, owner_type=owner_type, owner_id=owner_id) for p in range(1, 11)]
    embs = await memory.embedder.embed([c.text for c in chunks])
    for c, e in zip(chunks, embs):
        await memory.chunk_core.upsert_chunk(c, e)

    anchors = [_mk("d1", 5, owner_type=owner_type, owner_id=owner_id)]
    expanded = await memory.chunk_core.expand_neighbors(
        owner_type=owner_type,
        owner_id=owner_id,
        anchors=anchors,
        window=1,
        max_total=24,
    )
    assert [c.position for c in expanded] == [5, 4, 6]


@pytest.mark.asyncio
async def test_expand_neighbors_overlapping_anchors_dedupes(uma_memory) -> None:
    memory = uma_memory
    owner_type = "user"
    owner_id = "user:u1"

    chunks = [_mk("d1", p, owner_type=owner_type, owner_id=owner_id) for p in range(1, 11)]
    embs = await memory.embedder.embed([c.text for c in chunks])
    for c, e in zip(chunks, embs):
        await memory.chunk_core.upsert_chunk(c, e)

    anchors = [
        _mk("d1", 5, owner_type=owner_type, owner_id=owner_id),
        _mk("d1", 6, owner_type=owner_type, owner_id=owner_id),
    ]
    expanded = await memory.chunk_core.expand_neighbors(
        owner_type=owner_type,
        owner_id=owner_id,
        anchors=anchors,
        window=1,
        max_total=24,
    )
    assert [c.position for c in expanded] == [5, 4, 6, 7]


@pytest.mark.asyncio
async def test_expand_neighbors_enforces_max_total(uma_memory) -> None:
    memory = uma_memory
    owner_type = "user"
    owner_id = "user:u1"

    chunks = [_mk("d1", p, owner_type=owner_type, owner_id=owner_id) for p in range(1, 100)]
    embs = await memory.embedder.embed([c.text for c in chunks[:32]])
    # Keep this fast: only upsert a prefix large enough to cover anchors + window.
    for c, e in zip(chunks[:32], embs):
        await memory.chunk_core.upsert_chunk(c, e)

    anchors = [
        _mk("d1", 10, owner_type=owner_type, owner_id=owner_id),
        _mk("d1", 20, owner_type=owner_type, owner_id=owner_id),
        _mk("d1", 30, owner_type=owner_type, owner_id=owner_id),
    ]
    expanded = await memory.chunk_core.expand_neighbors(
        owner_type=owner_type,
        owner_id=owner_id,
        anchors=anchors,
        window=3,
        max_total=5,
    )
    assert len(expanded) == 5

