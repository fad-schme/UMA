from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from uma.core.chunk.core import ChunkCore
from uma.types_chunk import Chunk


def _mk(doc_id: str, pos: int, *, owner_type: str = "user", owner_id: str = "user:u1") -> Chunk:
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


class _NeighborStore:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def fetch_by_doc_and_position_range(self, *, owner_type, owner_id, doc_id, pos_start, pos_end, **_kwargs):
        out = []
        for ch in self._chunks:
            if ch.owner_type != owner_type or ch.owner_id != owner_id:
                continue
            if ch.doc_id != doc_id:
                continue
            if pos_start <= ch.position <= pos_end:
                out.append(ch)
        out.sort(key=lambda c: c.position)
        return out


def test_expand_neighbors_single_anchor_window_1() -> None:
    store = _NeighborStore([_mk("d1", p) for p in range(1, 11)])
    core = ChunkCore(store)
    anchors = [_mk("d1", 5)]

    async def run():
        return await core.expand_neighbors(owner_type="user", owner_id="user:u1", anchors=anchors, window=1, max_total=24)

    expanded = asyncio.run(run())
    assert [c.position for c in expanded] == [5, 4, 6]


def test_expand_neighbors_overlapping_anchors_dedupes() -> None:
    store = _NeighborStore([_mk("d1", p) for p in range(1, 11)])
    core = ChunkCore(store)
    anchors = [_mk("d1", 5), _mk("d1", 6)]

    async def run():
        return await core.expand_neighbors(owner_type="user", owner_id="user:u1", anchors=anchors, window=1, max_total=24)

    expanded = asyncio.run(run())
    # Anchor-first: 5 then its neighbors, then 6 (already included) then its neighbor 7.
    assert [c.position for c in expanded] == [5, 4, 6, 7]


def test_expand_neighbors_enforces_max_total() -> None:
    store = _NeighborStore([_mk("d1", p) for p in range(1, 100)])
    core = ChunkCore(store)
    anchors = [_mk("d1", 50), _mk("d1", 60), _mk("d1", 70)]

    async def run():
        return await core.expand_neighbors(owner_type="user", owner_id="user:u1", anchors=anchors, window=3, max_total=5)

    expanded = asyncio.run(run())
    assert len(expanded) == 5

