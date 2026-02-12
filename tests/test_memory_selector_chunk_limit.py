from __future__ import annotations

from uma.core.retrieval.selector import MemorySelector
from uma.types import Chunk
from datetime import datetime, timezone


def _mk(pos: int) -> Chunk:
    now = datetime.now(timezone.utc)
    return Chunk(
        id=f"chunk_{pos}",
        doc_id="doc1",
        text="x" * 200 + ".",
        page_range=(1, 1),
        position=pos,
        source_path="/tmp/x",
        source_hash="h",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:u1",
        meta={},
    )


def test_selector_uses_max_chunks_not_max_facts() -> None:
    sel = MemorySelector(max_episodes=1, max_facts=1, max_chunks=3, max_skills=1, max_graph_items=1)
    raw = {
        "working_memory": [],
        "episodes": [],
        "facts": [],
        "chunks": [_mk(1), _mk(2), _mk(3), _mk(4)],
        "skills": [],
        "graph": [],
    }
    out = sel.select(raw)
    assert len(out["chunks"]) == 3

