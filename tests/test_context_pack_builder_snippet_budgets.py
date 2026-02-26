from __future__ import annotations

import pytest

from uma.core.utils.context_pack_builder import ContextPackBuilder


class _Cfg:
    snippet_refiner_enabled = True
    snippet_refiner_top_k = 6
    max_chunks = 2
    snippet_max_chars = 50

    max_working_messages = 0
    max_episodic = 0
    max_semantic = 0
    max_procedural = 0
    max_graph = 0
    include_working_memory = True
    include_episodic = True
    include_graph = True
    include_procedural = True
    allowed_topics = None


@pytest.mark.asyncio
async def test_render_snippet_async_enforces_snippet_budgets(uma_memory):
    base = (
        "This is a very long snippet sentence that should be trimmed at a sentence boundary. "
        "Second sentence."
    )
    pack = {
        "query": "q",
        "facts": [],
        "chunks": [
            # Use non-adjacent positions so SnippetRefiner grouping yields multiple snippet candidates.
            {"id": "c1", "doc_id": "d1", "text": base, "position": 1},
            {"id": "c2", "doc_id": "d1", "text": base + " Extra.", "position": 10},
            {"id": "c3", "doc_id": "d1", "text": base + " Extra extra.", "position": 20},
        ],
    }
    out = await ContextPackBuilder.render_snippet_async(pack=pack, context_cfg=_Cfg(), llm=uma_memory.llm)
    assert "Document snippets:" in out
    in_snips = False
    lines = []
    for ln in out.splitlines():
        if ln.strip() == "Document snippets:":
            in_snips = True
            continue
        if in_snips:
            if ln and not ln.startswith("- "):
                break
            if ln.startswith("- "):
                lines.append(ln)
    # max_chunks=2 => only two snippet lines
    assert len(lines) == 2
    # snippet_max_chars=50 => each line should be bounded (account for "- " prefix)
    assert all(len(ln[2:]) <= 50 for ln in lines)
