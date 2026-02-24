from __future__ import annotations

from uma.core.ingest.types import DocumentChunk
from uma.core.semantic.extractor import FactExtractor


def _mk(chunk_id: str, text: str, page_range=(1, 1), position=1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        text=text,
        page_range=page_range,
        position=position,
        paragraph_index_start=0,
        paragraph_index_end=0,
    )


def test_select_chunks_for_fact_extraction_is_deterministic() -> None:
    chunks = [
        _mk("c1", "Table of contents.\n" + ("x" * 500) + ".", page_range=(1, 1), position=1),
        _mk("c2", ("Architecture " * 50) + ".", page_range=(2, 2), position=2),
        _mk("c3", ("Design " * 50) + ".", page_range=(2, 2), position=3),
        _mk("c4", ("Risk " * 50) + ".", page_range=(3, 3), position=4),
    ]

    a = FactExtractor.select_chunks_for_fact_extraction(chunks, max_chunks=3)
    b = FactExtractor.select_chunks_for_fact_extraction(chunks, max_chunks=3)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_select_chunks_for_fact_extraction_caps_per_page() -> None:
    chunks = [
        _mk("c2", ("Architecture " * 50) + ".", page_range=(2, 2), position=2),
        _mk("c3", ("Design " * 50) + ".", page_range=(2, 2), position=3),
        _mk("c5", ("Controls " * 50) + ".", page_range=(2, 2), position=5),
        _mk("c4", ("Risk " * 50) + ".", page_range=(3, 3), position=4),
    ]
    out = FactExtractor.select_chunks_for_fact_extraction(chunks, max_chunks=4, max_per_page=2)
    assert sum(1 for c in out if c.page_range == (2, 2)) <= 2
