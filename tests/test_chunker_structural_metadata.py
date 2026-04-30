from __future__ import annotations

import pytest

from uma.ingest.chunker import finalize_chunks
from uma.ingest.types import DocumentChunk


def _mk(
    *,
    doc_id: str = "doc_1",
    page_range: tuple[int, int] = (1, 1),
    position: int,
    text: str,
    p_start: int | None,
    p_end: int | None,
    char_start: int | None = None,
    char_end: int | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"c_{position}",
        doc_id=doc_id,
        text=text,
        page_range=page_range,
        position=position,
        paragraph_index_start=p_start,
        paragraph_index_end=p_end,
        char_start=char_start,
        char_end=char_end,
    )


def test_finalize_chunks_merges_short_and_preserves_paragraph_ranges() -> None:
    # 2nd chunk is < _MIN_CHUNK_CHARS so it should merge backward.
    chunks = [
        _mk(
            position=1,
            text=("A" * 90) + ".",
            p_start=0,
            p_end=1,
        ),
        _mk(
            position=2,
            text="Short.",
            p_start=2,
            p_end=2,
        ),
        _mk(
            position=3,
            text=("B" * 90) + ".",
            p_start=3,
            p_end=4,
        ),
    ]

    out = finalize_chunks(chunks)

    assert len(out) == 2
    assert out[0].paragraph_index_start == 0
    assert out[0].paragraph_index_end == 2
    assert out[1].paragraph_index_start == 3
    assert out[1].paragraph_index_end == 4


def test_finalize_chunks_terminal_merge_preserves_paragraph_ranges() -> None:
    # First chunk is non-terminal; it should merge forward with the second.
    chunks = [
        _mk(
            position=1,
            text=("A" * 90),  # no terminal punctuation
            p_start=5,
            p_end=5,
        ),
        _mk(
            position=2,
            text=("B" * 90) + ".",
            p_start=6,
            p_end=7,
        ),
    ]

    out = finalize_chunks(chunks)

    assert len(out) == 1
    assert out[0].paragraph_index_start == 5
    assert out[0].paragraph_index_end == 7


def test_finalize_chunks_char_ranges_propagate_min_max_when_present() -> None:
    chunks = [
        _mk(
            position=1,
            text=("A" * 90) + ".",
            p_start=0,
            p_end=0,
            char_start=100,
            char_end=199,
        ),
        _mk(
            position=2,
            text="Short.",
            p_start=1,
            p_end=1,
            char_start=200,
            char_end=249,
        ),
    ]

    out = finalize_chunks(chunks)

    assert len(out) == 1
    assert out[0].char_start == 100
    assert out[0].char_end == 249


def test_finalize_chunks_rejects_missing_paragraph_indices() -> None:
    # If emission ever regresses and paragraph indices are missing, we want this to fail loudly.
    chunks = [
        _mk(position=1, text=("A" * 90) + ".", p_start=None, p_end=None),
        _mk(position=2, text=("B" * 90) + ".", p_start=1, p_end=1),
    ]

    with pytest.raises(ValueError, match="missing paragraph indices"):
        finalize_chunks(chunks)


def test_finalize_chunks_allows_missing_char_offsets() -> None:
    chunks = [
        _mk(position=1, text=("A" * 90) + ".", p_start=0, p_end=0, char_start=None, char_end=None),
        _mk(position=2, text=("B" * 90) + ".", p_start=1, p_end=1, char_start=None, char_end=None),
    ]

    out = finalize_chunks(chunks)
    assert len(out) == 2
    assert out[0].char_start is None and out[0].char_end is None
    assert out[1].char_start is None and out[1].char_end is None


def test_finalize_chunks_rejects_partial_char_offsets() -> None:
    chunks = [
        _mk(position=1, text=("A" * 90) + ".", p_start=0, p_end=0, char_start=0, char_end=None),
        _mk(position=2, text=("B" * 90) + ".", p_start=1, p_end=1, char_start=None, char_end=None),
    ]

    with pytest.raises(ValueError, match="char_start/char_end"):
        finalize_chunks(chunks)
