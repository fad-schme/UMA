from uma.core.ingest.chunker import chunk_sections
from uma.core.ingest.types import NormalizedSection


def test_chunk_ids_are_deterministic():
    sections = [
        NormalizedSection(section_id="s1", doc_id="doc1", text="hello world " * 200, page_range=(1, 1)),
        NormalizedSection(section_id="s2", doc_id="doc1", text="another section " * 200, page_range=(2, 2)),
    ]

    chunks_a = chunk_sections(sections, chunk_size_tokens=50, overlap_tokens=10)
    chunks_b = chunk_sections(sections, chunk_size_tokens=50, overlap_tokens=10)

    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert [c.position for c in chunks_a] == [c.position for c in chunks_b]


def test_chunk_ids_do_not_depend_on_section_iteration_order():
    sections_a = [
        NormalizedSection(section_id="s1", doc_id="doc1", text="hello world " * 200, page_range=(1, 1)),
        NormalizedSection(section_id="s2", doc_id="doc1", text="another section " * 200, page_range=(2, 2)),
    ]
    sections_b = list(reversed(sections_a))

    chunks_a = chunk_sections(sections_a, chunk_size_tokens=50, overlap_tokens=10)
    chunks_b = chunk_sections(sections_b, chunk_size_tokens=50, overlap_tokens=10)

    # IDs should be stable even if upstream section order changes.
    assert sorted([c.chunk_id for c in chunks_a]) == sorted([c.chunk_id for c in chunks_b])
