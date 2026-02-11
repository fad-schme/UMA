from __future__ import annotations

from uma.core.ingest.normalizer import _clean_page_text, _drop_repeated_lines_across_pages


def test_clean_page_text_dehyphenates_and_reflows_soft_wraps() -> None:
    raw = (
        "This doc describes inter-\n"
        "nal VLANs and internal ACLs\n"
        "tightly controlled (only restore processes can read them).\n"
        "\n"
        "Second paragraph starts here.\n"
    )
    cleaned = _clean_page_text(raw)
    assert "internal VLANs" in cleaned
    # Soft wrap should reflow line break into a space.
    assert "ACLs tightly controlled" in cleaned
    # Blank line boundary preserved (paragraph split still possible).
    assert "\n\n" in cleaned


def test_drop_repeated_lines_across_pages_removes_headers() -> None:
    pages = [
        "CONFIDENTIAL\nBody A line 1\nBody A line 2\n1\n",
        "CONFIDENTIAL\nBody B line 1\nBody B line 2\n2\n",
        "CONFIDENTIAL\nBody C line 1\nBody C line 2\n3\n",
    ]
    out = _drop_repeated_lines_across_pages(pages, min_repeats=3)
    assert len(out) == 3
    assert all("CONFIDENTIAL" not in p for p in out)

