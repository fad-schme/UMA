"""
test_pr6_markdown_embedded_html.py
====================================
PR6: MarkdownParser sanitizes HTML embedded in markdown source.
"""
from __future__ import annotations

import os
import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "security")


def test_markdown_with_embedded_script_is_sanitized(tmp_path):
    from uma.ingest.parser import MarkdownParser
    f = tmp_path / "test.md"
    f.write_text("# Title\n\n<script>alert('xss')</script>\n\nSafe paragraph.")
    parser = MarkdownParser()
    text = parser.read(str(f))
    assert "alert" not in text
    assert "Safe paragraph" in text


def test_markdown_sanitization_counts_recorded(tmp_path):
    from uma.ingest.parser import MarkdownParser
    f = tmp_path / "test.md"
    f.write_text("# H\n\n<script>bad()</script>\n\n<iframe src='x'></iframe>\n\nOK.")
    parser = MarkdownParser()
    parser.read(str(f))
    counts = parser._last_sanitization_counts
    assert counts is not None
    assert sum(counts.values()) >= 1


def test_clean_markdown_has_no_sanitization_counts(tmp_path):
    from uma.ingest.parser import MarkdownParser
    f = tmp_path / "clean.md"
    f.write_text("# Title\n\nJust plain text with **bold** and *italic*.")
    parser = MarkdownParser()
    parser.read(str(f))
    # clean content should produce no counts (None or empty dict)
    counts = parser._last_sanitization_counts
    assert not counts  # None or {}


def test_markdown_fixture_with_embedded_script():
    from uma.ingest.parser import MarkdownParser
    fixture_path = os.path.join(FIXTURES, "embedded.md")
    parser = MarkdownParser()
    text = parser.read(fixture_path)
    assert "alert" not in text
    assert "Safe paragraph" in text or "Another paragraph" in text


def test_markdown_parse_threads_sanitization_counts_to_document(tmp_path):
    from uma.ingest.parser import FileContentParser
    f = tmp_path / "doc.md"
    f.write_text("# H\n\n<script>bad()</script>\n\nContent here.")
    parser = FileContentParser()
    doc = parser.parse(str(f))
    # sanitization_counts should be None for clean files; non-None when something was stripped
    # here a script was stripped so counts should be present
    assert doc.sanitization_counts is not None
    assert sum(doc.sanitization_counts.values()) >= 1


def test_html_parse_threads_sanitization_counts_to_document(tmp_path):
    from uma.ingest.parser import FileContentParser
    f = tmp_path / "doc.html"
    f.write_text("<html><body><script>bad()</script><p>text</p></body></html>")
    parser = FileContentParser()
    doc = parser.parse(str(f))
    assert doc.sanitization_counts is not None
    assert doc.sanitization_counts.get("scripts", 0) >= 1


def test_clean_html_parse_sanitization_counts_is_none(tmp_path):
    from uma.ingest.parser import FileContentParser
    f = tmp_path / "clean.html"
    f.write_text("<html><body><p>Hello world</p></body></html>")
    parser = FileContentParser()
    doc = parser.parse(str(f))
    assert not doc.sanitization_counts
