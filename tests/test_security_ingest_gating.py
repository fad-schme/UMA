"""Ingest security gates: HTML/Markdown sanitization, MIME detection, file path validation.

Covers all barriers that run before any chunk reaches storage:
HTML sanitization (script/iframe/event handlers/js:), MIME consistency checks,
executable rejection, extension/content mismatch, input path validation.
"""
from __future__ import annotations
from uma.ingest.parser import _sanitize_html
from unittest.mock import AsyncMock, MagicMock, patch
import os
import pytest

from tests.helpers.runtime import TEST_AGENT_ID

AGENT_ID = TEST_AGENT_ID

# ── test_pr6_html_sanitization ──────────────────────────────────────────





def test_script_tags_removed():
    html = "<html><body><script>evil()</script><p>safe</p></body></html>"
    text, counts = _sanitize_html(html)
    assert "evil()" not in text
    assert "safe" in text
    assert counts.get("scripts", 0) >= 1


def test_iframe_removed():
    html = "<html><body><iframe src='http://evil.com'></iframe><p>ok</p></body></html>"
    text, counts = _sanitize_html(html)
    assert "evil.com" not in text
    assert "ok" in text
    assert counts.get("iframes", 0) >= 1


def test_inline_event_handler_stripped():
    html = '<p onclick="doEvil()">Click</p>'
    text, counts = _sanitize_html(html)
    assert "doEvil" not in text
    assert "Click" in text
    assert counts.get("event_handlers", 0) >= 1


def test_javascript_href_stripped():
    html = '<a href="javascript:void(0)">link</a>'
    text, counts = _sanitize_html(html)
    assert "javascript" not in text.lower()
    assert counts.get("js_urls", 0) >= 1


def test_data_uri_stripped():
    html = '<img src="data:image/png;base64,abc123">'
    text, counts = _sanitize_html(html)
    assert "data:" not in text
    assert counts.get("data_urls", 0) >= 1


def test_conditional_comment_stripped():
    html = "before <!--[if IE]><script>ie()</script><![endif]--> after"
    text, counts = _sanitize_html(html)
    assert "ie()" not in text
    assert "before" in text
    assert "after" in text
    assert counts.get("cond_comments", 0) >= 1


def test_clean_html_produces_zero_counts():
    html = "<html><body><p>Hello world</p><ul><li>item</li></ul></body></html>"
    text, counts = _sanitize_html(html)
    assert "Hello world" in text
    assert counts == {}


def test_noscript_removed():
    html = "<html><body><noscript>enable js</noscript><p>content</p></body></html>"
    text, counts = _sanitize_html(html)
    assert "enable js" not in text
    assert "content" in text


def test_object_embed_removed():
    html = "<html><body><object data='x.swf'></object><embed src='y.swf'><p>text</p></body></html>"
    text, counts = _sanitize_html(html)
    assert "x.swf" not in text
    assert "text" in text


def test_malicious_fixture_file():
    import os
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "security", "malicious.html")
    with open(fixture, encoding="utf-8") as fh:
        html = fh.read()
    text, counts = _sanitize_html(html)
    assert "alert" not in text
    assert "Safe content" in text
    assert sum(counts.values()) >= 3


def test_svg_removed():
    html = '<html><body><svg><script>evil()</script></svg><p>text</p></body></html>'
    text, counts = _sanitize_html(html)
    assert "evil()" not in text
    assert "text" in text


# ── test_pr6_mime_detection ──────────────────────────────────────────



FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "security")


def fixture(name: str) -> str:
    return os.path.join(FIXTURES, name)


# ----- detect_content_type -----

def test_detect_pdf():
    from uma.ingest.mime_check import detect_content_type
    ct = detect_content_type(fixture("real.pdf"))
    assert ct.detected == "application/pdf"
    assert not ct.is_binary


def test_detect_pe_exe():
    from uma.ingest.mime_check import detect_content_type
    ct = detect_content_type(fixture("evil.exe"))
    assert ct.detected == "application/x-dosexec"
    assert ct.is_binary


def test_detect_elf():
    from uma.ingest.mime_check import detect_content_type
    ct = detect_content_type(fixture("evil.elf"))
    assert ct.detected == "application/x-elf"
    assert ct.is_binary


def test_detect_macho():
    from uma.ingest.mime_check import detect_content_type
    ct = detect_content_type(fixture("evil.macho"))
    assert ct.detected == "application/x-mach-binary"
    assert ct.is_binary


def test_detect_html_masquerading_as_pdf():
    from uma.ingest.mime_check import detect_content_type
    ct = detect_content_type(fixture("disguised.pdf"))
    # HTML bytes don't match any binary signature → text/plain
    assert ct.detected == "text/plain"
    assert not ct.is_binary


def test_detect_plain_text(tmp_path):
    from uma.ingest.mime_check import detect_content_type
    f = tmp_path / "note.txt"
    f.write_text("just some text")
    ct = detect_content_type(str(f))
    assert ct.detected == "text/plain"
    assert not ct.is_binary


# ----- check_mime_consistency -----

def test_pdf_extension_matches_pdf_detected():
    from uma.ingest.mime_check import check_mime_consistency, ContentType
    result = check_mime_consistency(".pdf", ContentType(detected="application/pdf", is_binary=False))
    assert result.consistent


def test_pdf_extension_mismatches_text():
    from uma.ingest.mime_check import check_mime_consistency, ContentType
    result = check_mime_consistency(".pdf", ContentType(detected="text/plain", is_binary=False))
    assert not result.consistent


def test_exe_always_inconsistent_regardless_of_ext():
    from uma.ingest.mime_check import check_mime_consistency, ContentType
    result = check_mime_consistency(".exe", ContentType(detected="application/x-dosexec", is_binary=True))
    assert not result.consistent


def test_txt_extension_allows_text_plain():
    from uma.ingest.mime_check import check_mime_consistency, ContentType
    result = check_mime_consistency(".txt", ContentType(detected="text/plain", is_binary=False))
    assert result.consistent


def test_unknown_extension_is_allowed_through():
    from uma.ingest.mime_check import check_mime_consistency, ContentType
    result = check_mime_consistency(".xyz", ContentType(detected="text/plain", is_binary=False))
    assert result.consistent


# ----- enforce_mime_consistency -----

def test_enforce_raises_mime_rejection_for_exe():
    from uma.ingest.mime_check import enforce_mime_consistency, MimeRejection
    with pytest.raises(MimeRejection) as exc_info:
        enforce_mime_consistency(fixture("evil.exe"))
    err = exc_info.value
    assert err.detected_type == "application/x-dosexec"
    assert "mime mismatch" in str(err)


def test_enforce_raises_for_disguised_pdf():
    from uma.ingest.mime_check import enforce_mime_consistency, MimeRejection
    with pytest.raises(MimeRejection):
        enforce_mime_consistency(fixture("disguised.pdf"))


def test_enforce_passes_for_real_pdf():
    from uma.ingest.mime_check import enforce_mime_consistency
    ct = enforce_mime_consistency(fixture("real.pdf"))
    assert ct.detected == "application/pdf"


def test_mime_rejection_carries_fields():
    from uma.ingest.mime_check import MimeRejection
    err = MimeRejection(detected_type="application/x-elf", declared_extension=".txt", file_path="/tmp/x.txt")
    assert err.detected_type == "application/x-elf"
    assert err.declared_extension == ".txt"
    assert err.file_path == "/tmp/x.txt"
    assert "mime mismatch" in str(err)


# ── test_pr6_input_validation ──────────────────────────────────────────




def _make_memory():
    from unittest.mock import MagicMock
    m = MagicMock()
    m._ensure_ingestion_ready = MagicMock(return_value=None)
    return m


@pytest.mark.asyncio
async def test_none_file_path_raises_value_error(tmp_path):
    from uma.api.memory import UMAMemory
    mem = MagicMock(spec=UMAMemory)
    mem._ensure_ingestion_ready = MagicMock()
    mem.ingest_document = UMAMemory.ingest_document.__get__(mem, UMAMemory)

    with pytest.raises(ValueError, match="file_path is required"):
        await mem.ingest_document(None)


@pytest.mark.asyncio
async def test_empty_string_raises_value_error(tmp_path):
    from uma.api.memory import UMAMemory
    mem = MagicMock(spec=UMAMemory)
    mem._ensure_ingestion_ready = MagicMock()
    mem.ingest_document = UMAMemory.ingest_document.__get__(mem, UMAMemory)

    with pytest.raises(ValueError, match="file_path is required"):
        await mem.ingest_document("")


@pytest.mark.asyncio
async def test_whitespace_only_raises_value_error(tmp_path):
    from uma.api.memory import UMAMemory
    mem = MagicMock(spec=UMAMemory)
    mem._ensure_ingestion_ready = MagicMock()
    mem.ingest_document = UMAMemory.ingest_document.__get__(mem, UMAMemory)

    with pytest.raises(ValueError, match="file_path is required"):
        await mem.ingest_document("   ")


@pytest.mark.asyncio
async def test_nonexistent_path_raises_file_not_found(tmp_path):
    from uma.api.memory import UMAMemory
    mem = MagicMock(spec=UMAMemory)
    mem._ensure_ingestion_ready = MagicMock()
    mem.ingest_document = UMAMemory.ingest_document.__get__(mem, UMAMemory)

    with pytest.raises(FileNotFoundError, match="file not found"):
        await mem.ingest_document(str(tmp_path / "does_not_exist.txt"))


@pytest.mark.asyncio
async def test_directory_path_raises_value_error(tmp_path):
    from uma.api.memory import UMAMemory
    mem = MagicMock(spec=UMAMemory)
    mem._ensure_ingestion_ready = MagicMock()
    mem.ingest_document = UMAMemory.ingest_document.__get__(mem, UMAMemory)

    with pytest.raises(ValueError, match="regular file"):
        await mem.ingest_document(str(tmp_path))


@pytest.mark.asyncio
async def test_valid_path_passes_validation_and_calls_ingest(tmp_path):
    """Validation passes for a real file; the downstream ingest is called."""
    from uma.api.memory import UMAMemory

    real_file = tmp_path / "doc.txt"
    real_file.write_text("hello")

    mem = MagicMock(spec=UMAMemory)
    mem._ensure_ingestion_ready = MagicMock()
    mem.ingest_document = UMAMemory.ingest_document.__get__(mem, UMAMemory)

    mock_ingest = AsyncMock(return_value="report")
    with patch("uma.ingest.ingest_service.ingest_document", mock_ingest):
        result = await mem.ingest_document(str(real_file))

    assert result == "report"
    mock_ingest.assert_awaited_once()


# ── test_pr6_markdown_embedded_html ──────────────────────────────────────────



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


# ── test_pr6_ingest_rejection_propagation ──────────────────────────────────────────



FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "security")


@pytest.mark.asyncio
async def test_capture_source_rejects_exe(tmp_path):
    from tests.helpers.runtime import init_uma_for_tests
    from uma.ingest.ingest_service import capture_source
    from uma.ingest.mime_check import MimeRejection

    memory = await init_uma_for_tests(tmp_path)
    exe_path = os.path.join(FIXTURES, "evil.exe")

    with pytest.raises(MimeRejection):
        await capture_source(
            exe_path,
            owner_type="agent",
            owner_id="agent:test",
            tenant_id="default",
            memory=memory,
        )


@pytest.mark.asyncio
async def test_capture_source_rejects_disguised_pdf(tmp_path):
    from tests.helpers.runtime import init_uma_for_tests
    from uma.ingest.ingest_service import capture_source
    from uma.ingest.mime_check import MimeRejection

    memory = await init_uma_for_tests(tmp_path)
    disguised = os.path.join(FIXTURES, "disguised.pdf")

    with pytest.raises(MimeRejection):
        await capture_source(
            disguised,
            owner_type="agent",
            owner_id="agent:test",
            tenant_id="default",
            memory=memory,
        )


@pytest.mark.asyncio
async def test_ingest_document_rejects_exe(tmp_path):
    from tests.helpers.runtime import init_uma_for_tests
    from uma.ingest.mime_check import MimeRejection

    memory = await init_uma_for_tests(tmp_path)
    exe_path = os.path.join(FIXTURES, "evil.exe")

    with pytest.raises(MimeRejection):
        await memory.ingest_document(exe_path, owner_type="agent", owner_id="agent:test")


@pytest.mark.asyncio
async def test_sanitization_counts_in_manifest_meta(tmp_path):
    """Manifest meta.security.sanitization records counts for sanitized HTML."""
    from uma.ingest.ingest_service import _merge_manifest_meta
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    counts = {"scripts": 2, "iframes": 1}
    meta = _merge_manifest_meta(
        existing={},
        ingest_signature={"source_hash": "abc"},
        now=now,
        sanitization_counts=counts,
    )
    assert meta["security"]["sanitization"] == counts


@pytest.mark.asyncio
async def test_sanitization_counts_absent_for_clean_file(tmp_path):
    """Manifest meta has no security.sanitization key when no sanitization occurred."""
    from uma.ingest.ingest_service import _merge_manifest_meta
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    meta = _merge_manifest_meta(
        existing={},
        ingest_signature={"source_hash": "abc"},
        now=now,
        sanitization_counts=None,
    )
    assert "sanitization" not in meta.get("security", {})


@pytest.mark.asyncio
async def test_mime_rejection_error_message():
    from uma.ingest.mime_check import MimeRejection
    err = MimeRejection(
        detected_type="application/x-elf",
        declared_extension=".txt",
        file_path="/tmp/bad.txt",
    )
    assert "mime mismatch" in str(err).lower()
    assert err.detected_type == "application/x-elf"
    assert err.declared_extension == ".txt"
