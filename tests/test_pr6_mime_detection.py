"""
test_pr6_mime_detection.py
==========================
PR6: MIME byte-signature detection and consistency checking.
"""
from __future__ import annotations

import os
import pytest

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
