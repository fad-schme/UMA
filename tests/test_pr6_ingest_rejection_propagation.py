"""
test_pr6_ingest_rejection_propagation.py
=========================================
PR6: MimeRejection propagates through capture_source and ingest_document.
Sanitization counts are stored in manifest meta.security.sanitization.
"""
from __future__ import annotations

import os
import pytest

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
    from tests.helpers.runtime import init_uma_for_tests
    from uma.ingest.ingest_service import capture_source, _merge_manifest_meta
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
