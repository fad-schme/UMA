"""
test_pr6_input_validation.py
============================
PR6: caller-input validation on UMAMemory.ingest_document.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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
