from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from uma.api.memory import UMAMemory
from uma.ingest import ingest_service

from tests.helpers.runtime import build_test_config


def _build_unbound_memory(tmp_path) -> UMAMemory:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    memory = UMAMemory.from_yaml(str(cfg_path))
    memory._ensure_ingestion_ready()
    return memory


def test_animus_bootstrap_methods_remain_public_on_umamemory() -> None:
    assert hasattr(UMAMemory, "load_userprofile")
    assert hasattr(UMAMemory, "load_agentprofile")
    assert hasattr(UMAMemory, "load_memory_bootstrap")
    assert hasattr(UMAMemory, "load_daily_diary_bootstrap")


@pytest.mark.asyncio
async def test_load_memory_bootstrap_requires_bound_runtime_context(tmp_path) -> None:
    memory = _build_unbound_memory(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="bound runtime_context"):
            await memory.load_memory_bootstrap(str(tmp_path / "MEMORY.md"))
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_load_daily_diary_bootstrap_requires_bound_runtime_context(tmp_path) -> None:
    memory = _build_unbound_memory(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="bound runtime_context"):
            await memory.load_daily_diary_bootstrap(str(tmp_path / "2026-05-05.md"))
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_memory_bootstrap_wrapper_delegates_to_ingest_service(uma_memory, tmp_path, monkeypatch) -> None:
    bootstrap_path = tmp_path / "MEMORY.md"
    bootstrap_path.write_text("- prefers explicit ownership fields\n", encoding="utf-8")
    calls: dict[str, object] = {}

    async def _fake_ingest_memory_bootstrap(file_path, *, memory, runtime_context, config=None):
        calls["file_path"] = file_path
        calls["memory"] = memory
        calls["runtime_context"] = runtime_context
        calls["config"] = config
        return {"status": "ingested", "path": file_path, "facts_created": 1}

    monkeypatch.setattr(ingest_service, "ingest_memory_bootstrap", _fake_ingest_memory_bootstrap)
    result = await uma_memory.load_memory_bootstrap(str(bootstrap_path), config={"mode": "test"})

    assert result == {"status": "ingested", "path": str(bootstrap_path), "facts_created": 1}
    assert calls["file_path"] == str(bootstrap_path)
    assert calls["memory"] is uma_memory
    assert calls["config"] == {"mode": "test"}
    assert calls["runtime_context"] is not None


@pytest.mark.asyncio
async def test_daily_diary_bootstrap_wrapper_delegates_to_ingest_service(uma_memory, tmp_path, monkeypatch) -> None:
    diary_path = tmp_path / "2026-05-05.md"
    diary_path.write_text("- preserved chronology\n", encoding="utf-8")
    calls: dict[str, object] = {}

    async def _fake_ingest_daily_diary_bootstrap(file_path, *, memory, runtime_context):
        calls["file_path"] = file_path
        calls["memory"] = memory
        calls["runtime_context"] = runtime_context
        return {"status": "ingested", "path": file_path, "episodes_created": 1}

    monkeypatch.setattr(ingest_service, "ingest_daily_diary_bootstrap", _fake_ingest_daily_diary_bootstrap)
    result = await uma_memory.load_daily_diary_bootstrap(str(diary_path))

    assert result == {"status": "ingested", "path": str(diary_path), "episodes_created": 1}
    assert calls["file_path"] == str(diary_path)
    assert calls["memory"] is uma_memory
    assert calls["runtime_context"] is not None


@pytest.mark.asyncio
async def test_memory_bootstrap_preserves_skip_and_ingest_result_shapes(uma_memory, tmp_path) -> None:
    missing = await uma_memory.load_memory_bootstrap(str(tmp_path / "missing-memory.md"))
    assert missing["status"] == "skipped"
    assert missing["reason"] == "missing_file"

    empty_path = tmp_path / "empty-memory.md"
    empty_path.write_text("", encoding="utf-8")
    empty = await uma_memory.load_memory_bootstrap(str(empty_path))
    assert empty["status"] == "skipped"
    assert empty["reason"] == "empty_file"

    no_entries_path = tmp_path / "headings-only-memory.md"
    no_entries_path.write_text("# Memory\n<!-- comment -->\n", encoding="utf-8")
    no_entries = await uma_memory.load_memory_bootstrap(str(no_entries_path))
    assert no_entries["status"] == "skipped"
    assert no_entries["reason"] == "no_entries"

    ingest_path = tmp_path / "MEMORY.md"
    ingest_path.write_text(
        "# Memory\n- prefers espresso over drip coffee\n- reviews incidents before publishing summaries\n",
        encoding="utf-8",
    )
    first = await uma_memory.load_memory_bootstrap(str(ingest_path))
    second = await uma_memory.load_memory_bootstrap(str(ingest_path))

    assert first["status"] == "ingested"
    assert first["entries_found"] == 2
    assert first["facts_created"] == 2
    assert len(first["fact_ids"]) == 2
    assert second["status"] == "skipped"
    assert second["reason"] == "idempotent"
    assert second["entries_found"] == 2


@pytest.mark.asyncio
async def test_daily_diary_bootstrap_preserves_skip_and_ingest_result_shapes(uma_memory, tmp_path) -> None:
    missing = await uma_memory.load_daily_diary_bootstrap(str(tmp_path / "missing-diary.md"))
    assert missing["status"] == "skipped"
    assert missing["reason"] == "missing_file"

    empty_path = tmp_path / "empty-diary.md"
    empty_path.write_text("", encoding="utf-8")
    empty = await uma_memory.load_daily_diary_bootstrap(str(empty_path))
    assert empty["status"] == "skipped"
    assert empty["reason"] == "empty_file"

    no_entries_path = tmp_path / "notes.md"
    no_entries_path.write_text("not a bullet\nanother line\n", encoding="utf-8")
    no_entries = await uma_memory.load_daily_diary_bootstrap(str(no_entries_path))
    assert no_entries["status"] == "skipped"
    assert no_entries["reason"] == "no_entries"

    diary_path = tmp_path / "2026-05-05.md"
    diary_path.write_text("- investigated alert routing\n- documented follow-up actions\n", encoding="utf-8")
    first = await uma_memory.load_daily_diary_bootstrap(str(diary_path))
    second = await uma_memory.load_daily_diary_bootstrap(str(diary_path))

    assert first["status"] == "ingested"
    assert first["diary_date"] == "2026-05-05"
    assert first["entries_found"] == 2
    assert first["episodes_created"] == 2
    assert len(first["episode_ids"]) == 2
    assert second["status"] == "skipped"
    assert second["reason"] == "idempotent"
    assert second["entries_found"] == 2


def test_memory_module_no_longer_contains_bootstrap_ingest_mechanics() -> None:
    source = Path("uma/api/memory.py").read_text(encoding="utf-8")

    assert "build_fact_embedding_text" not in source
    assert "write_daily_diary_episodes" not in source
    assert "_extract_daily_diary_entries" not in source
    assert "_build_diary_bootstrap_signature" not in source
    assert "Fact(" not in source


def test_bootstrap_ingest_service_reuses_shared_capture_and_manifest_helpers() -> None:
    ingest_source = Path("uma/ingest/ingest_service.py").read_text(encoding="utf-8")
    retrieve_source = Path("uma/retrieve/user_query_helper.py").read_text(encoding="utf-8")

    assert "_capture_bootstrap_source(" in ingest_source
    assert "_load_existing_manifest(" in ingest_source
    assert "_upsert_source_manifest(" in ingest_source
    assert "_build_memory_bootstrap_signature" not in retrieve_source
    assert "_persist_bootstrap_manifest" not in retrieve_source
