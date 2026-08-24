"""Bootstrap APIs: load_memory_bootstrap, load_daily_diary_bootstrap.

Covers public bootstrap method surface, explicit user_id requirement,
delegation to ingest service, and result shapes.
"""
from __future__ import annotations
from pathlib import Path
from tests.helpers.runtime import TEST_AGENT_ID, build_test_config
from uma.api.memory import UMAMemory
from uma.ingest import ingest_service
import pytest
import yaml

AGENT_ID = TEST_AGENT_ID

# ── test_bootstrap_ingest_surface ──────────────────────────────────────────







def _build_unbound_memory(tmp_path) -> UMAMemory:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)
    cfg = build_test_config(db_root=db_root)
    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    memory = UMAMemory.from_yaml(str(cfg_path))
    memory._ensure_ingestion_ready()
    return memory


def test_bootstrap_methods_remain_public_on_umamemory() -> None:
    assert hasattr(UMAMemory, "load_memory_bootstrap")
    assert hasattr(UMAMemory, "load_daily_diary_bootstrap")
    # The USER.md / SOUL.md profile-overlay loaders were removed: nothing in
    # UMA consumed the overlay through a supported surface.
    assert not hasattr(UMAMemory, "load_userprofile")
    assert not hasattr(UMAMemory, "load_agentprofile")


@pytest.mark.asyncio
async def test_load_memory_bootstrap_requires_explicit_user_id(tmp_path) -> None:
    memory = _build_unbound_memory(tmp_path)
    try:
        with pytest.raises(ValueError, match="explicit user_id"):
            await memory.load_memory_bootstrap(str(tmp_path / "MEMORY.md"), agent_id=AGENT_ID)
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_load_daily_diary_bootstrap_requires_explicit_user_id(tmp_path) -> None:
    memory = _build_unbound_memory(tmp_path)
    try:
        with pytest.raises(ValueError, match="explicit user_id"):
            await memory.load_daily_diary_bootstrap(str(tmp_path / "2026-05-05.md"), agent_id=AGENT_ID)
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
    result = await uma_memory.load_memory_bootstrap(
        str(bootstrap_path),
        user_id="user:u1",
        request_id="req-memory-bootstrap",
        session_id="session-bootstrap",
        config={"mode": "test"},
        agent_id=AGENT_ID,
    )

    assert result == {"status": "ingested", "path": str(bootstrap_path), "facts_created": 1}
    assert calls["file_path"] == str(bootstrap_path)
    assert calls["memory"] is uma_memory
    assert calls["config"] == {"mode": "test"}
    assert calls["runtime_context"] is not None
    assert calls["runtime_context"].user_id == "user:u1"
    assert calls["runtime_context"].session_id == "session-bootstrap"


@pytest.mark.asyncio
async def test_daily_diary_bootstrap_wrapper_delegates_to_ingest_service(uma_memory, tmp_path, monkeypatch) -> None:
    diary_path = tmp_path / "2026-05-05.md"
    diary_path.write_text("- preserved chronology\n", encoding="utf-8")
    calls: dict[str, object] = {}

    async def _fake_ingest_daily_diary_bootstrap(file_path, *, memory, runtime_context, config=None):
        calls["file_path"] = file_path
        calls["memory"] = memory
        calls["runtime_context"] = runtime_context
        calls["config"] = config
        return {"status": "ingested", "path": file_path, "episodes_created": 1}

    monkeypatch.setattr(ingest_service, "ingest_daily_diary_bootstrap", _fake_ingest_daily_diary_bootstrap)
    result = await uma_memory.load_daily_diary_bootstrap(
        str(diary_path),
        user_id="user:u1",
        request_id="req-diary-bootstrap",
        session_id="session-diary",
        agent_id=AGENT_ID,
    )

    assert result == {"status": "ingested", "path": str(diary_path), "episodes_created": 1}
    assert calls["file_path"] == str(diary_path)
    assert calls["memory"] is uma_memory
    assert calls["config"] is None
    assert calls["runtime_context"] is not None
    assert calls["runtime_context"].user_id == "user:u1"
    assert calls["runtime_context"].session_id == "session-diary"


@pytest.mark.asyncio
async def test_memory_bootstrap_preserves_skip_and_ingest_result_shapes(uma_memory, tmp_path) -> None:
    bootstrap_scope = {
        "user_id": "user:u1",
        "tenant_id": "default",
        "request_id": "req-memory-bootstrap-shape",
        "session_id": "session-memory-bootstrap",
    }
    missing = await uma_memory.load_memory_bootstrap(str(tmp_path / "missing-memory.md"), **bootstrap_scope, agent_id=AGENT_ID)
    assert missing["status"] == "skipped"
    assert missing["reason"] == "missing_file"

    empty_path = tmp_path / "empty-memory.md"
    empty_path.write_text("", encoding="utf-8")
    empty = await uma_memory.load_memory_bootstrap(str(empty_path), **bootstrap_scope, agent_id=AGENT_ID)
    assert empty["status"] == "skipped"
    assert empty["reason"] == "empty_file"

    no_entries_path = tmp_path / "headings-only-memory.md"
    no_entries_path.write_text("# Memory\n<!-- comment -->\n", encoding="utf-8")
    no_entries = await uma_memory.load_memory_bootstrap(str(no_entries_path), **bootstrap_scope, agent_id=AGENT_ID)
    assert no_entries["status"] == "skipped"
    assert no_entries["reason"] == "no_entries"

    ingest_path = tmp_path / "MEMORY.md"
    ingest_path.write_text(
        "# Memory\n- prefers espresso over drip coffee\n- reviews incidents before publishing summaries\n",
        encoding="utf-8",
    )
    first = await uma_memory.load_memory_bootstrap(str(ingest_path), **bootstrap_scope, agent_id=AGENT_ID)
    second = await uma_memory.load_memory_bootstrap(str(ingest_path), **bootstrap_scope, agent_id=AGENT_ID)

    assert first["status"] == "ingested"
    assert first["entries_found"] == 2
    assert first["facts_created"] == 2
    assert len(first["fact_ids"]) == 2
    assert second["status"] == "skipped"
    assert second["reason"] == "idempotent"
    assert second["entries_found"] == 2


@pytest.mark.asyncio
async def test_memory_bootstrap_persists_chunk_backed_provenance_for_retrieve_memory(uma_memory, tmp_path) -> None:
    bootstrap_path = tmp_path / "MEMORY.md"
    bootstrap_path.write_text(
        "# Memory\n- Favorite database: sqlite\n- Preferred editor: vim\n",
        encoding="utf-8",
    )

    result = await uma_memory.load_memory_bootstrap(
        str(bootstrap_path),
        user_id="user:u1",
        request_id="req-memory-bootstrap-provenance",
        session_id="session-memory-bootstrap-provenance",
        agent_id=AGENT_ID,
    )

    facts = await uma_memory.semantic_core.fetch_by_ids(
        result["fact_ids"],
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert len(facts) == 2

    source_chunk_ids = sorted({fact.source_ids[0] for fact in facts if getattr(fact, "source_ids", None)})
    assert len(source_chunk_ids) == 2
    assert all(not chunk_id.startswith("memory_bootstrap:") for chunk_id in source_chunk_ids)

    owner_chunks = await uma_memory.chunk_core._fetch_by_ids(
        source_chunk_ids,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u1",
    )
    assert len(owner_chunks) == 2
    assert {chunk.text for chunk in owner_chunks} == {"Favorite database: sqlite", "Preferred editor: vim"}

    leaked_chunks = await uma_memory.chunk_core._fetch_by_ids(
        source_chunk_ids,
        tenant_id="default",
        owner_type="user",
        owner_id="user:u2",
    )
    assert leaked_chunks == []

    memory_result = await uma_memory.retrieve_memory(
        query_text="sqlite",
        user_id="user:u1",
        request_id="req-memory-bootstrap-recall",
        session_id="session-memory-bootstrap-recall",
        agent_id=AGENT_ID,
    )

    assert memory_result.evidence
    assert memory_result.facts
    assert isinstance(memory_result.facts[0], dict)
    assert isinstance(memory_result.evidence[0], dict)
    assert memory_result.provenance_valid is True
    assert memory_result.provenance_error is None
    assert memory_result.product == "memory"
    # `memory_intent`, `compiled_answer`, `trace` are internal detail
    # not part of the MemoryResult surface — the model's `extra="forbid"`
    # config guarantees they cannot appear on the public return.

    other_user_result = await uma_memory.retrieve_memory(
        query_text="sqlite",
        user_id="user:u2",
        tenant_id="default",
        request_id="req-memory-bootstrap-other-user",
        session_id="session-memory-bootstrap-other-user",
        agent_id=AGENT_ID,
    )
    assert other_user_result.evidence == []


@pytest.mark.asyncio
async def test_daily_diary_bootstrap_preserves_skip_and_ingest_result_shapes(uma_memory, tmp_path) -> None:
    bootstrap_scope = {
        "user_id": "user:u1",
        "tenant_id": "default",
        "request_id": "req-diary-bootstrap-shape",
        "session_id": "session-diary-bootstrap",
    }
    missing = await uma_memory.load_daily_diary_bootstrap(str(tmp_path / "missing-diary.md"), **bootstrap_scope, agent_id=AGENT_ID)
    assert missing["status"] == "skipped"
    assert missing["reason"] == "missing_file"

    empty_path = tmp_path / "empty-diary.md"
    empty_path.write_text("", encoding="utf-8")
    empty = await uma_memory.load_daily_diary_bootstrap(str(empty_path), **bootstrap_scope, agent_id=AGENT_ID)
    assert empty["status"] == "skipped"
    assert empty["reason"] == "empty_file"

    no_entries_path = tmp_path / "notes.md"
    no_entries_path.write_text("not a bullet\nanother line\n", encoding="utf-8")
    no_entries = await uma_memory.load_daily_diary_bootstrap(str(no_entries_path), **bootstrap_scope, agent_id=AGENT_ID)
    assert no_entries["status"] == "skipped"
    assert no_entries["reason"] == "no_entries"

    diary_path = tmp_path / "2026-05-05.md"
    diary_path.write_text("- investigated alert routing\n- documented follow-up actions\n", encoding="utf-8")
    first = await uma_memory.load_daily_diary_bootstrap(str(diary_path), **bootstrap_scope, agent_id=AGENT_ID)
    second = await uma_memory.load_daily_diary_bootstrap(str(diary_path), **bootstrap_scope, agent_id=AGENT_ID)

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
    text_helper_source = Path("uma/common/text.py").read_text(encoding="utf-8")

    assert "_capture_bootstrap_source(" in ingest_source
    assert "_load_existing_manifest(" in ingest_source
    assert "_upsert_source_manifest(" in ingest_source
    assert "_build_memory_bootstrap_signature" not in text_helper_source
    assert "_persist_bootstrap_manifest" not in text_helper_source
