from __future__ import annotations

import pytest

from uma.ingest.ingest_service import capture_source, curate_compiled_memory, derive_memory_artifacts
from uma.ingest.types import IngestConfig


@pytest.mark.asyncio
async def test_capture_source_persists_terminal_evidence_without_forcing_derivation(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "capture-only.txt"
    path.write_text(
        "Kubernetes handles production orchestration for shared services. "
        "The operations handbook documents deployment ownership, rollback steps, and service boundaries.\n"
    )

    capture = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        memory=memory,
    )

    assert capture.parsed.doc_id
    assert capture.captured_chunks
    assert capture.captured_chunk_inputs
    facts = await memory.semantic_core.list_facts_for_owner(
        owner_type="user",
        owner_id="user:u1",
        limit=None,
    )
    assert facts == []


@pytest.mark.asyncio
async def test_capture_source_rerun_is_idempotent_and_returns_existing_chunks(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "capture-rerun.txt"
    path.write_text(
        "Databases store durable records for tenant-scoped memory. "
        "Chunks remain the terminal evidence surface for retrieval and audit.\n"
    )

    first = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        memory=memory,
    )
    second = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        memory=memory,
    )

    assert first.skipped is False
    assert second.skipped is True
    assert second.early_report is not None
    assert len(second.captured_chunks) == len(first.captured_chunks)
    assert [chunk.id for chunk in second.captured_chunks] == [chunk.id for chunk in first.captured_chunks]


@pytest.mark.asyncio
async def test_derive_memory_artifacts_reruns_from_capture_outputs_without_reparsing(uma_memory, tmp_path, monkeypatch) -> None:
    memory = uma_memory
    path = tmp_path / "derive-rerun.txt"
    path.write_text(
        "Prometheus collects metrics for critical services. "
        "Operators use alerts and dashboards to inspect latency, saturation, and failures over time.\n"
    )
    config = IngestConfig(doc_episode_enabled=False)
    capture = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        config=config,
        memory=memory,
    )

    first = await derive_memory_artifacts(
        capture,
        config=config,
        memory=memory,
    )
    before = await memory.semantic_core.list_facts_for_owner(
        owner_type="user",
        owner_id="user:u1",
        limit=None,
    )
    monkeypatch.setattr("uma.ingest.ingest_service.parse_file", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parse_file should not run during derive")))
    second = await derive_memory_artifacts(
        capture,
        config=config,
        memory=memory,
    )
    after = await memory.semantic_core.list_facts_for_owner(
        owner_type="user",
        owner_id="user:u1",
        limit=None,
    )

    assert first.captured_chunk_inputs == second.captured_chunk_inputs
    assert len(after) == len(before)


@pytest.mark.asyncio
async def test_curate_compiled_memory_rebuilds_from_capture_and_derivation_outputs(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "curate-stage.txt"
    path.write_text(
        "Grafana dashboards summarize service health for operators. "
        "The on-call guide explains how dashboards, alerts, and service ownership fit together.\n"
    )
    config = IngestConfig(doc_episode_enabled=False)
    capture = await capture_source(
        str(path),
        owner_type="user",
        owner_id="user:u1",
        config=config,
        memory=memory,
    )
    derive = await derive_memory_artifacts(
        capture,
        config=config,
        memory=memory,
    )

    curated_a = await curate_compiled_memory(
        capture,
        derive,
        memory=memory,
    )
    curated_b = await curate_compiled_memory(
        capture,
        derive,
        memory=memory,
    )

    assert len(curated_a.compiled_artifacts) == 1
    assert curated_a.compiled_artifacts[0]["artifact_type"] == "compiled_memory_artifact"
    assert curated_a.index_entries[0]["artifact_id"] == curated_a.compiled_artifacts[0]["id"]
    assert curated_a.log_events
    assert curated_b.compiled_artifacts[0]["id"] == curated_a.compiled_artifacts[0]["id"]
    assert curated_b.compiled_artifacts[0]["provenance"]["source_chunk_ids"] == curated_a.compiled_artifacts[0]["provenance"]["source_chunk_ids"]


@pytest.mark.asyncio
async def test_ingest_document_orchestrates_capture_derive_and_curate(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "full-ingest.txt"
    path.write_text(
        "Elasticsearch supports log retrieval for operations teams. "
        "The service manual explains indexing, search, and incident investigation workflows.\n"
    )

    report = await memory.ingest_document(str(path), owner_type="user", owner_id="user:u1")

    assert report.doc_id
    assert report.chunks_created > 0
    assert report.facts_created >= 0
