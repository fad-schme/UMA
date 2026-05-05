from __future__ import annotations

import copy

import pytest

import uma.api.management as management_api
from uma.api.management import (
    explain_result,
    export_wiki_projection,
    lint_memory_drift,
    update_wiki_page,
)
from uma.api.memory import UMAMemory
from uma.api.runtime import UMARuntime
from uma.common.types import RuntimeContext


def test_memory_public_surface_keeps_animus_support_and_drops_management_methods() -> None:
    assert hasattr(UMAMemory, "load_userprofile")
    assert hasattr(UMAMemory, "load_agentprofile")
    assert hasattr(UMAMemory, "load_memory_bootstrap")
    assert hasattr(UMAMemory, "load_daily_diary_bootstrap")
    assert not hasattr(UMAMemory, "expand_evidence")
    assert not hasattr(UMAMemory, "compile_memory_artifact")
    assert not hasattr(UMAMemory, "update_wiki_page")
    assert not hasattr(UMAMemory, "export_wiki_projection")
    assert not hasattr(UMAMemory, "explain_result")
    assert not hasattr(UMAMemory, "lint_memory_drift")


def test_management_module_exports_supported_operations_only() -> None:
    assert management_api.__all__ == [
        "explain_result",
        "export_wiki_projection",
        "lint_memory_drift",
        "update_wiki_page",
    ]


def test_animus_profile_loaders_remain_public_and_functional(uma_memory, tmp_path) -> None:
    user_profile = tmp_path / "USER.md"
    agent_profile = tmp_path / "SOUL.md"
    user_profile.write_text("# User\nlikes coffee\n", encoding="utf-8")
    agent_profile.write_text("# Agent\nprefers concise answers\n", encoding="utf-8")

    assert uma_memory.load_userprofile(str(user_profile)) is uma_memory
    assert uma_memory.load_agentprofile(str(agent_profile)) is uma_memory
    assert "likes coffee" in uma_memory.animus_profile_provider.get_user_profile_text()
    assert "prefers concise answers" in uma_memory.animus_profile_provider.get_agent_profile_text()


@pytest.mark.asyncio
async def test_management_update_explain_and_export_use_canonical_provenance(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "management-doc.txt"
    path.write_text(
        "Prometheus backs metrics collection for production systems and feeds alerting workflows.\n"
    )
    await memory.ingest_document(str(path), owner_type="user", owner_id="user:u1")

    runtime = UMARuntime.from_memory(memory)
    context = RuntimeContext(
        tenant_id="default",
        agent_id=memory.agent_id or "agent-default",
        request_id="req-management",
        user_id="user:u1",
    )
    memory_result = await runtime.retrieve_memory(
        context,
        query_text="What handles production metrics?",
        memory_intent="continuity",
    )

    artifact_result = update_wiki_page(
        memory,
        artifact_id="wiki:ops/metrics",
        title="Ops Metrics",
        owner_type="user",
        owner_id="user:u1",
        summary="Metrics and alerting summary.",
        parent_artifacts=[memory_result["compiled_answer"]],
        related_artifact_ids=[memory_result["compiled_answer"]["id"]],
        retrieval_tags=["ops", "metrics"],
    )
    artifact = artifact_result["artifact"]
    before_export = copy.deepcopy(artifact)

    explanation = await explain_result(memory, artifact)
    assert artifact == before_export
    projection_path = tmp_path / "wiki" / "ops-metrics.md"
    export_result = await export_wiki_projection(memory, artifact, output_path=str(projection_path))

    assert artifact_result["operation"] == "wiki_artifact_created"
    assert explanation["evidence"]
    assert explanation["chunk_ids"] == artifact["provenance"]["source_chunk_ids"]
    assert explanation["compiled_memory_index"]["artifact_id"] == artifact["id"]
    assert export_result["projection_only"] is True
    assert export_result["path"] == str(projection_path)
    assert projection_path.read_text(encoding="utf-8").startswith("# Ops Metrics")
    assert artifact == before_export


@pytest.mark.asyncio
async def test_management_lint_reports_invalid_parent_lineage_without_rewriting(uma_memory) -> None:
    memory = uma_memory
    manual_parent = update_wiki_page(
        memory,
        artifact_id="wiki:manual/root",
        title="Manual Root",
        owner_type="user",
        owner_id="user:u1",
        manual=True,
    )["artifact"]
    child = update_wiki_page(
        memory,
        artifact_id="wiki:manual/child",
        title="Manual Child",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[manual_parent],
    )["artifact"]

    lint_result = await lint_memory_drift(memory, [child], stale_after_seconds=0)

    issues = {finding["issue"] for finding in lint_result["findings"]}
    assert lint_result["status"] == "issues_found"
    assert "invalid_provenance" in issues
    assert "broken_parent_lineage" in issues
