from __future__ import annotations

import pytest

import uma.api.management as management_api
from uma.api.management import explain_result, lint_memory_drift
from uma.api.memory import UMAMemory
from uma.api.runtime import UMARuntime
from uma.common.types import RuntimeContext
from uma.memory import wiki as wiki_module


def test_memory_public_surface_keeps_animus_support_and_drops_management_methods() -> None:
    assert hasattr(UMAMemory, "load_userprofile")
    assert hasattr(UMAMemory, "load_agentprofile")
    assert hasattr(UMAMemory, "load_memory_bootstrap")
    assert hasattr(UMAMemory, "load_daily_diary_bootstrap")
    assert not hasattr(UMAMemory, "expand_evidence")
    assert not hasattr(UMAMemory, "compile_memory_artifact")
    assert not hasattr(UMAMemory, "explain_result")
    assert not hasattr(UMAMemory, "lint_memory_drift")


def test_management_module_exports_supported_operations_only() -> None:
    assert management_api.__all__ == [
        "explain_result",
        "lint_memory_drift",
        "QuarantinedRecord",
        "list_quarantined",
        "reinstate_quarantined",
        "purge_quarantined",
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
async def test_management_explain_uses_canonical_provenance(uma_memory, tmp_path) -> None:
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
        include_debug=True,
    )

    artifact = memory_result["debug"]["compiled_answer"]
    explanation = await explain_result(memory, artifact, user_id="user:u1")

    assert explanation["evidence"]
    assert explanation["chunk_ids"] == artifact["provenance"]["source_chunk_ids"]
    assert explanation["compiled_memory_index"]["artifact_id"] == artifact["id"]


@pytest.mark.asyncio
async def test_management_lint_reports_invalid_parent_lineage_without_rewriting(uma_memory) -> None:
    memory = uma_memory
    manual_parent = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:manual/root",
        title="Manual Root",
        owner_type="user",
        owner_id="user:u1",
        manual=True,
    )["compiled_artifact"]
    child = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="wiki:manual/child",
        title="Manual Child",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[manual_parent],
    )["compiled_artifact"]

    lint_result = await lint_memory_drift(memory, [child], user_id="user:u1", stale_after_seconds=0)

    issues = {finding["issue"] for finding in lint_result["findings"]}
    assert lint_result["status"] == "issues_found"
    assert "invalid_provenance" in issues
    assert "broken_parent_lineage" in issues
