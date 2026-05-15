from __future__ import annotations


import pytest

from uma.api.management import lint_memory_drift
from uma.api.runtime import UMARuntime
from uma.common.types import RuntimeContext
from uma.ingest.ingest_service import capture_source, curate_compiled_memory, derive_memory_artifacts
from uma.ingest.types import IngestConfig
from uma.memory import wiki as wiki_module


def test_wiki_page_identity_is_deterministic() -> None:
    first = wiki_module.page_identity_for_key("Operations Platform")
    second = wiki_module.page_identity_for_key("operations platform")
    prefixed = wiki_module.page_identity_for_key("wiki:operations/platform")

    assert first["slug"] == "operations-platform"
    assert first == second
    assert prefixed == {"page_id": "wiki:operations/platform", "slug": "operations/platform"}


@pytest.mark.asyncio
async def test_regenerated_wiki_page_is_canonical_record_with_evidence_links(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "wiki-canonical.txt"
    path.write_text(
        "VictoriaMetrics stores long-term metrics for platform operations and auditing. "
        "The operations handbook explains retention, service health review, and evidence-backed incident analysis.\n"
    )
    await memory.ingest_document(str(path), owner_type="user", owner_id="user:u1")

    runtime = UMARuntime.from_memory(memory)
    context = RuntimeContext(
        tenant_id="default",
        agent_id=memory.agent_id or "agent-default",
        request_id="req-wiki-canonical",
        user_id="user:u1",
    )
    memory_result = await runtime.retrieve_memory(
        context,
        query_text="What stores long-term metrics?",
        memory_intent="continuity",
        include_debug=True,
    )

    page = wiki_module.regenerate_wiki_page(
        memory=memory,
        page_key="ops/metrics",
        title="Ops Metrics",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[memory_result["debug"]["compiled_answer"]],
        category="operations",
    )

    assert page["page_type"] == wiki_module.WIKI_PAGE_RECORD_TYPE
    assert page["page_id"] == "wiki:ops/metrics"
    assert page["compiled_artifact"]["id"] == page["page_id"]
    assert page["evidence_links"]["source_chunk_ids"]
    assert page["provenance"]["valid"] is True


def test_wiki_page_refresh_keeps_identity_when_title_changes(uma_memory) -> None:
    original = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="wiki:ops/metrics",
        title="Ops Metrics",
        owner_type="user",
        owner_id="user:u1",
        manual=True,
    )
    refreshed = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="wiki:ops/metrics",
        title="Platform Metrics",
        owner_type="user",
        owner_id="user:u1",
        existing_page=original,
        manual=True,
    )

    assert original["page_id"] == "wiki:ops/metrics"
    assert refreshed["page_id"] == original["page_id"]
    assert refreshed["slug"] == original["slug"]
    assert refreshed["title"] == "Platform Metrics"



@pytest.mark.asyncio
async def test_wiki_lint_reports_invalid_parent_lineage(uma_memory) -> None:
    manual_parent = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="manual/root",
        title="Manual Root",
        owner_type="user",
        owner_id="user:u1",
        manual=True,
    )
    child = wiki_module.regenerate_wiki_page(
        memory=uma_memory,
        page_key="manual/child",
        title="Manual Child",
        owner_type="user",
        owner_id="user:u1",
        parent_artifacts=[manual_parent["compiled_artifact"]],
    )

    lint_result = await lint_memory_drift(uma_memory, [child], user_id="user:u1")

    issues = {finding["issue"] for finding in lint_result["findings"]}
    assert lint_result["status"] == "issues_found"
    assert "invalid_provenance" in issues
    assert "broken_parent_lineage" in issues


@pytest.mark.asyncio
async def test_stage3_curation_uses_wiki_subsystem(uma_memory, tmp_path, monkeypatch) -> None:
    memory = uma_memory
    path = tmp_path / "wiki-stage3.txt"
    path.write_text(
        "OpenSearch indexes operational logs for investigation workflows. "
        "The runbook explains incident search, retention policy checks, and evidence-backed response steps.\n"
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
    seen: list[str] = []
    original = wiki_module.regenerate_wiki_page

    def wrapped(**kwargs):
        seen.append(str(kwargs["page_key"]))
        return original(**kwargs)

    monkeypatch.setattr(wiki_module, "regenerate_wiki_page", wrapped)

    curated = await curate_compiled_memory(capture, derive, memory=memory)

    assert seen == [capture.parsed.doc_id]
    assert curated.compiled_artifacts[0]["id"].startswith("wiki:")


