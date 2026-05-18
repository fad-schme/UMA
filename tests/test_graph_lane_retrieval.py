from __future__ import annotations

import pytest
import pytest_asyncio

from tests.helpers.graph_adapter import RecordingGraphAdapter
from tests.helpers.runtime import init_uma_for_tests


@pytest_asyncio.fixture
async def uma_graph(tmp_path):
    """UMAMemory instance with RecordingGraphAdapter wired in."""
    mem = await init_uma_for_tests(
        tmp_path,
        graph_backend="tests.helpers.graph_adapter:RecordingGraphAdapter",
        graph_config={},
    )
    try:
        yield mem
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass


@pytest_asyncio.fixture
async def uma_no_graph(tmp_path):
    """UMAMemory instance with graph backend disabled."""
    mem = await init_uma_for_tests(tmp_path, graph_backend="disabled")
    try:
        yield mem
    finally:
        try:
            mem.shutdown()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_ingest_writes_graph_edges(uma_graph, tmp_path):
    """Graph adapter must receive Cypher writes after document ingest."""
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "The UMA memory system provides long-term knowledge retention for AI agents. "
        "It supports document ingestion pipelines that parse, chunk, embed, and index content. "
        "Fact extraction uses LLM prompts to derive structured subject-predicate-object triplets. "
        "Graph edges connect extracted facts enabling relational and temporal retrieval. "
        "The graph lane integrates with the RLM controller for structured context assembly."
    )

    adapter: RecordingGraphAdapter = uma_graph.graph_core.adapter
    queries_before = len(adapter.queries)

    report = await uma_graph.ingest_document(
        str(doc), owner_type="agent", owner_id="agent-default"
    )

    assert report.chunks_created > 0, "Expected at least one chunk to be created"
    assert len(adapter.queries) > queries_before, (
        "Expected graph Cypher queries after ingest; none recorded. "
        "Verify update_graph is called during fact extraction."
    )


@pytest.mark.asyncio
async def test_retrieve_context_with_graph_enabled(uma_graph, tmp_path):
    """retrieve_context must return a dict with a 'graph' key when graph is enabled."""
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "UMA supports structured graph memory via the TemporalGraphCore subsystem. "
        "Facts are extracted from documents and stored as subject-predicate-object triplets. "
        "The graph lane enables relational and temporal retrieval across documents and episodes. "
        "Graph edges carry full provenance: fact_id, owner_type, owner_id, source_chunk_id, timestamps."
    )
    await uma_graph.ingest_document(str(doc), owner_type="agent", owner_id="agent-default")

    result = await uma_graph.retrieve_context(query_text="How does UMA graph memory work?", user_id="user-test")

    assert isinstance(result, dict), "retrieve_context must return a dict"
    assert "graph" in result, "Result must include a 'graph' key when graph is enabled"


@pytest.mark.asyncio
async def test_retrieve_context_with_graph_disabled(uma_no_graph, tmp_path):
    """retrieve_context must not crash and return expected keys when graph is disabled."""
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "UMA is a modular memory runtime. It runs with or without a graph backend. "
        "Disabling graph still allows chunk and fact retrieval."
    )
    await uma_no_graph.ingest_document(str(doc), owner_type="agent", owner_id="agent-default")

    result = await uma_no_graph.retrieve_context(query_text="What is UMA?", user_id="user-test")

    assert isinstance(result, dict), "retrieve_context must return a dict even without graph"
    for key in ("chunks", "facts", "episodic"):
        assert key in result, f"Expected key '{key}' in context result"


@pytest.mark.asyncio
async def test_graph_disabled_graph_core_is_none(uma_no_graph):
    """When graph backend is 'disabled', graph_core must be None."""
    assert uma_no_graph.graph_core is None, (
        "graph_core should be None when graph_backend='disabled'"
    )
