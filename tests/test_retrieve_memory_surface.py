"""Tests for the retrieve_memory product surface.

Covers: empty-result shape, end-to-end retrieval after ingest, user scope
isolation, and the include_debug flag from the public UMAMemory boundary.
"""
from __future__ import annotations

import pytest

from tests.helpers.runtime import init_uma_for_tests


@pytest.mark.asyncio
async def test_retrieve_memory_empty_result_shape(uma_memory) -> None:
    result = await uma_memory.retrieve_memory(
        query_text="something that does not exist",
        user_id="user:u1",
        tenant_id="default",
        request_id="req-mem-empty",
        session_id="session-mem-empty",
    )

    assert "facts" in result
    assert "evidence" in result
    assert "provenance_valid" in result
    assert isinstance(result["facts"], list)
    assert isinstance(result["evidence"], list)
    assert "product" not in result


@pytest.mark.asyncio
async def test_retrieve_memory_returns_facts_after_ingest(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)

    bootstrap_path = tmp_path / "MEMORY.md"
    bootstrap_path.write_text(
        "# Memory\n- prefers espresso over drip coffee\n- reviews incidents before publishing\n",
        encoding="utf-8",
    )
    result = await memory.load_memory_bootstrap(
        str(bootstrap_path),
        user_id="user:u1",
        tenant_id="default",
        request_id="req-bootstrap",
        session_id="session-bootstrap",
    )
    assert result["status"] == "ingested"
    assert result["facts_created"] == 2

    recalled = await memory.retrieve_memory(
        query_text="coffee preferences",
        user_id="user:u1",
        tenant_id="default",
        request_id="req-recall",
        session_id="session-recall",
    )

    assert "facts" in recalled
    assert "evidence" in recalled
    assert "provenance_valid" in recalled


@pytest.mark.asyncio
async def test_retrieve_memory_user_scope_isolation(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)

    bootstrap_path = tmp_path / "MEMORY.md"
    bootstrap_path.write_text(
        "# Memory\n- prefers dark roast coffee\n",
        encoding="utf-8",
    )
    await memory.load_memory_bootstrap(
        str(bootstrap_path),
        user_id="user:alice",
        tenant_id="default",
        request_id="req-alice-ingest",
        session_id="session-alice",
    )

    bob_result = await memory.retrieve_memory(
        query_text="coffee preferences",
        user_id="user:bob",
        tenant_id="default",
        request_id="req-bob-recall",
        session_id="session-bob",
    )

    assert bob_result["facts"] == []


@pytest.mark.asyncio
async def test_retrieve_memory_include_debug_flag(uma_memory) -> None:
    result = await uma_memory.retrieve_memory(
        query_text="test query",
        user_id="user:u1",
        tenant_id="default",
        request_id="req-mem-debug",
        session_id="session-mem-debug",
        include_debug=True,
    )

    assert "facts" in result
    assert "debug" in result
    assert result["debug"] is not None
