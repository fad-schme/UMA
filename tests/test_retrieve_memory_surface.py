"""Tests for the retrieve_memory product surface.

Covers: empty-result shape, end-to-end retrieval after ingest, user scope
isolation, and the include_debug flag from the public UMAMemory boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.helpers.runtime import init_uma_for_tests
from uma.common.types import Fact


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
async def test_retrieve_memory_third_person_fact_stays_within_same_user_scope(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)
    now = datetime.now(timezone.utc)

    fact_alice = Fact(
        id="fact_maria_alice",
        subject="Maria",
        predicate="has hair color",
        object="red hair",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:alice",
        tenant_id="default",
        confidence=0.9,
        salience=0.9,
    )
    fact_bob = Fact(
        id="fact_maria_bob",
        subject="Maria",
        predicate="has hair color",
        object="red hair",
        created_at=now,
        updated_at=now,
        owner_type="user",
        owner_id="user:bob",
        tenant_id="default",
        confidence=0.9,
        salience=0.9,
    )

    emb_alice, emb_bob = await memory.embedder.embed(
        [
            "Maria has hair color red hair",
            "Maria has hair color red hair",
        ]
    )
    await memory.semantic_core.upsert_fact(fact_alice, emb_alice)

    bob_before = await memory.retrieve_memory(
        query_text="Is Maria blond?",
        user_id="user:bob",
        tenant_id="default",
        request_id="req-bob-before-own-maria-fact",
        session_id="session-bob",
    )
    assert bob_before["facts"] == []

    await memory.semantic_core.upsert_fact(fact_bob, emb_bob)
    bob_after = await memory.retrieve_memory(
        query_text="Is Maria blond?",
        user_id="user:bob",
        tenant_id="default",
        request_id="req-bob-after-own-maria-fact",
        session_id="session-bob",
    )

    objects = {
        str(item.get("object") or item.get("text") or "").lower()
        for item in list(bob_after.get("facts") or [])
        if isinstance(item, dict)
    }
    assert any("red hair" in value for value in objects)


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
