"""Tests for UMAMemory.health_check()."""
from __future__ import annotations

import pytest

from tests.helpers.runtime import init_uma_for_tests
from uma.api.memory import UMAMemory


_EXPECTED_CHECK_KEYS = {
    "db:episodic",
    "db:semantic",
    "db:procedural",
    "vector:episodic",
    "vector:semantic",
    "vector:procedural",
    "graph",
    "llm",
    "embedding",
}


@pytest.mark.asyncio
async def test_health_check_returns_ok_or_degraded_on_initialized_instance(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)
    result = memory.health_check()

    assert result["status"] in ("ok", "degraded")
    assert "checks" in result
    assert _EXPECTED_CHECK_KEYS.issubset(result["checks"].keys())

    for check in result["checks"].values():
        assert "name" in check
        assert "status" in check
        assert check["status"] in ("ok", "error", "skipped")


@pytest.mark.asyncio
async def test_health_check_returns_error_when_not_initialized(tmp_path) -> None:
    memory = await init_uma_for_tests(tmp_path)
    # Simulate the defensive guard: initialized=False is only reachable if
    # someone constructs UMAMemory directly (not via from_yaml) without warmup.
    memory.initialized = False

    result = memory.health_check()

    assert result["status"] == "error"
    assert "checks" in result
    assert "memory" in result["checks"]
    assert result["checks"]["memory"]["status"] == "error"
