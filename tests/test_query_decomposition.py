"""Unit tests for uma.retrieve.rlm.query_decomposition.

Covers retrieval-ranking-gap ticket 03: query decomposition must degrade
safely (never raise, return []) on any failure — missing LLM, malformed
JSON, non-list payload — since a broken decomposition step must fall back
to the single-query chunk search path rather than breaking retrieval.
"""
from __future__ import annotations

import json

import pytest

from uma.retrieve.rlm.query_decomposition import decompose_query


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def generate(self, messages, max_tokens=200, temperature=0.0, **kwargs):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self._response


class _RaisingLLM:
    async def generate(self, messages, max_tokens=200, temperature=0.0, **kwargs):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_decompose_query_parses_sub_queries() -> None:
    llm = _FakeLLM(json.dumps({"sub_queries": ["hobby a", "hobby b", "hobby c"]}))
    out = await decompose_query(llm, "What activities does Melanie partake in?")
    assert out == ["hobby a", "hobby b", "hobby c"]


@pytest.mark.asyncio
async def test_decompose_query_caps_at_max_sub_queries() -> None:
    llm = _FakeLLM(json.dumps({"sub_queries": ["a", "b", "c", "d", "e", "f"]}))
    out = await decompose_query(llm, "What events has Caroline participated in?", max_sub_queries=2)
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_decompose_query_dedupes_case_insensitively_and_against_original() -> None:
    llm = _FakeLLM(json.dumps({"sub_queries": ["Hobby A", "hobby a", "What events has Caroline participated in?", "hobby b"]}))
    out = await decompose_query(llm, "What events has Caroline participated in?")
    assert out == ["Hobby A", "hobby b"]


@pytest.mark.asyncio
async def test_decompose_query_returns_empty_on_malformed_json() -> None:
    llm = _FakeLLM("not json at all")
    out = await decompose_query(llm, "What activities does Melanie partake in?")
    assert out == []


@pytest.mark.asyncio
async def test_decompose_query_returns_empty_when_sub_queries_not_a_list() -> None:
    llm = _FakeLLM(json.dumps({"sub_queries": "hobby a"}))
    out = await decompose_query(llm, "What activities does Melanie partake in?")
    assert out == []


@pytest.mark.asyncio
async def test_decompose_query_returns_empty_on_llm_failure() -> None:
    out = await decompose_query(_RaisingLLM(), "What activities does Melanie partake in?")
    assert out == []


@pytest.mark.asyncio
async def test_decompose_query_returns_empty_when_llm_is_none() -> None:
    out = await decompose_query(None, "What activities does Melanie partake in?")
    assert out == []


@pytest.mark.asyncio
async def test_decompose_query_returns_empty_for_blank_query() -> None:
    llm = _FakeLLM(json.dumps({"sub_queries": ["a"]}))
    out = await decompose_query(llm, "   ")
    assert out == []


@pytest.mark.asyncio
@pytest.mark.parametrize("severity", ["medium", "high", "MEDIUM", "High"])
async def test_decompose_query_skips_llm_when_query_scan_severity_gates_it(severity: str) -> None:
    """Every other LLM hop in the RLM pipeline skips on medium/high boundary-scan
    severity (controller.py's _llm_hops_disabled) — decomposition must too."""
    llm = _FakeLLM(json.dumps({"sub_queries": ["a", "b"]}))
    out = await decompose_query(
        llm, "What activities does Melanie partake in?", query_scan_severity=severity
    )
    assert out == []
    assert llm.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("severity", [None, "low", "none", ""])
async def test_decompose_query_runs_when_query_scan_severity_does_not_gate_it(severity) -> None:
    llm = _FakeLLM(json.dumps({"sub_queries": ["a", "b"]}))
    out = await decompose_query(
        llm, "What activities does Melanie partake in?", query_scan_severity=severity
    )
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_decompose_query_returns_empty_when_max_sub_queries_is_zero() -> None:
    llm = _FakeLLM(json.dumps({"sub_queries": ["a", "b"]}))
    out = await decompose_query(llm, "What activities does Melanie partake in?", max_sub_queries=0)
    assert out == []
    assert llm.calls == []
