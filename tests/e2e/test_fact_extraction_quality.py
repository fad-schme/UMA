"""Opt-in fact-extraction quality measurement against a local Ollama model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

import pytest

from uma.adapters.llm.openai_compatible import OpenAICompatibleLLM
from uma.memory.semantic.extractor import FactExtractor


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_E2E") != "1",
        reason="set RUN_E2E=1 to run tests that require local Ollama",
    ),
]

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fact_extraction_heldout.json"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: Any) -> set[str]:
    return set(_TOKEN_RE.findall(str(value or "").lower()))


def _object_recall(expected: Any, actual: Any) -> float:
    expected_tokens = _tokens(expected)
    if not expected_tokens:
        return 1.0
    return len(expected_tokens & _tokens(actual)) / len(expected_tokens)


def _score_case(expected: list[dict[str, Any]], predicted: list[Any]) -> tuple[int, int, int]:
    unmatched = set(range(len(expected)))
    true_positives = 0
    for fact in predicted:
        predicate = str(getattr(fact, "predicate", "") or "").strip().upper()
        object_value = getattr(fact, "object", "")
        best_index = None
        best_score = 0.0
        for index in unmatched:
            gold = expected[index]
            if predicate != str(gold["predicate"]).strip().upper():
                continue
            score = _object_recall(gold["object"], object_value)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is not None and best_score >= 0.6:
            unmatched.remove(best_index)
            true_positives += 1
    return true_positives, len(predicted), len(expected)


async def test_local_ollama_fact_extraction_precision_recall() -> None:
    """Measure extraction quality without weakening the hermetic default suite."""
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm = OpenAICompatibleLLM(
        provider_name="ollama",
        model=model,
        base_url=f"{host.rstrip('/')}/v1",
        api_key="ollama",
        timeout=float(os.getenv("OLLAMA_TIMEOUT_S", "120")),
    )
    extractor = FactExtractor(llm)
    corpus = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

    true_positives = 0
    predicted_count = 0
    expected_count = 0
    case_results: list[dict[str, Any]] = []
    for case in corpus:
        predicted = await extractor.extract_user_facts(
            subject="heldout-user",
            text=case["text"],
            owner_type="user",
            owner_id="heldout-user",
        )
        matched, predicted_total, expected_total = _score_case(case["facts"], predicted)
        true_positives += matched
        predicted_count += predicted_total
        expected_count += expected_total
        case_results.append(
            {
                "id": case["id"],
                "matched": matched,
                "predicted": predicted_total,
                "expected": expected_total,
                "predicted_facts": [
                    {
                        "predicate": str(getattr(fact, "predicate", "") or ""),
                        "object": str(getattr(fact, "object", "") or ""),
                    }
                    for fact in predicted
                ],
            }
        )

    precision = true_positives / predicted_count if predicted_count else 1.0
    recall = true_positives / expected_count if expected_count else 1.0
    metrics = {
        "model": model,
        "cases": len(corpus),
        "true_positives": true_positives,
        "predicted": predicted_count,
        "expected": expected_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "case_results": case_results,
    }
    print("FACT_EXTRACTION_QUALITY=" + json.dumps(metrics, sort_keys=True))

    min_precision = float(os.getenv("E2E_MIN_PRECISION", "0.20"))
    min_recall = float(os.getenv("E2E_MIN_RECALL", "0.30"))
    assert precision >= min_precision, metrics
    assert recall >= min_recall, metrics
