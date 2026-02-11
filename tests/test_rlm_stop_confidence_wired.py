from __future__ import annotations

from uma.core.retrieval.policy import should_stop


def test_should_stop_uses_confidence_key() -> None:
    stop, reason = should_stop(
        recall_score=0.0,
        coverage={"confidence": 0.9, "facts": 0, "episodes": 0},
        calls_made=0,
        max_calls=6,
        tokens_used=0,
        token_budget=5000,
        user_results_count=0,
    )
    assert stop is True
    assert reason == "coverage_confident"

