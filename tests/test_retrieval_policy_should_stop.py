import pytest

from uma.retrieve.policy import should_stop


def test_should_stop_on_max_calls():
    stop, reason = should_stop(
        recall_score=0.0,
        coverage={"confidence": 0.0},
        calls_made=10,
        max_calls=10,
        tokens_used=0,
        token_budget=10000,
        user_results_count=0,
    )
    assert stop is True
    assert reason == "max_calls_reached"


def test_should_stop_on_token_budget():
    stop, reason = should_stop(
        recall_score=0.0,
        coverage={"confidence": 1.0},
        calls_made=0,
        max_calls=10,
        tokens_used=6000,
        token_budget=5000,
        user_results_count=0,
    )
    assert stop is True
    assert reason == "token_budget_exhausted"


def test_recall_expected_but_no_user_results_continues():
    stop, reason = should_stop(
        recall_score=1.0,
        coverage={"confidence": 0.1},
        calls_made=1,
        max_calls=10,
        tokens_used=100,
        token_budget=5000,
        user_results_count=0,
    )
    assert stop is False
    assert reason == "recall_expected_but_no_user_results"


def test_recall_with_user_results_and_high_confidence_stops():
    stop, reason = should_stop(
        recall_score=1.0,
        coverage={"confidence": 0.95},
        calls_made=1,
        max_calls=10,
        tokens_used=100,
        token_budget=5000,
        user_results_count=2,
    )
    assert stop is True
    assert reason == "coverage_confident_and_recall_satisfied"


def test_non_recall_high_confidence_stops():
    stop, reason = should_stop(
        recall_score=0.0,
        coverage={"confidence": 0.9, "facts": 0, "episodes": 0},
        calls_made=1,
        max_calls=10,
        tokens_used=100,
        token_budget=5000,
        user_results_count=0,
    )
    assert stop is True
    assert reason == "coverage_confident"