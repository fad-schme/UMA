from __future__ import annotations

from uma.core.utils.user_query_helper import build_query_term_set


def test_build_query_term_set_is_deterministic() -> None:
    q = 'How do I "reset MFA" for AWS IAM users, and why does it fail? Explain the details.'
    a = build_query_term_set(q, max_terms=10, max_phrases=4)
    b = build_query_term_set(q, max_terms=10, max_phrases=4)
    assert a == b


def test_build_query_term_set_filters_noise() -> None:
    q = "What is 123 456? how to explain a guide to the the the."
    ts = build_query_term_set(q, max_terms=10, max_phrases=4)
    assert "123" not in ts.terms
    assert "456" not in ts.terms
    assert "how" not in ts.terms
    assert "explain" not in ts.terms
    assert "guide" not in ts.terms

