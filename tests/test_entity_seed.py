from __future__ import annotations

from uma.retrieve.rlm.entity_seed import extract_candidate_entities


def test_extract_candidate_entities_includes_acronyms_and_is_bounded() -> None:
    q = "How do IAM and VPC integrate with KMS for TLS?"
    out = extract_candidate_entities(q, facts=[], chunks=[], limit=3)
    assert out == ["IAM", "VPC", "KMS"]


def test_extract_candidate_entities_dedupes_case_insensitive() -> None:
    q = "IAM iam VPC vpc"
    out = extract_candidate_entities(q, facts=[], chunks=[], limit=10)
    assert out[:2] == ["IAM", "VPC"]
