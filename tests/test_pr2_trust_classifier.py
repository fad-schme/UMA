"""
PR2 — source trust classifier: one assertion per policy table row.
"""
from __future__ import annotations

import pytest

from uma.common.trust import SourceDescriptor, score_source

_SESSION = "sess-abc-123"


class TestScoreSourcePolicyTable:
    def test_turn_user_authenticated(self):
        s = score_source(SourceDescriptor(kind="turn_user", session_id=_SESSION))
        assert s == pytest.approx(0.9)

    def test_turn_user_no_session(self):
        s = score_source(SourceDescriptor(kind="turn_user", session_id=None))
        assert s == pytest.approx(0.5)

    def test_turn_user_empty_session(self):
        s = score_source(SourceDescriptor(kind="turn_user", session_id="   "))
        assert s == pytest.approx(0.5)

    def test_turn_assistant_authenticated(self):
        s = score_source(SourceDescriptor(kind="turn_assistant", session_id=_SESSION))
        assert s == pytest.approx(0.7)

    def test_turn_assistant_no_session(self):
        s = score_source(SourceDescriptor(kind="turn_assistant", session_id=None))
        assert s == pytest.approx(0.5)

    def test_document(self):
        s = score_source(SourceDescriptor(kind="document"))
        assert s == pytest.approx(0.7)

    def test_bootstrap_memory_default(self):
        s = score_source(SourceDescriptor(kind="bootstrap_memory"))
        assert s == pytest.approx(0.6)

    def test_bootstrap_diary_default(self):
        s = score_source(SourceDescriptor(kind="bootstrap_diary"))
        assert s == pytest.approx(0.6)

    def test_bootstrap_memory_manual(self):
        s = score_source(SourceDescriptor(kind="bootstrap_memory", import_mode="manual"))
        assert s == pytest.approx(0.8)

    def test_bootstrap_diary_manual(self):
        s = score_source(SourceDescriptor(kind="bootstrap_diary", import_mode="manual"))
        assert s == pytest.approx(0.8)

    def test_tool_output(self):
        s = score_source(SourceDescriptor(kind="tool_output"))
        assert s == pytest.approx(0.5)

    def test_promotion_inherits_parent(self):
        s = score_source(SourceDescriptor(kind="promotion", parent_trust_score=0.85))
        assert s == pytest.approx(0.85)

    def test_promotion_no_parent_defaults_to_half(self):
        s = score_source(SourceDescriptor(kind="promotion", parent_trust_score=None))
        assert s == pytest.approx(0.5)

    def test_unknown_kind_defaults_to_half(self):
        s = score_source(SourceDescriptor(kind="unknown_future_kind"))
        assert s == pytest.approx(0.5)

    def test_result_in_unit_interval(self):
        kinds = [
            "turn_user", "turn_assistant", "document",
            "bootstrap_memory", "bootstrap_diary", "tool_output",
            "promotion", "something_else",
        ]
        for kind in kinds:
            v = score_source(SourceDescriptor(kind=kind))
            assert 0.0 <= v <= 1.0, f"{kind} produced out-of-range {v}"
