import ast
from pathlib import Path

import pytest


def _assert_no_subject_keyword_in_search_calls(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "search":
            continue
        if any(isinstance(kw, ast.keyword) and kw.arg == "subject" for kw in node.keywords or []):
            raise AssertionError(f"{path} contains a .search(subject=...) call; retrieval must be ownership-only")


def test_rlm_deterministic_decision_does_not_inject_subject_filter():
    from uma.core.retrieval.rlm.decisions import deterministic_decision

    class _Pack:
        facts = []
        chunks = []
        steps = []
        owner_type = "user"
        user_id = "user:123"

    class _Coverage:
        needs_semantic = True
        needs_clusters = False

    decision = deterministic_decision(_Pack(), _Coverage(), cfg={})
    assert decision is not None
    assert decision.actions
    assert decision.actions[0].action == "search_semantic"
    assert decision.actions[0].filters is None


@pytest.mark.parametrize(
    "relpath",
    [
        "uma/core/retrieval/rlm/controller.py",
        "uma/core/retrieval/rlm/environment.py",
        "uma/core/retrieval/service.py",
    ],
)
def test_no_subject_keyword_in_search_calls(relpath: str):
    _assert_no_subject_keyword_in_search_calls(Path(relpath))

