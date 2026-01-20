import json

import pytest

from uma3.core.retrieval.rlm.decisions import ControllerDecision


def test_controller_decision_validates_actions():
    payload = {
        "actions": [
            {"action": "search_semantic", "k": 3},
            {"action": "fetch_facts", "ids": ["f1", "f2"]},
            {"action": "graph_neighbors", "node_id": "n1", "predicate_scope": ["LIKES"]},
        ],
        "done": False,
    }

    decision = ControllerDecision.from_json(json.dumps(payload))
    assert len(decision.actions) == 3
    assert decision.actions[0].action == "search_semantic"
    assert decision.actions[1].ids == ["f1", "f2"]


def test_controller_decision_rejects_invalid_action():
    payload = {"actions": [{"action": "fetch_facts"}], "done": False}
    with pytest.raises(ValueError):
        ControllerDecision.from_json(json.dumps(payload))
