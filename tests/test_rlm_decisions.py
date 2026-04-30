import json

import pytest

from uma.retrieve.rlm.decisions import ControllerDecision


def test_controller_decision_validates_actions():
    payload = {
    "actions": [
        {"action": "search_semantic", "k": 3},
        {"action": "fetch_facts", "ids": ["f1", "f2"]},
        {
            "action": "graph_neighbors",
            "node_id": "n1",
            "predicate_scope": ["LIKES"],
            "k": 5,
            "depth": 1,
        },
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


def test_controller_decision_accepts_extended_actions():
    payload = {
        "actions": [
            {"action": "fetch_episode_clusters", "k": 2, "min_salience": 0.4},
            {
                "action": "expand_graph",
                "k": 5,
                "subject": "user:1",
                "direction": "both",
                "hops": 2,
            },
        ],
        "done": False,
    }

    decision = ControllerDecision.from_json(json.dumps(payload))
    assert decision.actions[0].action == "fetch_episode_clusters"
    assert decision.actions[1].subject == "user:1"


def test_controller_decision_rejects_unknown_action():
    payload = {"actions": [{"action": "resolve_conflicts", "fact_ids": ["f1"]}], "done": False}
    with pytest.raises(ValueError):
        ControllerDecision.from_json(json.dumps(payload))
