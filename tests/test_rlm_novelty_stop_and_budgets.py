import asyncio

import pytest

from uma.core.retrieval.rlm.controller import RLMController


class DummyEnv:
    def __init__(self, retrieval_cfg):
        self._agent_id = "agent:1"
        self._project_id = None
        self._memory = type("M", (), {"retrieval_cfg": retrieval_cfg, "working_memory": None})()
        self._semantic_core = self._SemanticCore()
        self._procedural_core = self._ProceduralCore()

    async def get_query_embedding(self, query_text):
        return [0.0, 0.0, 0.0]

    class _SemanticCore:
        async def search(self, subject, query_embedding, owner_type, owner_id, k=10, offset=0, filters=None, query_text=None, allowed_topics=None):
            # Return a stable set => novelty becomes 0 after baseline.
            return [{"id": "f1", "owner_type": owner_type, "owner_id": owner_id, "subject": subject, "predicate": "p", "object": "o"}]

    async def fetch_more_facts(self, user_id, predicate, k=10, offset=0, **kwargs):
        # Same stable set regardless of paging => no novelty.
        return [{"id": "f1", "owner_type": "agent", "owner_id": "agent:1", "subject": user_id, "predicate": "p", "object": "o"}]

    class _ProceduralCore:
        async def search(self, user_id, query_embedding, owner_type, owner_id, k=10, **kwargs):
            return []

    async def episodic_cluster_summaries(self, user_id, k=3, max_episodes=10, **kwargs):
        return []

    async def graph_neighbors(self, **kwargs):
        return []

class BudgetEnv(DummyEnv):
    def __init__(self, retrieval_cfg):
        super().__init__(retrieval_cfg)
        self._returned_once = False

    def __init__(self, retrieval_cfg):
        super().__init__(retrieval_cfg)
        self._semantic_core = self._BudgetSemanticCore(self)

    class _BudgetSemanticCore:
        def __init__(self, parent):
            self._parent = parent

        async def search(self, subject, query_embedding, owner_type, owner_id, k=10, offset=0, filters=None, query_text=None, allowed_topics=None):
            # Return multiple new facts only once (during the loop), not during baseline.
            if self._parent._returned_once:
                self._parent._returned_once = False
                return []
            return [
                {"id": "f1", "owner_type": owner_type, "owner_id": owner_id, "subject": subject, "predicate": "p", "object": "o"},
                {"id": "f2", "owner_type": owner_type, "owner_id": owner_id, "subject": subject, "predicate": "p", "object": "o2"},
            ]

    async def fetch_more_facts(self, user_id, predicate, k=10, offset=0, **kwargs):
        return []


@pytest.mark.asyncio
async def test_rlm_stops_on_diminishing_returns():
    rlm_cfg = type(
        "R",
        (),
        {
            "enabled": True,
            "test_mode": True,
            "max_steps": 5,
            "max_actions_per_step": 1,
            "max_env_calls": 50,
            "max_items_per_type": 30,
            "timeout_s": 5.0,
            "novelty_window": 2,
            "min_recent_novelty": 1,
            "max_new_facts_per_step": 100,
            "max_new_chunks_per_step": 100,
            "max_graph_expansions_per_step": 100,
            "salience_threshold": 0.6,
            "min_semantic_facts": 4,
            "min_high_salience_facts": 2,
            "min_cluster_summaries": 1,
            "cluster_k": 3,
            "graph_predicate_limit": 1,
            "predicate_weights": None,
            "semantic_first": True,
            "clusters_first": True,
            "max_state_chars": 1200,
        },
    )()
    retrieval_cfg = type("RC", (), {"rlm": rlm_cfg, "max_evidence_chunks": 0})()
    env = DummyEnv(retrieval_cfg)
    c = RLMController(llm=None, env=env)

    pack = await c.retrieve_context("u1", "q")
    assert any(w == "stop:diminishing_returns" for w in pack.warnings)


@pytest.mark.asyncio
async def test_rlm_stops_when_max_new_facts_per_step_hit():
    rlm_cfg = type(
        "R",
        (),
        {
            "enabled": True,
            "test_mode": True,
            "max_steps": 5,
            "max_actions_per_step": 1,
            "max_env_calls": 50,
            "max_items_per_type": 30,
            "timeout_s": 5.0,
            "novelty_window": 3,
            "min_recent_novelty": 1,
            "max_new_facts_per_step": 1,
            "max_new_chunks_per_step": 100,
            "max_graph_expansions_per_step": 100,
            "salience_threshold": 0.6,
            "min_semantic_facts": 4,
            "min_high_salience_facts": 2,
            "min_cluster_summaries": 0,
            "cluster_k": 3,
            "graph_predicate_limit": 1,
            "predicate_weights": None,
            "semantic_first": True,
            "clusters_first": True,
            "max_state_chars": 1200,
        },
    )()
    retrieval_cfg = type("RC", (), {"rlm": rlm_cfg, "max_evidence_chunks": 0})()
    env = BudgetEnv(retrieval_cfg)
    # Baseline retrieval runs before the loop; force baseline semantic to return empty.
    env._returned_once = True
    c = RLMController(llm=None, env=env)

    pack = await c.retrieve_context("u1", "q")
    assert any(w == "stop:max_new_facts_per_step" for w in pack.warnings)
