import pytest

from uma.core.retrieval.rlm.controller import RLMController


class NoQueryTextEnv:
    def __init__(self, retrieval_cfg):
        self._agent_id = "agent:1"
        self._project_id = None
        self._memory = type("M", (), {"retrieval_cfg": retrieval_cfg, "working_memory": None})()
        self._chunk_core = self._ChunkCore()
        self._procedural_core = self._ProceduralCore()

    async def get_query_embedding(self, query_text):
        return [0.0, 0.0, 0.0]

    async def search_semantic(self, user_id, query_embedding, k=10, **kwargs):
        return []

    async def episodic_cluster_summaries(self, user_id, k=3, max_episodes=10, **kwargs):
        return []

    async def graph_neighbors(self, **kwargs):
        return []

    class _ProceduralCore:
        async def search(self, user_id, query_embedding, owner_type, owner_id, k=10, **kwargs):
            return []

    class _ChunkCore:
        async def search_chunks(self, **kwargs):
            owner_type = kwargs.get("owner_type", "agent")
            owner_id = kwargs.get("owner_id")
            return [{"id": "c1", "owner_type": owner_type, "owner_id": owner_id, "text": "x"}]


@pytest.mark.asyncio
async def test_rlm_controller_search_chunks_works_without_query_text_param():
    rlm_cfg = type(
        "R",
        (),
        {"enabled": True, "test_mode": True, "max_steps": 1, "max_actions_per_step": 1, "max_env_calls": 1},
    )()
    retrieval_cfg = type("RC", (), {"rlm": rlm_cfg, "max_evidence_chunks": 0})()
    env = NoQueryTextEnv(retrieval_cfg)
    c = RLMController(llm=None, env=env)
    pack = await c.retrieve_context("u1", "q")
    assert getattr(pack, "chunks", [])
