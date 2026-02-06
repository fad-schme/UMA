import pytest

from uma.core.retrieval.rlm.controller import RLMController


class LaneEnv:
    def __init__(self, retrieval_cfg):
        self._agent_id = "agent-default"
        self._memory = type("M", (), {"retrieval_cfg": retrieval_cfg, "working_memory": None})()
        self.last = {}
        self._semantic_core = self._SemanticCore(self)
        self._chunk_core = self._ChunkCore()
        self._procedural_core = self._ProceduralCore()

    async def get_query_embedding(self, query_text):
        return [0.0, 0.0, 0.0]

    class _SemanticCore:
        def __init__(self, parent):
            self._parent = parent

        async def search(
            self,
            subject,
            query_embedding,
            owner_type,
            owner_id,
            k=10,
            offset=0,
            filters=None,
            query_text=None,
            allowed_topics=None,
        ):
            self._parent.last = {
                "filters": filters,
                "owner_type": owner_type,
                "owner_id": owner_id,
            }
            return []

    class _ChunkCore:
        async def search_chunks(self, **kwargs):
            return []

    class _ProceduralCore:
        async def search(self, user_id, query_embedding, owner_type, owner_id, k=10, **kwargs):
            return []


    async def episodic_cluster_summaries(self, *args, **kwargs):
        return []

    async def graph_neighbors(self, *args, **kwargs):
        return []


@pytest.mark.asyncio
async def test_rlm_lane_subject_filters_only_for_recall():
    rlm_cfg = type(
        "R",
        (),
        {"enabled": True, "test_mode": True, "max_steps": 1, "max_actions_per_step": 1, "max_env_calls": 1},
    )()
    retrieval_cfg = type("RC", (), {"rlm": rlm_cfg, "max_evidence_chunks": 0})()
    env = LaneEnv(retrieval_cfg)
    c = RLMController(llm=None, env=env)

    # Non-recall => agent lane, no subject filter
    await c.retrieve_context("u1", "How should a secure cloud architecture be structured?")
    assert env.last["owner_type"] == "agent"
    assert env.last["filters"] is None

    # Recall => user lane, subject filter applied
    await c.retrieve_context("u1", "Remember what we discussed last time about zero trust?")
    assert env.last["owner_type"] == "user"
    assert env.last["filters"] == {"subject": "user:u1"}
