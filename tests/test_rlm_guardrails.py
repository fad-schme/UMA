import asyncio

from uma.core.retrieval.rlm.controller import RLMController
class DummyEnv:
    def __init__(self):
        class _RLM:
            max_env_calls = 1
            max_items_per_type = 5

        class _Retrieval:
            rlm = _RLM()

        class _Memory:
            retrieval_cfg = _Retrieval()
            working_memory = None

        self._memory = _Memory()
        self._agent_id = "agent-1"
        self._semantic_core = self._SemanticCore()
        self._chunk_core = self._ChunkCore()
        self._episodic_core = self._EpisodicCore()
        self._procedural_core = self._ProceduralCore()

    async def get_query_embedding(self, query_text):
        return [0.1, 0.2, 0.3]

    # --- Semantic ---
    class _SemanticCore:
        async def search(
            self,
            query_embedding,
            owner_type,
            owner_id,
            k=10,
            offset=0,
            filters=None,
            query_text=None,
        ):
            # Return a dict with a large string field to exercise truncation/guardrails
            return [
                {
                    "id": "f1",
                    "predicate": "likes",
                    "summary": "x" * 200,
                }
            ]

    async def fetch_more_facts(
        self,
        user_id,
        predicate,
        k,
        offset=0,
        owner_type="agent",
        owner_id=None,
    ):
        return [
            {
                "id": f"f{offset + i}",
                "predicate": predicate,
                "summary": "more facts",
            }
            for i in range(1, min(k, 2) + 1)
        ]

    # --- Episodic ---
    class _EpisodicCore:
        async def search(self, user_id, query_embedding, owner_type, owner_id, k=10, offset=0, **kwargs):
            return [{"id": "e1", "summary": "episode"}]

    async def episodic_cluster_summaries(self, user_id, k=5, max_episodes=50, time_range=None, owner_type="agent", owner_id=None):
        return [{"id": "cluster:1", "summary": "cluster summary", "episode_ids": ["e1", "e2"]}]

    async def fetch_episode_clusters(
        self,
        user_id,
        k=5,
        max_episodes=50,
        time_range=None,
        min_salience=None,
        owner_type="agent",
        owner_id=None,
    ):
        return await self.episodic_cluster_summaries(user_id, k=k, max_episodes=max_episodes, time_range=time_range)

    # --- Procedural ---
    class _ProceduralCore:
        async def search(self, user_id, query_embedding, owner_type, owner_id, k=10, **kwargs):
            return [{"id": "s1", "name": "skill"}]

    # --- Graph ---
    async def graph_neighbors(self, user_id, node_id, predicate_scope=None, depth=1, k=10, owner_type="agent", owner_id=None):
        return [{"labels": ["Entity"], "properties": {"name": "x"}}]

    async def expand_graph(self, user_id, subject, predicate=None, hops=1, direction=None, k=10):
        return await self.graph_neighbors(user_id, subject, predicate_scope=[predicate] if predicate else None, depth=hops, k=k)

    class _ChunkCore:
        async def search_chunks(self, **kwargs):
            return []


class DummyLLM:
    async def generate(self, messages, max_tokens=256, temperature=0.0, **kwargs):
        # Two actions to trigger env call limit
        return """
        {"actions": [
            {"action": "search_semantic", "k": 1},
            {"action": "search_semantic", "k": 1}
        ], "done": false}
        """.strip()


def test_max_env_calls_and_truncation():
    ctl = RLMController(llm=DummyLLM(), env=DummyEnv())

    pack = asyncio.run(ctl.retrieve_context("u1", "query"))
    assert any("max_env_calls" in warning for warning in pack.warnings)
    # Ensure truncation applied to fact snippets (dict strings)
    if pack.facts:
        assert len(pack.facts[0].get("summary", "")) <= 2000
