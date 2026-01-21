import asyncio

from uma.core.retrieval.rlm.controller import RLMController
from uma.core.retrieval.rlm.environment import MemoryEnvironment


class DummyEnv(MemoryEnvironment):
    async def retrieve_slice(self, user_id, memory_type, query):
        return []

    async def retrieve_all(self, user_id, query):
        return {"episodes": [], "facts": [], "skills": [], "graph": []}

    async def get_working_memory(self, user_id, window=None):
        return []

    async def search_semantic(self, user_id, query_embedding, k=10, filters=None):
        return [{"summary": "x" * 200}]

    async def fetch_facts_by_ids(self, user_id, ids):
        return []

    async def search_episodic(self, user_id, query_embedding, k=10, time_range=None):
        return []

    async def fetch_episode_summaries(self, ids):
        return []

    async def fetch_episode_transcripts(self, ids):
        return []

    async def graph_neighbors(self, user_id, node_id, predicate_scope=None, depth=1, k=10):
        return []

    async def episodic_cluster_summaries(self, user_id, k=5, max_episodes=50, time_range=None):
        return []

    async def get_query_embedding(self, query_text):
        return [0.1, 0.2, 0.3]


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
    ctl = RLMController(
        llm=DummyLLM(),
        env=DummyEnv(),
        max_steps=1,
        max_actions_per_step=2,
        max_env_calls=1,
        max_return_chars=50,
    )

    pack = asyncio.run(ctl.retrieve_context("u1", "query"))
    assert "max_env_calls" in pack.warnings
    # Ensure truncation applied to fact snippets (dict strings)
    if pack.facts:
        assert len(pack.facts[0].get("summary", "")) == 50
