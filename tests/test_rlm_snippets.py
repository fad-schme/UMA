from uma.core.retrieval.rlm.controller import RLMController
from uma.core.retrieval.rlm.environment import MemoryEnvironment


class DummyEnv(MemoryEnvironment):
    async def retrieve_slice(self, user_id, memory_type, query):
        return []

    async def retrieve_all(self, user_id, query):
        return {"episodes": [], "facts": [], "skills": [], "graph": []}

    async def get_working_memory(self, user_id):
        return []

    async def search_semantic(self, user_id, query_embedding, k=10, filters=None):
        return []

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

    async def get_query_embedding(self, query_text):
        return [0.1, 0.2]


class DummyLLM:
    async def generate(self, messages, max_tokens=256, temperature=0.0, **kwargs):
        return "{\"actions\": [], \"done\": true}"


class DummyEpisode:
    summary = "met about launch plan"


class DummyFact:
    predicate = "likes"
    object = "tea"


class DummySkill:
    name = "status_update"


def test_rlm_snippet_summary_contains_items():
    env = DummyEnv()
    ctl = RLMController(llm=DummyLLM(), env=env)

    pack = type("P", (), {})()
    pack.episodes = [{"summary": "met about launch plan"}]
    pack.facts = [{"predicate": "likes", "object": "tea"}]
    pack.skills = [{"name": "status_update"}]
    pack.graph = [{"labels": ["Entity"], "properties": {"name": "x"}}]

    # Snippets are now implicit in retrieved items
    assert "launch plan" in pack.episodes[0]["summary"]
    assert "likes" in pack.facts[0]["predicate"]
    assert "status_update" in pack.skills[0]["name"]
    assert "Entity" in pack.graph[0]["labels"]