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
    ctl = RLMController(llm=DummyLLM(), env=DummyEnv())
    pack = type("P", (), {})()
    pack.episodes = [DummyEpisode()]
    pack.facts = [DummyFact()]
    pack.skills = [DummySkill()]
    pack.graph = [{"labels": ["Entity"], "properties": {"name": "x"}}]

    snippets = ctl._build_snippet_summary(pack)
    assert "launch plan" in snippets["episodes"]
    assert "likes" in snippets["facts"]
    assert "status_update" in snippets["skills"]
    assert "Entity" in snippets["graph"]
