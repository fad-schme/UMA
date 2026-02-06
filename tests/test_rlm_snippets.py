from uma.core.retrieval.rlm.controller import RLMController


class DummyEnv:
    def __init__(self):
        self._semantic_core = self._SemanticCore()
        self._chunk_core = self._ChunkCore()
        self._episodic_core = self._EpisodicCore()
        self._procedural_core = self._ProceduralCore()

    class _SemanticCore:
        async def search(self, subject, query_embedding, owner_type, owner_id, k=10, offset=0, filters=None, query_text=None, allowed_topics=None):
            return []

    class _ChunkCore:
        async def search_chunks(self, **kwargs):
            return []

    class _EpisodicCore:
        async def search(self, user_id, query_embedding, owner_type, owner_id, k=10, offset=0, **kwargs):
            return []

    class _ProceduralCore:
        async def search(self, user_id, query_embedding, owner_type, owner_id, k=10, **kwargs):
            return []

    async def episodic_cluster_summaries(self, user_id, k=5, max_episodes=50, time_range=None):
        return []

    async def fetch_episode_clusters(self, user_id, k=5, max_episodes=50, time_range=None, min_salience=None):
        return []

    async def graph_neighbors(self, user_id, node_id, predicate_scope=None, depth=1, k=10):
        return []

    async def expand_graph(self, user_id, subject, predicate=None, hops=1, direction=None, k=10):
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
