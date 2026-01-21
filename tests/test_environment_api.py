from datetime import datetime, timedelta

from uma.core.retrieval.rlm.environment import UMAMemoryEnvironment


class DummyFact:
    def __init__(self, fid, subject, predicate, obj):
        self.id = fid
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.confidence = 0.8
        self.meta = {"salience": 0.6}


class DummyEpisode:
    def __init__(self, eid, user_id, ts):
        self.id = eid
        self.user_id = user_id
        self.timestamp = ts
        self.summary = f"summary-{eid}"


class DummySemanticStore:
    async def search(self, query_embedding, subject=None, k=10):
        return [DummyFact("f1", subject or "user:1", "likes", "tea")]

    async def fetch_facts_by_ids(self, ids):
        return [DummyFact(fid, "user:1", "likes", "tea") for fid in ids]


class DummyEpisodicStore:
    def __init__(self):
        now = datetime.utcnow()
        self._eps = [
            DummyEpisode("e1", "u1", now - timedelta(days=2)),
            DummyEpisode("e2", "u1", now),
        ]

    async def list_recent(self, user_id, n=5):
        return self._eps[:n]

    async def search(self, query_embedding, user_id=None, k=10):
        return self._eps

    async def fetch_summaries(self, ids):
        return [{"id": i, "summary": f"summary-{i}"} for i in ids]

    async def fetch_transcripts(self, ids):
        return [{"id": i, "summary": f"summary-{i}", "raw": "raw"} for i in ids]

    async def list_cluster_summaries(self, user_id, k=5, time_range=None, max_episodes=None):
        return [
            {
                "id": "cluster:1",
                "user_id": user_id,
                "summary": "cluster summary",
                "episode_ids": ["e1", "e2"],
                "latest_timestamp": datetime.utcnow().isoformat(),
                "count": 2,
            }
        ]


class DummyGraphCore:
    def get_neighbors(self, entity_id, depth=1):
        return [{"labels": ["Entity"], "properties": {"name": "x"}}]


class DummyWM:
    def get_context(self, user_id, last_n=None):
        return ["wm"]


class DummyMemory:
    def __init__(self):
        self.retrieval_service = object()
        self.working_memory = DummyWM()
        self.semantic_store = DummySemanticStore()
        self.episodic_store = DummyEpisodicStore()
        self.graph_core = DummyGraphCore()
        self.embedder = DummyEmbedder()


class DummyEmbedder:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_environment_semantic_and_episodic_search():
    env = UMAMemoryEnvironment(DummyMemory())

    facts = asyncio_run(env.search_semantic("u1", [0.1, 0.2], k=1, filters={"subject": "user:1"}))
    assert facts and facts[0]["predicate"] == "likes"

    start = datetime.utcnow() - timedelta(days=1)
    eps = asyncio_run(env.search_episodic("u1", [0.1, 0.2], k=10, time_range={"start": start}))
    assert len(eps) == 1
    assert eps[0]["id"] == "e2"


def test_environment_fetch_and_neighbors():
    env = UMAMemoryEnvironment(DummyMemory())

    facts = asyncio_run(env.fetch_facts_by_ids("u1", ["f1", "f2"]))
    assert len(facts) == 2

    summaries = asyncio_run(env.fetch_episode_summaries(["e1"]))
    assert summaries[0]["id"] == "e1"

    transcripts = asyncio_run(env.fetch_episode_transcripts(["e1"]))
    assert transcripts[0]["raw"] == "raw"

    neighbors = asyncio_run(env.graph_neighbors("u1", "node1", predicate_scope=["likes"], depth=2, k=5))
    assert neighbors and neighbors[0]["labels"] == ["Entity"]


def test_environment_working_memory_window():
    env = UMAMemoryEnvironment(DummyMemory())
    wm = asyncio_run(env.get_working_memory("u1", window=1))
    assert wm == ["wm"]


def test_environment_episode_clusters():
    env = UMAMemoryEnvironment(DummyMemory())
    clusters = asyncio_run(env.episodic_cluster_summaries("u1", k=2, max_episodes=10))
    assert clusters
    assert "episode_ids" in clusters[0]


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
