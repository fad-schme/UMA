from datetime import datetime, timedelta

from uma.core.retrieval.rlm.environment import UMAMemoryEnvironment


class DummyFact:
    def __init__(self, fid, subject, predicate, obj, owner_type="user", owner_id="user:1"):
        self.id = fid
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.confidence = 0.8
        self.meta = {"salience": 0.6}
        self.owner_type = owner_type
        self.owner_id = owner_id


class DummyEpisode:
    def __init__(self, eid, owner_id, ts):
        self.id = eid
        self.user_id = owner_id
        self.owner_type = "user"
        self.owner_id = owner_id
        self.timestamp = ts
        self.summary = f"summary-{eid}"


class DummySemanticCore:
    def __init__(self):
        self.calls = []

    async def search(self, subject, query_embedding, owner_type=None, owner_id=None, k=10, offset=0):
        self.calls.append((owner_type, owner_id))
        label = owner_type or "user"
        return [DummyFact(f"f_{label}", subject or "user:1", "likes", label, owner_type=owner_type or "user", owner_id=owner_id or "user:1")]

    async def search_tiered(
        self,
        subject,
        query_embedding,
        k=10,
        offset=0,
        agent_id=None,
        project_id=None,
        owner_scope=None,
        filters=None,
        query_text=None,
        allowed_topics=None,
    ):
        results = []
        for owner_type, owner_id in (
            ("user", subject),
            ("agent", "agent-1"),
            ("project", f"{subject}:proj-1"),
        ):
            results.extend(await self.search(subject, query_embedding, owner_type, owner_id, k=k, offset=offset))
        return results

    async def fetch_by_ids(self, ids):
        return [DummyFact(fid, "user:1", "likes", "tea") for fid in ids]

    async def search_text(self, subject, query_text, limit=5, owner_type=None, owner_id=None):
        self.calls.append((owner_type, owner_id))
        label = owner_type or "user"
        return [DummyFact(f"f_text_{label}", subject or "user:1", "document", label, owner_type=owner_type or "user", owner_id=owner_id or "user:1")]


class DummyEpisodicStore:
    def __init__(self):
        now = datetime.utcnow()
        self._eps = [
            DummyEpisode("e1", "user:u1", now - timedelta(days=2)),
            DummyEpisode("e2", "user:u1", now),
        ]

    async def list_recent(self, owner_type, owner_id, n=5):
        return self._eps[:n]

    async def search(self, query_embedding, owner_type=None, owner_id=None, k=10, offset=None):
        return self._eps

    async def fetch_summaries(self, ids):
        return [{"id": i, "summary": f"summary-{i}"} for i in ids]

    async def fetch_transcripts(self, ids):
        return [{"id": i, "summary": f"summary-{i}", "raw": "raw"} for i in ids]

    async def list_cluster_summaries(self, owner_type, owner_id, k=5, time_range=None, max_episodes=None):
        return [
            {
                "id": "cluster:1",
                "owner_type": owner_type,
                "owner_id": owner_id,
                "summary": "cluster summary",
                "episode_ids": ["e1", "e2"],
                "latest_timestamp": datetime.utcnow().isoformat(),
                "count": 2,
                "salience": 0.75,
            }
        ]

class DummyEpisodicCore:
    def __init__(self, store):
        self._store = store

        self.last_time_range = None

    async def search(self, owner_type, owner_id, query_embedding, k=10, offset=0):
        return await self._store.search(query_embedding, owner_type=owner_type, owner_id=owner_id, k=k, offset=offset)

    async def search_tiered(self, user_id, query_embedding, k=10, offset=0, agent_id=None, project_id=None, owner_scope=None):
        return await self.search("user", user_id, query_embedding, k=k, offset=offset)

    async def list_cluster_summaries(self, owner_type, owner_id, k=5, max_episodes=None, time_range=None):
        self.last_time_range = time_range
        return await self._store.list_cluster_summaries(
            owner_type=owner_type,
            owner_id=owner_id,
            k=k,
            max_episodes=max_episodes,
            time_range=time_range,
        )

    async def list_cluster_summaries_tiered(self, user_id, k=5, max_episodes=None, time_range=None, agent_id=None, project_id=None, owner_scope=None):
        return await self.list_cluster_summaries("user", user_id, k=k, max_episodes=max_episodes, time_range=time_range)


class DummyGraphCore:
    def neighbors_tiered(
        self,
        user_id,
        node_id,
        predicate_scope=None,
        depth=1,
        k=10,
        agent_id=None,
        project_id=None,
        owner_scope=None,
    ):
        return [{"labels": ["Entity"], "properties": {"name": "x"}}]

    async def neighbors(
        self,
        user_id,
        node_id,
        owner_type,
        owner_id,
        predicate_scope=None,
        depth=1,
        k=10,
    ):
        # Validate that strict ownership arguments are being passed
        assert owner_type in {"user", "agent", "project"}
        assert isinstance(owner_id, str) and owner_id
        return [{"labels": ["Entity"], "properties": {"name": "x"}}]


class DummyWM:
    def get_context(self, user_id, last_n=None):
        return ["wm"]


class DummyMemory:
    def __init__(self):
        self.retrieval_service = object()
        self.working_memory = DummyWM()
        self.semantic_core = DummySemanticCore()
        self.episodic_core = DummyEpisodicCore(DummyEpisodicStore())
        self.graph_core = DummyGraphCore()
        self.embedder = DummyEmbedder()
        self.agent_id = "agent-1"
        self.project_id = "proj-1"


class DummyEmbedder:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_environment_semantic_and_episodic_search():
    memory = DummyMemory()
    env = UMAMemoryEnvironment(memory)

    facts = asyncio_run(env.search_semantic("u1", [0.1, 0.2], k=1, filters={"subject": "user:1"}))
    assert facts and facts[0].predicate == "likes"
    owner_types = {getattr(f, "owner_type", None) for f in facts}
    assert owner_types == {"user", "agent", "project"}
    assert set(memory.semantic_core.calls) == {
        ("user", "user:u1"),
        ("agent", "agent-1"),
        ("project", "user:u1:proj-1"),
    }

    start = datetime.utcnow() - timedelta(days=1)
    eps = asyncio_run(env.search_episodic("u1", [0.1, 0.2], k=10, time_range={"start": start}))
    assert len(eps) == 1
    assert eps[0].id == "e2"


def test_environment_fetch_and_neighbors():
    env = UMAMemoryEnvironment(DummyMemory())

    facts = asyncio_run(env.fetch_facts_by_ids("u1", ["f1", "f2"]))
    assert len(facts) == 2

    neighbors = asyncio_run(env.graph_neighbors("u1", "node1", predicate_scope=["likes"], depth=2, k=5))
    assert neighbors and neighbors[0]["labels"] == ["Entity"]


def test_environment_episode_clusters():
    env = UMAMemoryEnvironment(DummyMemory())
    clusters = asyncio_run(env.episodic_cluster_summaries("u1", k=2, max_episodes=10))
    assert clusters
    assert "episode_ids" in clusters[0]


def test_environment_fetch_episode_clusters_filters():
    env = UMAMemoryEnvironment(DummyMemory())
    assert not asyncio_run(env.fetch_episode_clusters("u1", min_salience=0.8))
    clusters = asyncio_run(env.fetch_episode_clusters("u1", min_salience=0.5))
    assert clusters
    assert clusters[0]["salience"] >= 0.5


def test_environment_expand_graph():
    env = UMAMemoryEnvironment(DummyMemory())
    nodes = asyncio_run(
        env.expand_graph("u1", "node1", predicate="likes", hops=2, direction="inbound", k=5)
    )
    assert nodes and nodes[0]["labels"] == ["Entity"]


def test_fetch_episode_clusters_sanitizes_time_range():
    env = UMAMemoryEnvironment(DummyMemory())
    env._memory.episodic_core.last_time_range = None
    clusters = asyncio_run(
        env.fetch_episode_clusters(
            "u1",
            min_salience=0.5,
            time_range={"start": 1000, "end": 500, "offset": -5},
        )
    )
    assert clusters
    assert env._memory.episodic_core.last_time_range == {"start": 1000, "offset": 0}


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
