from uma.core.retrieval.policy import RetrievalPolicy
from uma.core.retrieval.service import RetrievalService
from uma.core.retrieval.selector import MemorySelector


class DummyEmbedder:
    dimension = 3
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class DummyMemory:
    def __init__(self):
        self.embedder = DummyEmbedder()
        self.chunk_core = DummyChunkCore()
        self.chunk_store = DummyChunkStore()
        self.retrieval_cfg = DummyRetrievalConfig()


class DummyChunkStore:
    async def search_text(self, query_text, *, owner_type=None, owner_id=None, k=10):
        return [
            {"id": "c_text", "text": "remember last time", "position": 1, "owner_type": owner_type, "owner_id": owner_id}
        ]


class DummyChunkCore:
    class _Store:
        async def _fetch_ranked_rows_by_ids(self, ids, log_context="", owner_type=None, owner_id=None):
            # Return objects in the same order as ids.
            out = []
            for cid in ids:
                out.append({"id": cid, "text": f"chunk {cid}", "position": 99, "owner_type": owner_type, "owner_id": owner_id})
            return out

    def __init__(self):
        self.store = self._Store()

    async def search_chunks(self, **kwargs):
        owner_type = kwargs.get("owner_type")
        owner_id = kwargs.get("owner_id")
        return [
            {"id": "c_text", "text": "remember last time", "position": 1, "owner_type": owner_type, "owner_id": owner_id}
        ]

    async def _fetch_ranked_by_ids(self, ids, log_context="", owner_type=None, owner_id=None):
        return await self.store._fetch_ranked_rows_by_ids(
            ids,
            log_context=log_context,
            owner_type=owner_type,
            owner_id=owner_id,
        )


def _dummy_raw():
    return {
        "episodes": [],
        "facts": [
            {"id": "a1", "meta": {"salience": 0.7}, "confidence": 0.7, "owner_type": "agent", "owner_id": "a1", "source_ids": ["c_ev"]},
            {"id": "u1", "meta": {"salience": 0.6}, "confidence": 0.6, "owner_type": "user", "owner_id": "user:u1"},
        ],
        "chunks": [
            {"id": "c1", "text": "agent chunk", "position": 2, "owner_type": "agent", "owner_id": "a1"},
            {"id": "c2", "text": "user chunk", "position": 3, "owner_type": "user", "owner_id": "user:u1"},
        ],
        "skills": [],
        "graph": [],
    }


class DummySelector:
    def __init__(self):
        self.captured_policy = None
        self.max_episodes = 3
        self.max_facts = 5
        self.max_skills = 2
        self.max_graph_items = 2

    def select(self, raw, *, policy=None):
        self.captured_policy = policy
        # defer to real selector behavior in tests that need ordering
        return raw


class DummyRetrievalConfig:
    max_episodes = 3
    max_facts = 5
    max_skills = 2
    max_graph_items = 2
    lexical_chunks_k = 10
    max_evidence_chunks = 3
    strict = True


def test_retrieval_service_passes_policy_for_text_query():
    svc = RetrievalService(DummyMemory(), DummyRetrievalConfig())
    async def _raw(**kwargs):
        return _dummy_raw()
    svc._retrieve_raw = _raw
    selector = DummySelector()
    svc.selector = selector

    result = asyncio_run(
        svc.retrieve(
            user_id="u1",
            memory_type="all",
            query_text_or_embedding="remember last time",
            agent_id="a1",
            project_id="p1",
        )
    )
    assert isinstance(selector.captured_policy, RetrievalPolicy)
    assert result["facts"][0]["id"] == "a1"
    assert any(c["id"] == "c_ev" for c in result["chunks"])


def test_retrieval_service_recall_prefers_user():
    svc = RetrievalService(DummyMemory(), DummyRetrievalConfig())
    async def _raw(**kwargs):
        return _dummy_raw()
    svc._retrieve_raw = _raw
    svc.selector = MemorySelector(
        max_episodes=3,
        max_facts=5,
        max_skills=2,
        max_graph_items=2,
    )

    result = asyncio_run(
        svc.retrieve(
            user_id="u1",
            memory_type="all",
            query_text_or_embedding="remember last time",
            agent_id="a1",
            project_id="p1",
        )
    )
    assert result["facts"][0]["id"] == "a1"


def test_retrieval_service_prefers_agent_without_recall():
    svc = RetrievalService(DummyMemory(), DummyRetrievalConfig())
    async def _raw(**kwargs):
        return _dummy_raw()
    svc._retrieve_raw = _raw
    svc.selector = MemorySelector(
        max_episodes=3,
        max_facts=5,
        max_skills=2,
        max_graph_items=2,
    )

    result = asyncio_run(
        svc.retrieve(
            user_id="u1",
            memory_type="all",
            query_text_or_embedding="how to configure X",
            agent_id="a1",
            project_id="p1",
        )
    )
    assert result["facts"][0]["id"] == "a1"


def test_retrieval_service_no_policy_for_embedding_query():
    svc = RetrievalService(DummyMemory(), DummyRetrievalConfig())
    async def _raw(**kwargs):
        return _dummy_raw()
    svc._retrieve_raw = _raw
    selector = DummySelector()
    svc.selector = selector

    asyncio_run(
        svc.retrieve(
            user_id="u1",
            memory_type="all",
            query_text_or_embedding=[0.1, 0.2, 0.3],
            agent_id="a1",
            project_id="p1",
        )
    )
    assert selector.captured_policy is None


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
