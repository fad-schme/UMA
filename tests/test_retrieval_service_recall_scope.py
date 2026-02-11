from uma.core.retrieval.policy import RetrievalPolicy
from uma.core.retrieval.service import RetrievalService
from uma.core.retrieval.selector import MemorySelector
from uma.types_chunk import Chunk
from datetime import datetime, timezone


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
            Chunk(
                id="c_text",
                doc_id="d1",
                text="remember last time",
                owner_type=owner_type,
                owner_id=owner_id,
                page_range=(1, 1),
                position=1,
                source_path="",
                source_hash="",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                meta={},
            )
        ]


class DummyChunkCore:
    class _Store:
        async def _fetch_ranked_rows_by_ids(self, ids, log_context="", owner_type=None, owner_id=None):
            # Return objects in the same order as ids.
            out = []
            for cid in ids:
                out.append(
                    Chunk(
                        id=cid,
                        doc_id="d1",
                        text=f"chunk {cid}",
                        owner_type=owner_type,
                        owner_id=owner_id,
                        page_range=(1, 1),
                        position=99,
                        source_path="",
                        source_hash="",
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        meta={},
                    )
                )
            return out

    def __init__(self):
        self.store = self._Store()

    async def search_chunks(self, **kwargs):
        owner_type = kwargs.get("owner_type")
        owner_id = kwargs.get("owner_id")
        return [
            Chunk(
                id="c_text",
                doc_id="d1",
                text="remember last time",
                owner_type=owner_type,
                owner_id=owner_id,
                page_range=(1, 1),
                position=1,
                source_path="",
                source_hash="",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                meta={},
            )
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
            Chunk(
                id="c1",
                doc_id="d1",
                text="agent chunk",
                owner_type="agent",
                owner_id="a1",
                page_range=(1, 1),
                position=2,
                source_path="",
                source_hash="",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                meta={},
            ),
            Chunk(
                id="c2",
                doc_id="d1",
                text="user chunk",
                owner_type="user",
                owner_id="user:u1",
                page_range=(1, 1),
                position=3,
                source_path="",
                source_hash="",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                meta={},
            ),
        ],
        "skills": [],
        "graph": [],
    }


class DummySelector:
    def __init__(self):
        self.captured_policy = None
        self.max_episodes = 3
        self.max_facts = 5
        self.max_chunks = 5
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
        )
    )
    assert isinstance(selector.captured_policy, RetrievalPolicy)
    assert result["facts"][0]["id"] == "a1"
    assert any(getattr(c, "id", None) == "c_ev" for c in result["chunks"])


def test_retrieval_service_recall_prefers_user():
    svc = RetrievalService(DummyMemory(), DummyRetrievalConfig())
    async def _raw(**kwargs):
        return _dummy_raw()
    svc._retrieve_raw = _raw
    svc.selector = MemorySelector(
        max_episodes=3,
        max_facts=5,
        max_chunks=5,
        max_skills=2,
        max_graph_items=2,
    )

    result = asyncio_run(
        svc.retrieve(
            user_id="u1",
            memory_type="all",
            query_text_or_embedding="remember last time",
            agent_id="a1",
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
        max_chunks=5,
        max_skills=2,
        max_graph_items=2,
    )

    result = asyncio_run(
        svc.retrieve(
            user_id="u1",
            memory_type="all",
            query_text_or_embedding="how to configure X",
            agent_id="a1",
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
        )
    )
    assert selector.captured_policy is None


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
