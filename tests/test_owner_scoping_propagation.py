import asyncio

from uma.core.retrieval.service import RetrievalService


class DummyEmbedder:
    dimension = 3

    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class AssertingCore:
    def __init__(self, expected_owner_type: str, expected_owner_id: str):
        self.expected_owner_type = expected_owner_type
        self.expected_owner_id = expected_owner_id

    async def search(self, **kwargs):
        assert kwargs.get("owner_type") == self.expected_owner_type
        assert kwargs.get("owner_id") == self.expected_owner_id
        return []

    async def search_text(self, *args, **kwargs):
        assert kwargs.get("owner_type") == self.expected_owner_type
        assert kwargs.get("owner_id") == self.expected_owner_id
        return []

    async def search_chunks(self, **kwargs):
        assert kwargs.get("owner_type") == self.expected_owner_type
        assert kwargs.get("owner_id") == self.expected_owner_id
        return []


class DummyMemory:
    def __init__(self):
        self.embedder = DummyEmbedder()
        # force user scope via retrieval policy by using a recall-ish query text
        self.retrieval_cfg = type("Cfg", (), {"strict": True, "lexical_chunks_k": 1, "max_evidence_chunks": 0})()
        self.episodic_core = AssertingCore("user", "user:u1")
        self.semantic_core = AssertingCore("user", "user:u1")
        self.chunk_core = AssertingCore("user", "user:u1")
        self.procedural_core = AssertingCore("user", "user:u1")
        self.graph_core = None


class DummyRetrievalConfig:
    max_episodes = 1
    max_facts = 1
    max_skills = 1
    max_graph_items = 1


def test_classic_retrieval_passes_owner_scope():
    mem = DummyMemory()
    svc = RetrievalService(mem, DummyRetrievalConfig())

    # Use a high-recall query (contains "remember") so RetrievalPolicy sets user scope.
    asyncio.run(
        svc.retrieve(
            user_id="u1",
            memory_type="all",
            query_text_or_embedding="remember last time",
            agent_id="a1",
            project_id="p1",
        )
    )
