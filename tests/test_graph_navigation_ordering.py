import asyncio

from uma.core.retrieval.service import RetrievalService


class DummyEmbedder:
    dimension = 3

    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class DummyGraphCore:
    def __init__(self):
        self.called = 0

    def neighbors(self, **kwargs):
        self.called += 1
        return [{"id": "g1"}]


class DummyMemory:
    def __init__(self):
        self.embedder = DummyEmbedder()
        self.graph_core = DummyGraphCore()
        hybrid = type(
            "Hybrid",
            (),
            {"enabled": False, "top_k_dense": 0, "top_k_sparse": 0, "fusion_strategy": "rrf"},
        )()
        self.retrieval_cfg = type(
            "Cfg",
            (),
            {"strict": True, "hybrid": hybrid, "max_evidence_chunks": 0},
        )()


class DummyRetrievalConfig:
    max_episodes = 1
    max_facts = 1
    max_skills = 1
    max_graph_items = 1


def test_graph_not_fetched_when_no_signal():
    mem = DummyMemory()
    svc = RetrievalService(mem, DummyRetrievalConfig())

    async def _raw(**kwargs):
        return {"episodes": [], "facts": [], "chunks": [], "skills": [], "graph": []}

    svc._retrieve_raw = _raw

    asyncio.run(
        svc.retrieve(
            user_id="u1",
            memory_type="all",
            query_text_or_embedding="q",
            agent_id="a1",
        )
    )

    assert mem.graph_core.called == 0
