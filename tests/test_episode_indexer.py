import asyncio

from uma3.core.episodic.indexer import EpisodeIndexer


class DummyLLM:
    async def generate(self, messages, max_tokens=256, temperature=0.0, **kwargs):
        return "episode summary"


class DummyEmbedder:
    def __init__(self):
        self.last_texts = None

    async def embed(self, texts):
        # Expect list of strings
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("embed expects list[str]")
        self.last_texts = texts
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_episode_indexer_embed_input_shape():
    llm = DummyLLM()
    embedder = DummyEmbedder()
    indexer = EpisodeIndexer(llm=llm, embedder=embedder)

    wm_entries = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    ep, embedding = asyncio.run(indexer.build_episode("u1", wm_entries))
    assert ep.summary
    assert embedding == [0.1, 0.2, 0.3]
    assert embedder.last_texts == [ep.summary]
