from __future__ import annotations

import asyncio

from uma.memory.episodic.indexer import EpisodeIndexer

from tests.helpers.providers import fake_embed, fake_llm


class FakeLLM:
    async def generate(self, messages, max_tokens=256, temperature=0.0, **kwargs):
        return await fake_llm(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )


class FakeEmbedder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed(self, texts):
        return await fake_embed(texts=list(texts), dimension=self.dimension)


def test_episode_indexer_builds_episode_with_valid_embedding_shape():
    llm = FakeLLM()
    embedder = FakeEmbedder(dimension=16)
    indexer = EpisodeIndexer(llm=llm, embedder=embedder)

    wm_entries = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    ep, embedding = asyncio.run(
        indexer.build_episode(owner_type="user", owner_id="user:u1", turn_entries=wm_entries)
    )
    assert isinstance(ep.summary, str) and ep.summary.strip()
    assert isinstance(embedding, list) and len(embedding) == 16
