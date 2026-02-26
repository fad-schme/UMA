from __future__ import annotations

import asyncio

from uma.adapters.llm.callable_adapter import CallableEmbedderAdapter, CallableLLMAdapter
from uma.core.episodic.indexer import EpisodeIndexer

from tests.helpers.providers import fake_embed, fake_llm


def test_episode_indexer_builds_episode_with_valid_embedding_shape():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    embedder = CallableEmbedderAdapter(
        callable_fn=fake_embed,
        dimension=16,
        name="tests.fake_embed",
        default_kwargs={"dimension": 16},
    )
    indexer = EpisodeIndexer(llm=llm, embedder=embedder)

    wm_entries = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    ep, embedding = asyncio.run(
        indexer.build_episode(owner_type="user", owner_id="user:u1", wm_entries=wm_entries)
    )
    assert isinstance(ep.summary, str) and ep.summary.strip()
    assert isinstance(embedding, list) and len(embedding) == 16
