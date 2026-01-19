"""
Complete Episodic Memory Demo for UMA-3.

This script:
1. Creates EpisodicSQLStore + FAISS index
2. Builds EpisodeIndexer
3. Converts conversation history into an Episode
4. Stores in SQL + FAISS
5. Retrieves it using semantic search
6. Demonstrates archival subsystem

Coding agent instructions:
--------------------------
- Replace DummyLLM and DummyEmbedder with actual adapters.
- Use this script as an integration test.
"""

import asyncio
import os
from datetime import datetime

from ..stores.episodic_sql import EpisodicSQLStore
from ..adapters.vector.faiss_adapter import FaissIndex
from ..adapters.llm.base import LLMInterface, EmbeddingInterface
from ..features.episodic.indexer import EpisodeIndexer
from ..features.episodic.archive import EpisodicArchive
from ..features.episodic.feature import EpisodicFeature
from ..core.uma3_memory import UMA3Memory


class DummyLLM(LLMInterface):
    async def generate(self, messages, max_tokens=200, temperature=0.0, **kw):
        return "User discussed database timeout issues."


class DummyEmbedder(EmbeddingInterface):
    async def embed(self, texts):
        return [[0.2] * 16 for _ in texts]


async def main():
    base = os.path.dirname(__file__)

    # Store + index
    episodic_store = EpisodicSQLStore(os.path.join(base, "episodes.db"), FaissIndex(16))
    archive = EpisodicArchive(episodic_store)

    # Memory
    memory = UMA3Memory(config=None)  # config unused in this isolated demo

    # Register episode indexer + feature
    llm = DummyLLM()
    embedder = DummyEmbedder()
    indexer = EpisodeIndexer(llm, embedder)
    memory.register_episode_indexer(indexer)

    episodic_feature = EpisodicFeature(episodic_store, archive)
    episodic_feature.attach(memory)

    # Build episode
    messages = [
        {"role": "user", "content": "I keep getting a database timeout error."},
        {"role": "assistant", "content": "Let me investigate the issue."},
    ]

    ep, emb = await indexer.build_episode("user:demo", messages)
    await memory.add_episode(ep, emb)

    print("Episode stored:", ep)

    # Search
    query_emb = (await embedder.embed(["database timeout"]))[0]
    results = await memory.search_episodes(query_emb, user_id="user:demo", k=5)
    print("\nSearch results:")
    for r in results:
        print("-", r.id, r.summary)

    # Archive
    archived = await memory.archive_old_episodes("user:demo")
    print("\nArchived:", archived)


if __name__ == "__main__":
    asyncio.run(main())