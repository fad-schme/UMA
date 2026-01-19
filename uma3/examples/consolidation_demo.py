"""
Consolidation Demo for UMA-3.

This example:
1. Creates episodic and semantic stores.
2. Attaches ConsolidationFeature.
3. Runs a consolidation cycle for a user.

Coding agent instructions:
--------------------------
- Replace DummyLLM + DummyEmbedder with real backends.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

from ..stores.episodic_sql import EpisodicSQLStore
from ..stores.semantic_sql import SemanticSQLStore
from ..adapters.vector.faiss_adapter import FaissIndex
from ..features.consolidation.feature import ConsolidationFeature
from ..core.uma3_memory import UMA3Memory
from ..types_episode import Episode


class DummyLLM:
    async def generate(self, messages, max_tokens=200, temperature=0.0, **kw):
        return "The user frequently mentions issues with the database."


class DummyEmbedder:
    async def embed(self, texts):
        return [[0.3] * 16 for _ in texts]


async def main():
    base = os.path.dirname(__file__)
    episodic = EpisodicSQLStore(os.path.join(base, "episodes.db"), FaissIndex(16))
    semantic = SemanticSQLStore(os.path.join(base, "semantic.db"), FaissIndex(16))

    memory = UMA3Memory()
    llm = DummyLLM()
    embedder = DummyEmbedder()

    # Attach feature
    feature = ConsolidationFeature(
        episodic_store=episodic,
        semantic_store=semantic,
        llm=llm,
        embedder=embedder,
    )
    feature.attach(memory)

    # Insert dummy episodes
    for i in range(5):
        ep = Episode(
            id=f"ep{i}",
            user_id="user:demo",
            timestamp=datetime.utcnow(),
            summary=f"Database timeout error occurred. ({i})",
            raw=None,
            tags=["error", "database"],
            embedding=[0.1] * 16,
        )
        await episodic.add_episode(ep, ep.embedding)

    print("Running consolidation...")
    result = await memory.run_consolidation("user:demo")
    print("New distilled facts:", result)


if __name__ == "__main__":
    asyncio.run(main())