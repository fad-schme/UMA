"""
UMA-3 Semantic Memory Demo (SQLite + FAISS)

This example demonstrates how to use the SemanticSQLStore + SalienceFeature +
UMA3Memory to extract salient facts from user input, generate embeddings, and
persist them in SQLite + FAISS. It then performs a semantic search.

Coding agent instructions:
--------------------------
- Use this script as a reference to test the semantic subsystem end-to-end.
- Replace DummyLLM and DummyEmbedder with real implementations.
- Ensure FAISS is installed (faiss-cpu) before running this.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Dict, List

from ..types_fact import Fact
from ..stores.semantic_sql import SemanticSQLStore
from ..adapters.vector.faiss_adapter import FaissIndex
from ..adapters.llm.base import LLMInterface, EmbeddingInterface
from ..core.uma3_memory import UMA3Memory
from ..features.salience.feature import SalienceFeature


class DummyLLM(LLMInterface):
    """
    Very simple LLM mock for demonstration only.

    It ALWAYS returns a single JSON fact indicating that the user prefers
    concise answers. Real LLM integration should implement retries,
    streaming, temperature control, etc.
    """

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        return """
        {
            "facts": [
                {
                    "predicate": "prefers_tone",
                    "object": "concise",
                    "confidence": 0.95,
                    "source_ids": []
                }
            ]
        }
        """


class DummyEmbedder(EmbeddingInterface):
    """
    Simple embedder mock producing deterministic vectors.

    Real embedder should:
    - Produce high-dimensional dense vectors (e.g., 1536 floats)
    - Handle batching
    - Support different embedding models
    """

    async def embed(self, texts):
        return [[0.1 for _ in range(16)] for _ in texts]  # fixed 16-D vectors


async def main():
    # Prepare paths
    current_dir = os.path.dirname(__file__)
    db_path = os.path.join(current_dir, "semantic_demo.db")

    # Instantiate FAISS
    vector_index = FaissIndex(dim=16)

    # Instantiate SQLite-backed store
    semantic_store = SemanticSQLStore(
        db_path=db_path,
        vector_index=vector_index,
    )

    # Create UMA3Memory and attach SalienceFeature
    memory = UMA3Memory(semantic_store=semantic_store)
    llm = DummyLLM()
    embedder = DummyEmbedder()

    salience_feature = SalienceFeature(
        llm=llm,
        embedder=embedder,
        semantic_store=semantic_store,
        salience_threshold=0.5,  # discard weak facts
    )
    salience_feature.attach(memory)

    user_id = "user:example"
    text = "I absolutely love concise answers. I prefer short explanations."

    print("\n--- Extracting + Ingesting Facts ---")
    persisted = await memory.ingest_salient_facts(user_id, text)

    for fact in persisted:
        print("Persisted:", fact)

    # Semantic search demonstration
    print("\n--- Searching for related facts ---")
    query_vector = (await embedder.embed(["concise answers"]))[0]
    results = await semantic_store.search(query_vector, subject=user_id, k=5)

    for r in results:
        print("Search result:", r)


if __name__ == "__main__":
    asyncio.run(main())