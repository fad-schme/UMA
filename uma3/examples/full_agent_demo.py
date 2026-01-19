"""
FULL UMA-3 AGENT DEMO

This script demonstrates:
- How to configure UMA3Memory
- How to initialize stores, indexers, and features
- How to wire the Orchestrator
- How to run a simple conversation loop

Coding agent instructions:
--------------------------
- Replace DummyLLM and DummyEmbedder with real implementations.
- Replace FAISS with the right vector backend if needed.
- Use Neo4j for TKG if available.
"""

import asyncio
import os

from ..core.memory_config import UMA3Config, WorkingMemoryConfig, EpisodicMemoryConfig, SemanticMemoryConfig, ProceduralMemoryConfig, TemporalGraphConfig, ConsolidationConfig, HybridRetrievalConfig
from ..core.uma3_memory import UMA3Memory
from ..core.orchestrator import UMA3Orchestrator

from ..adapters.vector.faiss_adapter import FaissIndex
from ..adapters.graph.neo4j_adapter import Neo4jBackend

from ..stores.semantic_sql import SemanticSQLStore
from ..stores.episodic_sql import EpisodicSQLStore
from ..stores.procedural_sql import ProceduralSQLStore

from ..core.working_memory.core import WorkingMemoryFeature
from ..features.salience.feature import SalienceFeature
from ..features.temporal_graph.feature import TemporalGraphFeature
from ..features.procedural.feature import ProceduralFeature
from ..features.hybrid_retrieval.feature import HybridRetrievalFeature
from ..features.consolidation.feature import ConsolidationFeature

from ..features.hybrid_retrieval.feature import HybridRetrievalFeature
from ..features.hybrid_retrieval.reranker import Reranker


class DummyLLM:
    async def generate(self, messages, max_tokens=200, temperature=0.0, **kw):
        return "This is a dummy LLM response. Replace with real implementation."


class DummyEmbedder:
    async def embed(self, texts):
        return [[0.1] * 16 for _ in texts]


async def main():
    base = os.path.dirname(__file__)

    # 1. Create config
    config = UMA3Config(
        working_memory=WorkingMemoryConfig(),
        episodic=EpisodicMemoryConfig(db_path=os.path.join(base, "episodes.db")),
        semantic=SemanticMemoryConfig(db_path=os.path.join(base, "semantic.db")),
        procedural=ProceduralMemoryConfig(db_path=os.path.join(base, "skills.db")),
        temporal_graph=TemporalGraphConfig(uri="bolt://localhost:7687", user="neo4j", password="password"),
        consolidation=ConsolidationConfig(),
        retrieval=HybridRetrievalConfig(),
    )

    # 2. Instantiate UMA3Memory
    memory = UMA3Memory(config)

    # 3. Register LLM + Embedder
    llm = DummyLLM()
    embedder = DummyEmbedder()

    memory.register_llm(llm)
    memory.register_embedder(embedder)

    # 4. Register Stores
    memory.register_episodic_store(EpisodicSQLStore(config.episodic.db_path, FaissIndex(16)))
    memory.register_semantic_store(SemanticSQLStore(config.semantic.db_path, FaissIndex(16)))
    memory.register_procedural_store(ProceduralSQLStore(config.procedural.db_path, FaissIndex(16)))
    memory.register_graph(Neo4jBackend(config.temporal_graph.uri, config.temporal_graph.user, config.temporal_graph.password))

    # 5. Register EpisodeIndexer
    from ..features.episodic.indexer import EpisodeIndexer
    memory.register_episode_indexer(EpisodeIndexer(llm, embedder))

    # 6. Enable all features
    memory.enable_feature(WorkingMemoryFeature(llm, config.working_memory.max_tokens))
    memory.enable_feature(SalienceFeature(llm, embedder, memory.semantic_store))
    memory.enable_feature(TemporalGraphFeature(memory.graph))
    memory.enable_feature(ProceduralFeature(memory.procedural_store, embedder))
    memory.enable_feature(
        HybridRetrievalFeature(
            embedder,
            retriever=HybridRetriever(),
            selector=MemorySelector(),
            reranker=Reranker(),
        )
    )
    if config.consolidation.enabled:
        memory.enable_feature(
            ConsolidationFeature(
                episodic_store=memory.episodic_store,
                semantic_store=memory.semantic_store,
                llm=llm,
                embedder=embedder,
            )
        )

    orchestrator = UMA3Orchestrator(memory)

    # 7. Run conversation
    user_id = "user:demo"
    print("Agent: Hello! How can I help you today?")

    while True:
        user_msg = input("User: ")
        if user_msg.lower() in ("exit", "quit"):
            break
        reply = await orchestrator.handle_turn(user_id, user_msg)
        print("Agent:", reply)


if __name__ == "__main__":
    asyncio.run(main())