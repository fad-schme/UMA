"""
Procedural Memory Demo for UMA-3 (SQLite + FAISS)

This example:
1. Creates a ProceduralSQLStore
2. Creates a ProceduralFeature
3. Defines a new Skill and embeds it
4. Runs a query to find applicable skills

Coding agent instructions
-------------------------
- Replace DummyLLM and DummyEmbedder with real implementations.
- Expand the example to execute the skill plan.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Dict, List

from ..types_skill import Skill
from ..stores.procedural_sql import ProceduralSQLStore
from ..adapters.vector.faiss_adapter import FaissIndex
from ..adapters.llm.base import EmbeddingInterface, LLMInterface
from ..core.uma3_memory import UMA3Memory
from ..features.procedural.feature import ProceduralFeature
from ..features.procedural.skill_indexer import SkillIndexer


class DummyLLM(LLMInterface):
    async def generate(self, messages, max_tokens=256, temperature=0.0, **kw):
        return "This is a dummy LLM."


class DummyEmbedder(EmbeddingInterface):
    async def embed(self, texts):
        return [[0.2] * 32 for _ in texts]  # 32-D vectors


async def main():
    db_path = os.path.join(os.path.dirname(__file__), "procedural_demo.db")
    index = FaissIndex(dim=32)
    store = ProceduralSQLStore(db_path=db_path, vector_index=index)

    memory = UMA3Memory()
    embedder = DummyEmbedder()
    llm = DummyLLM()

    feature = ProceduralFeature(store=store, embedder=embedder)
    feature.attach(memory)

    indexer = SkillIndexer(llm=llm, embedder=embedder)

    # Build a skill
    skill, emb = await indexer.build_skill_from_definition(
        name="Reset Database Procedure",
        trigger_phrases=["reset database", "wipe db"],
        trigger_patterns=[r"reset.*database", r"wipe.*db"],
        plan={"steps": ["Backup", "Shutdown", "Drop Tables", "Recreate"]},
        tools=["sql_admin"],
        example="User asked to reset the staging DB.",
        meta={"domain": "database"},
    )

    await memory.add_skill(skill, emb)

    print("Searching for skills matching query: 'Please reset the database now'")
    matched = await memory.find_skills("Please reset the database now")

    for s in matched:
        print("Matched Skill:", s.name)


if __name__ == "__main__":
    asyncio.run(main())