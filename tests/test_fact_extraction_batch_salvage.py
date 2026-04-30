from __future__ import annotations

import asyncio
import json

from uma.ingest.types import DocumentChunk
from uma.memory.semantic.extractor import FactExtractor


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, **_kwargs):
        self.calls += 1
        # Batch prompt returns JSON with only one chunk key (forces salvage for the other).
        user = messages[-1]["content"]
        payload = json.loads(user)
        chunk_ids = [c["chunk_id"] for c in payload["chunks"]]
        first = chunk_ids[0]
        return json.dumps(
            {
                "chunks": {
                    first: {
                        "facts": [
                            {
                                "subject": "X",
                                "predicate": "STATES",
                                "object": "This is a sufficiently long object sentence for extraction.",
                                "confidence": 0.9,
                            }
                        ]
                    }
                }
            }
        )


class _FakeLLMMixed:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, **_kwargs):
        self.calls += 1
        user = messages[-1]["content"]
        # Batch call (JSON object with "chunks" list)
        try:
            payload = json.loads(user)
        except Exception:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
            chunk_ids = [c["chunk_id"] for c in payload["chunks"]]
            first = chunk_ids[0]
            return json.dumps(
                {
                    "chunks": {
                        first: {
                            "facts": [
                                {
                                    "subject": "X",
                                    "predicate": "STATES",
                                    "object": "This is a sufficiently long object sentence for extraction.",
                                    "confidence": 0.9,
                                }
                            ]
                        }
                    }
                }
            )
        # Per-chunk fallback call
        return json.dumps(
            {
                "facts": [
                    {
                        "subject": "Y",
                        "predicate": "STATES",
                        "object": "This is a sufficiently long object sentence for extraction.",
                        "confidence": 0.8,
                    }
                ]
            }
        )

class _FakeLLMPerChunk(_FakeLLM):
    async def generate(self, messages, **_kwargs):
        self.calls += 1
        # Per-chunk prompt: always return a fact.
        return json.dumps(
            {
                "facts": [
                    {
                        "subject": "Y",
                        "predicate": "STATES",
                        "object": "This is a sufficiently long object sentence for extraction.",
                        "confidence": 0.8,
                    }
                ]
            }
        )


def test_extract_facts_batch_salvages_missing_chunks() -> None:
    chunks = [
        DocumentChunk(
            chunk_id="chunk_a",
            doc_id="doc1",
            text="Architecture " * 30 + ".",
            page_range=(1, 1),
            position=1,
            paragraph_index_start=0,
            paragraph_index_end=0,
        ),
        DocumentChunk(
            chunk_id="chunk_b",
            doc_id="doc1",
            text="Design " * 60 + ".",
            page_range=(1, 1),
            position=2,
            paragraph_index_start=1,
            paragraph_index_end=1,
        ),
    ]

    llm = _FakeLLMMixed()

    async def run():
        extractor = FactExtractor(llm=llm)
        return await extractor.extract_chunk_facts_batch(
            chunks,
            owner_type="user",
            owner_id="user:u1",
            source_path="p.pdf",
            source_hash="h",
            doc_id="doc1",
            min_fact_words=5,
            batch_size_chunks=2,
            max_chars=12000,
        )

    facts = asyncio.run(run())
    # Expect at least one fact, and it must be attributed to one of the chunks.
    assert facts
    sources = set()
    for f in facts:
        if getattr(f, "source_ids", None):
            sources.add(str(f.source_ids[0]))
    assert {"chunk_a", "chunk_b"}.issubset(sources)
