from __future__ import annotations

"""
uma.core.semantic.extractor
==========================

Canonical API for semantic fact extraction.

This module intentionally exposes two related but distinct extraction surfaces:

1) `FactExtractor` (turn/text extraction)
   - Input: subject, free-form text (e.g., assistant reply)
   - Output: `List[uma.types.Fact]` with `meta["salience"]` populated

2) Document-chunk extraction (ingestion)
   - Input: `List[DocumentChunk]`
   - Output: `List[ExtractedFact]` (ingestion-friendly facts with `source_chunk_id`)

All implementation details for chunk extraction (selection heuristics, batching,
parsing, and enforcement) live in `uma.core.semantic.extractor_utils`. This keeps
this module stable as the single public import surface for ingestion callers.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from ...adapters.llm.base import LLMInterface
from ...types import Fact
from ..ingest.types import DocumentChunk, ExtractedFact
from ..utils.json_utils import try_parse_json_object
from . import extractor_utils as utils
from .scorer import SalienceScorer

logger = logging.getLogger(__name__)


class FactExtractor:
    """Extracts structured facts from text using an LLM."""

    def __init__(self, llm: LLMInterface, scorer: SalienceScorer) -> None:
        self.llm = llm
        self.scorer = scorer
        logger.debug("FactExtractor initialized.")

    async def extract_from_text(self, subject: str, text: str, *, extra_meta: dict | None = None) -> List[Fact]:
        if not isinstance(subject, str) or not subject.strip():
            logger.warning("FactExtractor: invalid subject=%r", subject)
            return []
        if not isinstance(text, str) or not text.strip():
            return []

        system_prompt = (
            "Extract long-term, stable facts about the USER from the text.\n"
            "Do NOT include ephemeral or turn-specific details.\n\n"
            "Return ONLY valid JSON in this schema:\n"
            "{\n"
            '  "facts": [\n'
            "    {\n"
            '      "predicate": "likes",\n'
            '      "object": "sushi",\n'
            '      "confidence": 0.0-1.0,\n'
            '      "source_ids": []\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"SUBJECT: {subject}\nTEXT:\n{text}\n"},
        ]

        try:
            raw = await self.llm.generate(messages, max_tokens=400, temperature=0.0)
        except Exception:
            logger.exception("FactExtractor: LLM generate failed.")
            return []

        data = try_parse_json_object(raw)
        if data is None:
            logger.error("FactExtractor: invalid JSON output (unsalvageable). RAW=%r", raw)
            return []

        facts_payload = data.get("facts")
        if not isinstance(facts_payload, list):
            logger.warning("FactExtractor: JSON missing list 'facts'.")
            return []

        now = datetime.now(timezone.utc)
        out: List[Fact] = []
        turn_id = None
        if isinstance(extra_meta, dict) and extra_meta.get("turn_id"):
            turn_id = str(extra_meta["turn_id"])

        for f in facts_payload:
            if not isinstance(f, dict):
                continue

            predicate = f.get("predicate")
            obj = f.get("object")
            conf = f.get("confidence", 0.7)
            source_ids = f.get("source_ids", [])

            if not predicate or obj is None:
                continue

            try:
                confidence = float(conf)
                confidence = max(0.0, min(1.0, confidence))
            except Exception:
                confidence = 0.7

            try:
                sid_list = [str(s) for s in (source_ids if isinstance(source_ids, list) else [])]
            except Exception:
                sid_list = []

            fact = Fact(
                id=str(uuid.uuid4()),
                subject=subject,
                predicate=str(predicate),
                object=obj,
                created_at=now,
                updated_at=now,
                source_ids=sid_list,
                confidence=confidence,
                meta={},
            )
            fact.meta["salience"] = self.scorer.score(fact)
            if turn_id:
                fact.meta["turn_id"] = turn_id
            out.append(fact)

        logger.info("FactExtractor: extracted %d facts for subject=%s", len(out), subject)
        return out


def select_chunks_for_fact_extraction(
    chunks: List[DocumentChunk],
    *,
    max_chunks: Optional[int] = None,
    max_per_page: Optional[int] = None,
) -> List[DocumentChunk]:
    return utils.select_chunks_for_fact_extraction(chunks, max_chunks=max_chunks, max_per_page=max_per_page)


async def extract_facts_batch(
    chunks: List[DocumentChunk],
    *,
    llm: Any,
    min_fact_words: int = utils._DEFAULT_MIN_FACT_WORDS,
    batch_size_chunks: int = 4,
    max_chars: int = 12000,
    max_facts_per_chunk: int = utils._DEFAULT_MAX_FACTS_PER_CHUNK,
    object_max_words: int = utils._DEFAULT_OBJECT_MAX_WORDS,
    max_fact_tokens: int = utils._DEFAULT_MAX_FACT_TOKENS,
) -> List[ExtractedFact]:
    return await utils.extract_facts_batch(
        chunks,
        llm=llm,
        min_fact_words=min_fact_words,
        batch_size_chunks=batch_size_chunks,
        max_chars=max_chars,
        max_facts_per_chunk=max_facts_per_chunk,
        object_max_words=object_max_words,
        max_fact_tokens=max_fact_tokens,
    )


async def extract_facts_one(
    chunk: DocumentChunk,
    *,
    llm: Any,
    min_fact_words: int = utils._DEFAULT_MIN_FACT_WORDS,
    scorer: Any | None = None,
    max_facts: int = utils._DEFAULT_MAX_FACTS_PER_CHUNK,
    object_max_words: int = utils._DEFAULT_OBJECT_MAX_WORDS,
    max_fact_tokens: int = utils._DEFAULT_MAX_FACT_TOKENS,
) -> List[ExtractedFact]:
    return await utils.extract_facts_one(
        chunk,
        llm=llm,
        min_fact_words=min_fact_words,
        scorer=scorer,
        max_facts=max_facts,
        object_max_words=object_max_words,
        max_fact_tokens=max_fact_tokens,
    )


async def extract_facts(
    chunks: List[DocumentChunk],
    *,
    llm: Any,
    min_fact_words: int = utils._DEFAULT_MIN_FACT_WORDS,
    max_facts_per_chunk: int = utils._DEFAULT_MAX_FACTS_PER_CHUNK,
    object_max_words: int = utils._DEFAULT_OBJECT_MAX_WORDS,
    max_fact_tokens: int = utils._DEFAULT_MAX_FACT_TOKENS,
) -> List[ExtractedFact]:
    return await utils.extract_facts(
        chunks,
        llm=llm,
        min_fact_words=min_fact_words,
        max_facts_per_chunk=max_facts_per_chunk,
        object_max_words=object_max_words,
        max_fact_tokens=max_fact_tokens,
    )

