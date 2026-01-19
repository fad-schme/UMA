"""
semantic/extractor.py
=====================

FactExtractor — production-grade LLM-based fact extraction engine.

It extracts long-term, salient, stable facts about a user from text
and returns structured Fact objects.

This logic was previously implemented in features/salience/extractor.py.
It is now a CORE subsystem.

Coding Agent Instructions
-------------------------
- Ensure LLM output strictly adheres to the JSON schema.
- Add retry/backoff logic for provider failures in future.
- Avoid hallucinations by forcing JSON-only responses.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from ...adapters.llm.base import LLMInterface
from ...types_fact import Fact
from .scorer import SalienceScorer

logger = logging.getLogger(__name__)


class FactExtractor:
    """
    Extracts structured, salient facts from text using an LLM.
    """

    def __init__(self, llm: LLMInterface, scorer: SalienceScorer):
        self.llm = llm
        self.scorer = scorer
        logger.info("FactExtractor initialized.")

    async def extract_from_text(self, subject: str, text: str) -> List[Fact]:
        """
        Extract stable facts about `subject` from `text`.

        Parameters
        ----------
        subject : str
            e.g., "user:123"
        text : str
            Conversation text or assistant reply.

        Returns
        -------
        List[Fact]
        """
        if not text.strip():
            logger.debug("FactExtractor: empty text.")
            return []

        system_prompt = (
            "Extract long-term factual information about the USER from the text. "
            "Facts must be stable, not ephemeral.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "facts": [\n'
            "    {\n"
            '      "predicate": "likes_food",\n'
            '      "object": "sushi",\n'
            '      "confidence": 0.9,\n'
            '      "source_ids": []\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        user_prompt = f"TEXT:\n{text}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # ---------------------------------------------------------
        # 1. LLM call
        # ---------------------------------------------------------
        try:
            raw = await self.llm.generate(messages, max_tokens=400, temperature=0.0)
        except Exception:
            logger.exception("FactExtractor: LLM generate() failed.")
            return []

        # ---------------------------------------------------------
        # 2. JSON parsing
        # ---------------------------------------------------------
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.exception("FactExtractor: invalid JSON output. RAW=%r", raw)
            return []

        if not isinstance(data, dict) or "facts" not in data:
            logger.warning("FactExtractor: missing 'facts' key in output.")
            return []

        now = datetime.now(timezone.utc)
        results: List[Fact] = []

        # ---------------------------------------------------------
        # 3. Fact object construction
        # ---------------------------------------------------------
        for idx, f in enumerate(data.get("facts", [])):
            if not isinstance(f, dict):
                continue

            predicate = f.get("predicate")
            obj = f.get("object")
            confidence = f.get("confidence", 0.7)
            source_ids = f.get("source_ids", [])

            if not predicate or obj is None:
                logger.debug("FactExtractor: skipping incomplete fact %r", f)
                continue

            fact = Fact(
                id=str(uuid.uuid4()),
                subject=subject,
                predicate=str(predicate),
                object=obj,
                created_at=now,
                updated_at=now,
                source_ids=[str(s) for s in source_ids],
                confidence=float(confidence),
                meta={},
            )

            salience = self.scorer.score(fact)
            fact.meta["salience"] = salience
            results.append(fact)

        logger.info("FactExtractor: extracted %d facts", len(results))
        return results