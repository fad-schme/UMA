"""
uma.core.semantic.extractor
============================

FactExtractor — LLM-based semantic fact extraction.

Contracts
---------
- Input: subject (e.g. "user:123" or just user_id), text (assistant reply)
- Output: List[Fact] with meta["salience"] populated

Reliability
-----------
- Never raises; returns [] on failure.
- Logs invalid JSON and schema problems.

Production robustness
---------------------
LLMs may wrap JSON with markdown fences or additional text.
This extractor attempts:
  1) strict json.loads(raw)
  2) salvage: locate first {...} JSON object and parse again
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...adapters.llm.base import LLMInterface
from ...types_fact import Fact
from .scorer import SalienceScorer

logger = logging.getLogger(__name__)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parsing helper.

    Attempts strict parsing first, then tries to salvage the first JSON object
    found in the string (handles markdown fences / leading prose).
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    # 1) Strict parse
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) Salvage: locate the first JSON object substring
    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        return None

    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class FactExtractor:
    """Extracts structured facts from text using an LLM."""

    def __init__(self, llm: LLMInterface, scorer: SalienceScorer) -> None:
        self.llm = llm
        self.scorer = scorer
        logger.info("FactExtractor initialized.")

    async def extract_from_text(self, subject: str, text: str) -> List[Fact]:
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
            '  \"facts\": [\n'
            "    {\n"
            '      \"predicate\": \"likes\",\n'
            '      \"object\": \"sushi\",\n'
            '      \"confidence\": 0.0-1.0,\n'
            '      \"source_ids\": []\n'
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

        data = _try_parse_json(raw)
        if data is None:
            logger.error("FactExtractor: invalid JSON output (unsalvageable). RAW=%r", raw)
            return []

        facts_payload = data.get("facts")
        if not isinstance(facts_payload, list):
            logger.warning("FactExtractor: JSON missing list 'facts'.")
            return []

        now = datetime.now(timezone.utc)
        out: List[Fact] = []

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
            out.append(fact)

        logger.info("FactExtractor: extracted %d facts for subject=%s", len(out), subject)
        return out