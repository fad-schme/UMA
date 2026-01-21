"""
Improved LLM-based summarizer for UMA consolidation.

Summarizes clusters of related episodes into stable, compressed,
and semantically meaningful summaries used for fact extraction.

Production guarantees:
----------------------
• Safe-fail behavior (never breaks consolidation)
• Input normalization and deduplication
• Anti-hallucination guardrails
• Token-efficient cluster merging
• Strict separation of summarization vs extraction logic
"""

from __future__ import annotations

import logging
import re
from typing import List

from ...adapters.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class ConsolidationSummarizer:
    """
    Summarizer used by Consolidator to produce:
    - macro-episode summaries
    - distilled semantic knowledge

    This version is production-grade and designed for:
    • stability
    • explicit anti-hallucination behavior
    • low-fluff, high-signal output
    """

    def __init__(self, llm: LLMInterface, max_tokens: int = 256):
        self.llm = llm
        self.max_tokens = max_tokens
        logger.info("ConsolidationSummarizer initialized (max_tokens=%d).", max_tokens)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    async def summarize_cluster(self, texts: List[str]) -> str:
        """
        Summarize multiple related episodes into a distilled semantic memory.

        Parameters
        ----------
        texts : List[str]
            Summaries or raw content of clustered episodes.

        Returns
        -------
        str
            A clean, compact summary suitable for high-salience fact extraction.
        """
        if not texts:
            logger.warning("ConsolidationSummarizer: empty cluster input.")
            return ""

        # --------------------------------------------------------------
        # 1. Clean and normalize inputs
        # --------------------------------------------------------------
        cleaned = [self._normalize_text(t) for t in texts if t and t.strip()]
        cleaned = self._dedupe(cleaned)

        if not cleaned:
            return ""

        # Limit overlong clusters (safety + token efficiency)
        combined = "\n---\n".join(cleaned[:25])
        if len(combined) > 4000:  # a safe cutoff before LLM call
            combined = combined[:4000] + "\n[...]"

        # --------------------------------------------------------------
        # 2. Build summarization prompt
        # --------------------------------------------------------------
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that performs STRICT, FACTUAL knowledge "
                    "consolidation for a memory system.\n"
                    "You will NOT hallucinate, infer missing details, or fabricate content.\n"
                    "Your job is to:\n"
                    "- detect recurring patterns\n"
                    "- extract important user information\n"
                    "- compress events to essentials\n"
                    "- highlight temporal trends or repeated issues\n"
                    "- note contradictions or anomalies\n\n"
                    "Output should be:\n"
                    "- concise (2–5 sentences)\n"
                    "- factual\n"
                    "- free of speculation\n"
                    "- suitable for structured fact extraction\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following related user episodes. DO NOT hallucinate.\n\n"
                    f"{combined}\n\n"
                    "Summary:"
                ),
            },
        ]

        # --------------------------------------------------------------
        # 3. Call LLM (safe fail)
        # --------------------------------------------------------------
        try:
            summary = await self.llm.generate(
                messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            return summary.strip()
        except Exception:
            logger.exception("ConsolidationSummarizer: LLM summarization failed.")
            return ""

    # ------------------------------------------------------------------
    # INTERNAL TEXT UTILITIES
    # ------------------------------------------------------------------
    def _normalize_text(self, t: str) -> str:
        """Normalize whitespace, remove noise, and sanitize text."""
        t = t.replace("\r", " ").replace("\n", " ").strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def _dedupe(self, items: List[str]) -> List[str]:
        """Deduplicate while preserving order."""
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out