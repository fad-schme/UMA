"""
Summarization logic for working memory.

This module uses an LLM interface to compress older conversation segments into
short summary messages, which can replace raw messages in the working memory.

Design goals
------------
- Keep prompt construction explicit and easy to audit.
- Avoid hallucination by instructing the model to use only given content.
- Provide a pure function-like API:
    - Input: list of messages to summarize.
    - Output: a single text summary.

Integration
-----------
- QueueManager decides WHEN to summarize.
- WorkingMemoryBuffer stores the summary as a synthetic "summary" message.
"""

from __future__ import annotations

import logging
from typing import List, Iterable, Dict, Any

from ...adapters.llm.base import LLMInterface  # you must implement this
from ..llm.controller import LLMCallContext, generate_text

logger = logging.getLogger(__name__)


class WorkingMemorySummarizer:
    """
    LLM-based summarizer for working memory segments.

    The implementation assumes an LLMInterface with an async `generate`
    method that accepts a list of chat messages and returns a string.

    The summarizer itself:
    - Builds a concise prompt.
    - Concatenates the messages into a single text block.
    - Instructs the LLM to output a short, factual summary.
    """

    def __init__(self, llm: LLMInterface, max_summary_tokens: int = 256) -> None:
        """
        Parameters
        ----------
        llm:
            LLM client implementing LLMInterface.
        max_summary_tokens:
            Hint for the model to keep summaries short.
        """
        self._llm = llm
        self._max_summary_tokens = max_summary_tokens
        logger.info(
            "Initialized WorkingMemorySummarizer with max_summary_tokens=%d",
            max_summary_tokens,
        )

    async def summarize_messages(
        self,
        messages: Iterable[Dict[str, str]],
        extra_instructions: str | None = None,
    ) -> str:
        """
        Summarize a list of messages into a short paragraph.

        Parameters
        ----------
        messages:
            Iterable of dicts with at least:
            - {"role": "user"|"assistant"|"system", "content": str}
        extra_instructions:
            Optional string appended to the system prompt to further steer
            the summarization (e.g., "Focus on user preferences.").

        Returns
        -------
        str
            Summary text. Returns a best-effort fallback string in case of
            errors.

        Error handling
        --------------
        - Logs LLM errors and returns a generic fallback summary instead of
          raising, so the calling code can continue without crashing.
        """
        # Build content to summarize
        text_blocks: List[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")
            # keep very defensive to avoid KeyError
            if not content:
                continue
            text_blocks.append(f"[{role}] {content}")

        if not text_blocks:
            logger.warning(
                "WorkingMemorySummarizer.summarize_messages called with no content; "
                "returning empty summary."
            )
            return ""

        concatenated = "\n".join(text_blocks)

        sys_prompt = (
            "You are a helpful assistant that summarizes conversation history.\n"
            "Given the following chat transcript, write a concise summary that "
            "captures the key facts, user preferences, decisions, and unresolved "
            "questions.\n\n"
            "Rules:\n"
            "- ONLY use information explicitly present in the transcript.\n"
            "- Do NOT invent or hallucinate facts.\n"
            "- Be as concise as possible.\n"
        )

        if extra_instructions:
            sys_prompt += f"\nAdditional instructions:\n{extra_instructions}\n"

        llm_messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    "Summarize this conversation:\n\n"
                    f"{concatenated}\n\n"
                    "Summary:"
                ),
            },
        ]

        try:
            logger.debug(
                "Calling LLM for working memory summarization; num_input_msgs=%d",
                len(text_blocks),
            )
            summary = await generate_text(
                llm=self._llm,
                messages=llm_messages,
                max_tokens=self._max_summary_tokens,
                ctx=LLMCallContext(op="wm_summarize"),
            )
            logger.info("Working memory summarization completed successfully.")
            return summary.strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Error during working memory summarization: %s", exc)
            # Fallback: truncate concatenated text
            fallback = concatenated[:500]
            logger.warning("Returning truncated transcript as fallback summary.")
            return fallback
