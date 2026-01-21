"""
WorkingMemoryCore (Production Version)
=====================================
High-level orchestrator for working memory subsystem.

Provides:
    - append(user_id, role, content, metadata)
    - get_context(user_id, last_n=None)
    - compact(user_id, extra_instructions=None)
    - total_tokens(user_id)

Responsibilities:
- Manage token budgets
- Trigger LLM-based summarization when needed
- Integrate buffer + queue + summarizer
- Remain 100% internal to UMAMemory (no monkey patching)

Coding Agent Instructions
-------------------------
- Do NOT attach methods dynamically onto UMAMemory here.
- Keep this orchestrator thin; heavy logic is in buffer/summarizer.
- ALL failures must be logged but should never break UMA.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from .buffer import WorkingMemoryBuffer, WorkingMemoryMessage
from .queue_manager import QueueManager, QueuePolicy
from .summarizer import WorkingMemorySummarizer

logger = logging.getLogger(__name__)


class WorkingMemoryCore:
    """
    High-level working memory subsystem.

    Parameters
    ----------
    llm : Any
        LLMInterface-compatible object for summarization.
    max_tokens : int
        Soft token budget for working memory.
    warning_ratio : float
        Ratio threshold to *consider* summarization.
    hard_limit_ratio : float
        Ratio threshold to *require* summarization immediately.
    """

    def __init__(
        self,
        llm: Any,
        max_tokens: int = 4096,
        warning_ratio: float = 0.7,
        hard_limit_ratio: float = 0.95,
    ) -> None:

        self.buffer = WorkingMemoryBuffer(max_tokens=max_tokens)
        self.queue = QueueManager(
            QueuePolicy(
                warning_ratio=warning_ratio,
                hard_limit_ratio=hard_limit_ratio,
            )
        )
        self.summarizer = WorkingMemorySummarizer(llm=llm)

        logger.info(
            "WorkingMemoryCore initialized (max_tokens=%d, warn=%.2f, hard=%.2f)",
            max_tokens,
            warning_ratio,
            hard_limit_ratio,
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def append(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkingMemoryMessage:
        """
        Append a message to working memory and check thresholds.
        """
        msg = self.buffer.append(user_id, role, content, metadata=metadata)

        used = self.buffer.total_tokens(user_id)
        max_t = self.buffer.max_tokens

        if self.queue.should_summarize(used, max_t):
            logger.info(
                "WorkingMemory nearing capacity for user=%s (used=%d/%d). "
                "Consider compact().",
                user_id,
                used,
                max_t,
            )

        if self.queue.must_evict(used, max_t):
            logger.warning(
                "WorkingMemory exceeded HARD LIMIT for user=%s (used=%d/%d). "
                "Summarization required soon.",
                user_id,
                used,
                max_t,
            )

        return msg

    def get_context(self, user_id: str, last_n: Optional[int] = None) -> List[WorkingMemoryMessage]:
        """Return working memory context."""
        ctx = self.buffer.get_context(user_id)
        return ctx if last_n is None else ctx[-last_n:]

    def total_tokens(self, user_id: str) -> int:
        """Return approximate total token usage."""
        return self.buffer.total_tokens(user_id)

    # ------------------------------------------------------------------
    # SUMMARIZATION / COMPACTION LOGIC
    # ------------------------------------------------------------------

    async def compact(self, user_id: str, extra_instructions: Optional[str] = None) -> None:
        """
        Summarize and replace older WM messages with a single summary message.

        Strategy:
        ---------
        - If under warning threshold → do nothing.
        - Keep the last K messages verbatim (K=6 default).
        - Summarize the rest using the LLM.
        """
        messages = self.buffer.get_context(user_id)
        if not messages:
            logger.debug("compact: no messages for user=%s", user_id)
            return

        used = self.buffer.total_tokens(user_id)
        max_t = self.buffer.max_tokens

        if not self.queue.should_summarize(used, max_t):
            return

        KEEP_N = 6
        if len(messages) <= KEEP_N:
            return

        old_msgs = messages[:-KEEP_N]
        new_msgs = messages[-KEEP_N:]

        to_summarize = [{"role": m.role, "content": m.content} for m in old_msgs]

        try:
            summary_text = await self.summarizer.summarize_messages(
                to_summarize,
                extra_instructions=extra_instructions,
            )
        except Exception:
            logger.exception("compact: summarization failed for user=%s", user_id)
            return

        if not summary_text.strip():
            logger.warning("compact: summarizer returned empty summary; skipping.")
            return

        summary_msg = WorkingMemoryMessage(
            role="summary",
            content=summary_text.strip(),
            token_estimate=len(summary_text.split()),
            metadata={"summary_of_indices": list(range(len(old_msgs)))},
        )

        final_msgs = [summary_msg] + new_msgs
        self.buffer.replace_messages(user_id, final_msgs)

        logger.info(
            "WorkingMemory compacted for user=%s: old=%d new=%d",
            user_id,
            len(messages),
            len(final_msgs),
        )