"""
core/working_memory/core.py
===========================

WorkingMemoryCore — UMA-3 Core Subsystem

This module implements MemGPT-style hierarchical working memory as a
FIRST-CLASS CORE SERVICE (not a feature).

It:

- Maintains a per-user working memory buffer of recent messages.
- Estimates token usage and consults QueueManager for summarization decisions.
- Uses WorkingMemorySummarizer to compress old context into a summary message.

Public API (used by UMA3Memory + MemoryPipeline + RetrievalService)
-------------------------------------------------------------------
    working_memory = WorkingMemoryCore(...)

    # Add new messages:
    working_memory.append(user_id, role, content, metadata=None)

    # Read context for retrieval/LLM:
    messages = working_memory.get_context(user_id, last_n=None)

    # Check token usage:
    tokens = working_memory.total_tokens(user_id)

    # Compact when nearing limits:
    await working_memory.compact(user_id, extra_instructions=None)

Design Notes
------------
- This is a CORE component, not a UMA3Feature.
- No attach() / monkey-patching: UMA3Memory instantiates and holds it directly.
- Episodic/semantic stores are NOT written here; that belongs in higher-level
  orchestrators (Pipeline / EpisodicCore / SemanticCore).

Coding Agent Instructions
-------------------------
- Do NOT add UMA3Feature or attach semantics here.
- Keep this class focused on in-memory working context management.
- Treat summarization failures as non-fatal: log and leave buffer unchanged.
- If you extend policies (e.g., multiple summaries, tagging), do so in methods
  here or in dedicated policy/strategy objects, not in UMA3Memory.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from .buffer import WorkingMemoryBuffer, WorkingMemoryMessage
from .queue_manager import QueueManager, QueuePolicy
from .summarizer import WorkingMemorySummarizer
from ...adapters.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class WorkingMemoryCore:
    """
    UMA-3 core working memory subsystem.

    Responsibilities
    ----------------
    - Manage per-user recent context with a soft token budget.
    - Warn when nearing capacity, and strongly recommend summarization at
      hard-limit threshold.
    - Summarize older messages into a compact summary message when requested.

    This class is used directly by UMA3Memory and MemoryPipeline.
    It is NOT a feature and must not rely on UMA3Feature or attach().
    """

    def __init__(
        self,
        llm: LLMInterface,
        max_tokens: int = 4096,
        warning_ratio: float = 0.7,
        hard_limit_ratio: float = 0.95,
        chunk_size: int = 20,
        keep_recent_messages: int = 4,
        keep_recent_token_fraction: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        llm:
            LLMInterface implementation to be used for summarization.
        max_tokens:
            Soft token budget for working memory per user.
        warning_ratio:
            See QueuePolicy; threshold to *start* summarizing.
        hard_limit_ratio:
            See QueuePolicy; threshold to *require* summarization.
        """
        self._buffer = WorkingMemoryBuffer(max_tokens=max_tokens)
        policy = QueuePolicy(
            warning_ratio=warning_ratio,
            hard_limit_ratio=hard_limit_ratio,
        )
        self._queue = QueueManager(policy=policy)
        self._summarizer = WorkingMemorySummarizer(llm=llm)
        # Configurable chunk size for long-term summarization. UMA3Memory
        # MUST pass `chunk_size` explicitly when constructing this object.
        self._chunk_size = int(chunk_size)

        # Hybrid preservation policy for compaction: keep at least this
        # many recent messages, and ensure recent messages cover at least
        # `keep_recent_token_fraction * max_tokens` tokens before
        # summarizing older messages.
        self._keep_recent_messages = int(keep_recent_messages)
        self._keep_recent_token_fraction = float(keep_recent_token_fraction)

        logger.info(
            "WorkingMemoryCore initialized: max_tokens=%d warning_ratio=%.2f "
            "hard_limit_ratio=%.2f",
            max_tokens,
            warning_ratio,
            hard_limit_ratio,
        )

    # ------------------------------------------------------------------ #
    # Public API (used by UMA3Memory / Pipeline / RetrievalService)
    # ------------------------------------------------------------------ #

    def append(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkingMemoryMessage:
        """Append a message to the working memory buffer.

        This method defers actual summarization to :meth:`compact` and
        only logs when thresholds are crossed.
        """
        try:
            msg = self._buffer.append(user_id, role, content, metadata=metadata)
        except ValueError as exc:
            logger.error(
                "WorkingMemoryCore.append: failed to append message user_id=%s error=%s",
                user_id,
                exc,
            )
            raise

        used_tokens = self._buffer.total_tokens(user_id)
        max_tokens = self._buffer.max_tokens

        if self._queue.should_summarize(used_tokens, max_tokens):
            logger.info(
                "WorkingMemoryCore.append: nearing capacity for user_id=%s (used=%d, max=%d). compact() recommended.",
                user_id,
                used_tokens,
                max_tokens,
            )

        if self._queue.must_evict(used_tokens, max_tokens):
            logger.warning(
                "WorkingMemoryCore.append: exceeded hard limit for user_id=%s (used=%d, max=%d). compact() strongly recommended.",
                user_id,
                used_tokens,
                max_tokens,
            )

        return msg

    def get_context(self, user_id: str, last_n: Optional[int] = None) -> List[WorkingMemoryMessage]:
        """
        Return current working memory messages for the user.

        Parameters
        ----------
        user_id:
            User/session identifier.
        last_n:
            If provided, only the last N messages are returned.

        Returns
        -------
        List[WorkingMemoryMessage]
        """
        messages = self._buffer.get_context(user_id)
        if last_n is not None and last_n > 0:
            return messages[-last_n:]
        return messages

    def total_tokens(self, user_id: str) -> int:
        """
        Return the approximate total token usage for this user's buffer.

        Parameters
        ----------
        user_id:
            User/session identifier.

        Returns
        -------
        int
            Approximate token count for all messages in working memory.
        """
        return self._buffer.total_tokens(user_id)

    def reset(self, user_id: str) -> None:
        """
        Hard wipe of all working memory messages for a given user.
        Useful for emergency cleanup, session resets, or preventing
        unbounded memory growth.
        """
        try:
            self._buffer.replace_messages(user_id, [])
            logger.info(
                "WorkingMemoryCore.reset: cleared all WM messages for user_id=%s",
                user_id,
            )
        except Exception:
            logger.exception(
                "WorkingMemoryCore.reset: failed to clear WM for user_id=%s",
                user_id,
            )

    # ------------------------------------------------------------------ #
    # Summarization / Compaction
    # ------------------------------------------------------------------ #

    async def compact(self, user_id: str, extra_instructions: Optional[str] = None) -> None:
        """
        Compact the working memory for a user by summarizing older messages.

        Strategy (simple baseline):
        ---------------------------
        1. Retrieve all messages for the user.
        2. If total_tokens <= warning_ratio * max_tokens, do nothing.
        3. Otherwise:
            - Keep the most recent N messages untouched (KEEP_RECENT).
            - Summarize the older messages into a single "summary" message.
            - Replace the buffer with [summary_msg] + recent_msgs.

        Parameters
        ----------
        user_id:
            User/session identifier.
        extra_instructions:
            Optional additional instructions for the summarizer (e.g.,
            "Focus only on user preferences and long-term facts.")

        Notes
        -----
        - This method is async because it calls the LLM summarizer.
        - Errors in summarization are logged; in case of errors, the buffer
          is left untouched.
        """
        messages = self._buffer.get_context(user_id)
        if not messages:
            logger.debug(
                "WorkingMemoryCore.compact: no messages for user_id=%s; skipping.",
                user_id,
            )
            return

        used_tokens = self._buffer.total_tokens(user_id)
        max_tokens = self._buffer.max_tokens

        # Emergency prune: if memory grows far beyond limits, wipe oldest entries
        # Threshold: 2 × max_tokens worth of estimated message tokens
        if used_tokens > (max_tokens * 2):
            logger.warning(
                "WorkingMemoryCore.compact: emergency prune triggered for user_id=%s "
                "(used_tokens=%d > 2×max_tokens=%d).",
                user_id,
                used_tokens,
                max_tokens * 2,
            )
            # Keep only the last 10 messages, wipe the rest
            preserved = messages[-10:]
            self._buffer.replace_messages(user_id, preserved)
            return

        if not self._queue.should_summarize(used_tokens, max_tokens):
            logger.debug(
                "WorkingMemoryCore.compact: summarization not needed for user_id=%s "
                "(used=%d, max=%d).",
                user_id,
                used_tokens,
                max_tokens,
            )
            return

        # Hybrid heuristic: decide how many recent messages to keep based
        # on both a minimum message count and a token fraction threshold.
        # We iterate from the end accumulating token counts until both
        # thresholds are met (or we hit the start).
        min_keep = int(getattr(self, "_keep_recent_messages", 6))
        token_fraction = float(getattr(self, "_keep_recent_token_fraction", 0.1))

        threshold_tokens = max_tokens * token_fraction

        # Accumulate from the most recent message backwards.
        cum_tokens = 0
        keep_count = 0
        for m in reversed(messages):
            cum_tokens += getattr(m, "token_estimate", 0)
            keep_count += 1
            if keep_count >= min_keep and cum_tokens >= threshold_tokens:
                break

        if keep_count >= len(messages):
            logger.debug(
                "WorkingMemoryCore.compact: not enough messages to summarize "
                "for user_id=%s (len=%d) (keep_count=%d, min_keep=%d).",
                user_id,
                len(messages),
                keep_count,
                min_keep,
            )
            return

        old_msgs = messages[:-keep_count]
        recent_msgs = messages[-keep_count:]

        # Convert WorkingMemoryMessage to simple dicts for summarizer.
        to_summarize: List[Dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in old_msgs
        ]

        try:
            summary_text = await self._summarizer.summarize_messages(
                to_summarize,
                extra_instructions=extra_instructions,
            )
        except Exception as exc:  # pragma: no cover - summarizer logs internally too
            logger.exception(
                "WorkingMemoryCore.compact: summarizer error for user_id=%s: %s",
                user_id,
                exc,
            )
            return

        if not summary_text:
            logger.warning(
                "WorkingMemoryCore.compact: summarizer returned empty summary "
                "for user_id=%s; leaving buffer unchanged.",
                user_id,
            )
            return

        # Build summary message, tracking which messages it replaces.
        summarized_indices = list(range(len(old_msgs)))
        summary_metadata = {"summary_of_indices": summarized_indices}
        summary_msg = WorkingMemoryMessage(
            role="summary",
            content=summary_text,
            token_estimate=len(summary_text.split()),  # rough estimate
            metadata=summary_metadata,
        )

        new_messages = [summary_msg] + recent_msgs
        self._buffer.replace_messages(user_id, new_messages)

        new_total = self._buffer.total_tokens(user_id)
        logger.info(
            "WorkingMemoryCore.compact: completed summarization for user_id=%s; "
            "old_len=%d new_len=%d old_tokens=%d new_tokens=%d",
            user_id,
            len(messages),
            len(new_messages),
            used_tokens,
            new_total,
        )
    
    # ------------------------------------------------------------------ #
    # Retrieve + Summarize Long-Term Memory (Non-Mutating, Chunked)
    # ------------------------------------------------------------------ #
    async def retrieve_long_memory(
        self,
        user_id: str,
        query_text: str,
        retrieval_service: Any,
        extra_instructions: Optional[str] = None,
    ) -> List[WorkingMemoryMessage]:
        """
        Retrieve episodic/semantic/procedural/graph memory relevant to a query.

        Enhancements:
        - CHUNKED summarization controlled by `self._chunk_size`
        - Predicate-weighted sorting for semantic facts (high-importance first)
        - Non-mutating: produces synthetic WM messages without modifying real WM
        """

        if not retrieval_service:
            logger.warning("WorkingMemoryCore.retrieve_long_memory called without retrieval_service.")
            return []

        # ----------------------
        # 1. Perform retrieval
        # ----------------------
        try:
            retrieved = await retrieval_service.retrieve(
                user_id=user_id,
                memory_type="all",
                query_text_or_embedding=query_text,
            )
        except Exception:
            logger.exception("retrieve_long_memory: retrieval failed for user_id=%s", user_id)
            return []

        # ----------------------
        # 2. Build pseudo-messages for chunking
        # ----------------------
        text_blocks: List[Dict[str, str]] = []

        def _fmt(label: str, obj: Any):
            return {"role": "system", "content": f"[{label}] {obj}"}

        # Episodic
        for ep in retrieved.get("episodes", []):
            summary = getattr(ep, "summary", None) or getattr(ep, "text", None) or repr(ep)
            text_blocks.append(_fmt("EPISODE", summary))

        # ----------------------
        # Semantic (predicate-weighted sorting)
        # ----------------------
        semantic_facts = retrieved.get("semantic", [])

        predicate_weights = {
            "prefers": 3.0,
            "likes": 2.5,
            "dislikes": 2.5,
            "works_on": 2.0,
            "interested_in": 2.0,
        }

        def _semantic_weight(fact):
            if isinstance(fact, dict):
                pred = fact.get("predicate")
            else:
                pred = getattr(fact, "predicate", None)
            return predicate_weights.get(pred, 1.0)

        semantic_facts_sorted = sorted(
            semantic_facts,
            key=lambda f: _semantic_weight(f),
            reverse=True,
        )

        for fact in semantic_facts_sorted:
            if isinstance(fact, dict):
                text_blocks.append(_fmt("FACT", fact))
            else:
                text_blocks.append(_fmt("FACT", repr(fact)))

        # Procedural
        for skill in retrieved.get("procedural", []):
            if hasattr(skill, "name"):
                text_blocks.append(_fmt("SKILL", f"{skill.name}: {getattr(skill,'description','')}"))
            else:
                text_blocks.append(_fmt("SKILL", repr(skill)))

        # Graph
        for node in retrieved.get("graph", []):
            text_blocks.append(_fmt("GRAPH", repr(node)))

        if not text_blocks:
            return []

        # ----------------------
        # 3. Chunking
        # ----------------------
        chunk_size = getattr(self, "_chunk_size", 20)
        chunks = [text_blocks[i:i + chunk_size] for i in range(0, len(text_blocks), chunk_size)]

        chunk_summaries: List[str] = []

        for idx, chunk in enumerate(chunks):
            try:
                summary = await self._summarizer.summarize_messages(
                    chunk,
                    extra_instructions=(
                        extra_instructions
                        or "Summarize these retrieved memory items into a concise intermediate note."
                    ),
                )
                if summary:
                    chunk_summaries.append(summary)
            except Exception:
                logger.exception("Failed summarizing chunk %d", idx)

        if not chunk_summaries:
            return []

        # ----------------------
        # 4. Final SUMMARY combining chunks
        # ----------------------
        try:
            final_summary = await self._summarizer.summarize_messages(
                [{"role": "system", "content": s} for s in chunk_summaries],
                extra_instructions=(
                    extra_instructions
                    or "Merge all chunk summaries into a single concise note for reasoning."
                ),
            )
        except Exception:
            logger.exception("Failed to merge chunk summaries.")
            final_summary = "\n".join(chunk_summaries)

        if not final_summary:
            return []

        # ----------------------
        # 5. Return synthetic WM node
        # ----------------------
        return [
            WorkingMemoryMessage(
                role="system",
                content=final_summary,
                token_estimate=len(final_summary.split()),
                metadata={"source": "uma3_retrieved_context"},
            )
        ]