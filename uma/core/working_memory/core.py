"""
uma.core.working_memory.core
=============================

WorkingMemoryCore — short-term, mutable working context (MemGPT-style RAM).

Scope (v2)
----------
This component manages only *working memory*:
- append(): add messages
- get_context(): read WM
- compact(): summarize older content when near capacity
- reset(): hard wipe WM for a session scope
- total_tokens(): approximate budget tracking

Non-goals (v1)
--------------
WorkingMemoryCore does NOT:
- perform long-term retrieval
- inject retrieved memories
- build prompts
- generate assistant replies

Long-term memory retrieval is handled by the bound runtime/request-handle path
via `UMARuntime.bind(RuntimeContext(...))` and `UMARequestHandle`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...types import RuntimeContext, SessionScope
from ..utils.identity import normalize_user_id
from .buffer import WorkingMemoryBuffer, WorkingMemoryMessage
from .queue_manager import QueueManager, QueuePolicy
from .summarizer import WorkingMemorySummarizer
from ...adapters.llm.base import LLMInterface

logger = logging.getLogger(__name__)

_LEGACY_WM_SESSION_PREFIX = "legacy-user:"


def legacy_session_scope_for_user(
    *,
    tenant_id: str,
    agent_id: str,
    user_id: str,
) -> SessionScope:
    normalized_user_id = normalize_user_id(user_id)
    return SessionScope(
        tenant_id=tenant_id,
        agent_id=agent_id,
        session_id=f"{_LEGACY_WM_SESSION_PREFIX}{normalized_user_id}",
        user_id=normalized_user_id,
    )


def session_scope_from_runtime_context(context: RuntimeContext) -> Optional[SessionScope]:
    if not isinstance(context, RuntimeContext):
        raise TypeError("context must be a RuntimeContext")
    if not context.session_id:
        return None
    return SessionScope(
        tenant_id=context.tenant_id,
        agent_id=context.agent_id,
        session_id=context.session_id,
        user_id=context.user_id,
        workspace_id=context.workspace_id,
    )


class WorkingMemoryCore:
    """
    In-memory working memory with budget monitoring and LLM compaction.

    Notes
    -----
    - Token usage is approximate (word-based or token_estimate-based).
    - Compaction is best-effort; failures do not crash the system.
    - Emergency prune prevents runaway growth if summarization fails repeatedly.
    """

    def __init__(
        self,
        llm: LLMInterface,
        memory_client: Any,
    ) -> None:
        wm_cfg = getattr(memory_client, "working_memory_cfg", None)
        if wm_cfg is None:
            raise ValueError("WorkingMemoryCore: working_memory_cfg missing from memory client")

        max_tokens = getattr(wm_cfg, "max_tokens", None)
        warning_ratio = getattr(wm_cfg, "warning_ratio", None)
        hard_limit_ratio = getattr(wm_cfg, "hard_limit_ratio", None)
        chunk_size = getattr(wm_cfg, "chunk_size", 20)
        keep_recent_messages = getattr(wm_cfg, "keep_recent_messages", 6)
        keep_recent_token_fraction = getattr(wm_cfg, "keep_recent_token_fraction", 0.10)

        if max_tokens is None or max_tokens <= 0:
            raise ValueError("WorkingMemoryCore: max_tokens must be > 0")
        if warning_ratio is None or not (0.0 < warning_ratio < 1.0):
            raise ValueError("WorkingMemoryCore: warning_ratio must be in (0,1)")
        if hard_limit_ratio is None or not (0.0 < hard_limit_ratio <= 1.5):
            raise ValueError("WorkingMemoryCore: hard_limit_ratio must be in (0,1.5]")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("WorkingMemoryCore: chunk_size must be a positive integer")
        if keep_recent_messages <= 0:
            raise ValueError("WorkingMemoryCore: keep_recent_messages must be > 0")
        if not (0.0 <= keep_recent_token_fraction <= 1.0):
            raise ValueError("WorkingMemoryCore: keep_recent_token_fraction must be in [0,1]")

        self._buffer = WorkingMemoryBuffer(max_tokens=int(max_tokens))
        self._queue = QueueManager(
            QueuePolicy(
                warning_ratio=float(warning_ratio),
                hard_limit_ratio=float(hard_limit_ratio),
            )
        )
        self._summarizer = WorkingMemorySummarizer(llm=llm)

        self._chunk_size = int(chunk_size)
        self._keep_recent_messages = int(keep_recent_messages)
        self._keep_recent_token_fraction = float(keep_recent_token_fraction)

        logger.info(
            "WorkingMemoryCore initialized: max_tokens=%d warning_ratio=%.2f hard_limit_ratio=%.2f",
            self._buffer.max_tokens,
            warning_ratio,
            hard_limit_ratio,
        )

    def append(
        self,
        scope: SessionScope,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkingMemoryMessage:
        """
        Append a new message to working memory.

        This does NOT trigger retrieval or LLM calls.
        """
        if not isinstance(scope, SessionScope):
            raise TypeError("WorkingMemoryCore.append: scope must be a SessionScope.")
        if not role:
            raise ValueError("WorkingMemoryCore.append: role must not be empty.")
        if content is None:
            raise ValueError("WorkingMemoryCore.append: content must not be None.")

        msg = self._buffer.append(scope, role, content, metadata=metadata)

        used = self._buffer.total_tokens(scope)
        max_t = self._buffer.max_tokens

        if self._queue.should_summarize(used, max_t):
            logger.info(
                "WM nearing capacity tenant=%s agent=%s session=%s used=%d/%d",
                scope.tenant_id,
                scope.agent_id,
                scope.session_id,
                used,
                max_t,
            )

        if self._queue.must_evict(used, max_t):
            logger.warning(
                "WM exceeded hard limit tenant=%s agent=%s session=%s used=%d/%d",
                scope.tenant_id,
                scope.agent_id,
                scope.session_id,
                used,
                max_t,
            )

        return msg

    def get_context(self, scope: SessionScope, last_n: Optional[int] = None) -> List[WorkingMemoryMessage]:
        """Return the working memory message list (optionally last N)."""
        if not isinstance(scope, SessionScope):
            raise TypeError("WorkingMemoryCore.get_context: scope must be a SessionScope.")

        ctx = self._buffer.get_context(scope)
        return ctx if last_n is None else ctx[-int(last_n) :]

    def total_tokens(self, scope: SessionScope) -> int:
        """Return approximate token usage for the session-scoped WM."""
        if not isinstance(scope, SessionScope):
            raise TypeError("WorkingMemoryCore.total_tokens: scope must be a SessionScope.")
        return self._buffer.total_tokens(scope)

    def reset(self, scope: SessionScope) -> None:
        """Hard wipe working memory for a session scope."""
        if not isinstance(scope, SessionScope):
            raise TypeError("WorkingMemoryCore.reset: scope must be a SessionScope.")

        self._buffer.replace_messages(scope, [])
        logger.info(
            "WM reset tenant=%s agent=%s session=%s",
            scope.tenant_id,
            scope.agent_id,
            scope.session_id,
        )

    async def compact(self, scope: SessionScope, extra_instructions: Optional[str] = None) -> None:
        """
        Compact WM by summarizing older messages into a single summary node.

        This is best-effort:
        - If summarization fails, WM remains unchanged.
        - If WM grows beyond 2x max_tokens, apply emergency prune.
        """
        if not isinstance(scope, SessionScope):
            raise TypeError("WorkingMemoryCore.compact: scope must be a SessionScope.")

        messages = self._buffer.get_context(scope)
        if not messages:
            return

        used = self._buffer.total_tokens(scope)
        max_t = self._buffer.max_tokens

        # Emergency prune if summarizer repeatedly fails and WM grows without bound.
        if used > 2 * max_t:
            logger.warning(
                "WM emergency prune tenant=%s agent=%s session=%s used=%d > 2*max=%d",
                scope.tenant_id,
                scope.agent_id,
                scope.session_id,
                used,
                2 * max_t,
            )
            self._buffer.replace_messages(scope, messages[-10:])
            return

        if not self._queue.should_summarize(used, max_t):
            return

        # Keep a stable “recent block”
        threshold_tokens = int(max_t * self._keep_recent_token_fraction)
        keep_count = 0
        cum = 0
        for m in reversed(messages):
            cum += int(getattr(m, "token_estimate", 0) or 0)
            keep_count += 1
            if keep_count >= self._keep_recent_messages and cum >= threshold_tokens:
                break

        if keep_count >= len(messages):
            return

        old_msgs = messages[:-keep_count]
        recent_msgs = messages[-keep_count:]

        payload = [{"role": m.role, "content": m.content} for m in old_msgs]

        try:
            # Summarize in chunks to avoid oversized prompts.
            if len(payload) > self._chunk_size:
                chunk_summaries: List[str] = []
                for i in range(0, len(payload), self._chunk_size):
                    chunk = payload[i : i + self._chunk_size]
                    chunk_summary = await self._summarizer.summarize_messages(
                        chunk,
                        extra_instructions=extra_instructions,
                    )
                    chunk_summary = (chunk_summary or "").strip()
                    if chunk_summary:
                        chunk_summaries.append(chunk_summary)

                if not chunk_summaries:
                    logger.warning("WM compact produced no chunk summaries session=%s", scope.session_id)
                    return

                if len(chunk_summaries) == 1:
                    summary = chunk_summaries[0]
                else:
                    summary_msgs = [
                        {"role": "summary", "content": s} for s in chunk_summaries
                    ]
                    summary = await self._summarizer.summarize_messages(
                        summary_msgs,
                        extra_instructions=extra_instructions,
                    )
            else:
                summary = await self._summarizer.summarize_messages(
                    payload,
                    extra_instructions=extra_instructions,
                )
        except Exception:
            logger.exception("WM compact summarization failed session=%s", scope.session_id)
            return

        summary = (summary or "").strip()
        if not summary:
            logger.warning("WM compact returned empty summary session=%s", scope.session_id)
            return

        summary_tokens = self._buffer._estimate_tokens(summary)
        summary_msg = WorkingMemoryMessage(
            role="summary",
            content=summary,
            token_estimate=summary_tokens,
            metadata={"summary_of_indices": list(range(len(old_msgs)))},
        )

        self._buffer.replace_messages(scope, [summary_msg] + recent_msgs)
        logger.info(
            "WM compacted tenant=%s agent=%s session=%s old_len=%d new_len=%d",
            scope.tenant_id,
            scope.agent_id,
            scope.session_id,
            len(messages),
            1 + len(recent_msgs),
        )
