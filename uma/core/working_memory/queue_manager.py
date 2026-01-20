"""
Queue management logic for UMA-3 working memory.

This module contains a small policy engine that determines:

- When to *warn* that summarization should be considered.
- When to *require* eviction/summarization before adding more context.

It does not perform summarization itself; it only decides based on usage.

Typical usage:
--------------
1. WorkingMemoryBuffer computes `total_tokens(user_id)`.
2. QueueManager is consulted with (used_tokens, max_tokens).
3. If `should_summarize` is True, Summarizer is invoked.
4. If `must_evict` is True, the system must compress before adding more text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueuePolicy:
    """
    Configuration for working memory queue behavior.

    Attributes
    ----------
    warning_ratio:
        Fraction of the max token budget at which to *start* summarizing.
        e.g., 0.7 means "when total_tokens >= 70% of max_tokens".

    hard_limit_ratio:
        Fraction at which we must *force* summarization/eviction before
        adding more content. e.g., 0.95.
    """

    warning_ratio: float = 0.7
    hard_limit_ratio: float = 0.95

    def __post_init__(self) -> None:
        if not (0.0 < self.warning_ratio <= 1.0):
            raise ValueError("QueuePolicy.warning_ratio must be in (0, 1]")
        if not (0.0 < self.hard_limit_ratio <= 1.0):
            raise ValueError("QueuePolicy.hard_limit_ratio must be in (0, 1]")
        if self.warning_ratio > self.hard_limit_ratio:
            raise ValueError(
                "QueuePolicy.warning_ratio cannot exceed hard_limit_ratio"
            )


class QueueManager:
    """
    Implements simple threshold-based decisions for working memory management.

    Responsibilities
    ----------------
    - Given current usage and max_tokens, decide if:
        - It's safe to continue.
        - Summarization should be triggered soon.
        - Summarization/eviction is required immediately.

    This class is intentionally stateless; all state lives in the buffer.
    """

    def __init__(self, policy: QueuePolicy) -> None:
        self._policy = policy
        logger.info(
            "Initialized QueueManager with warning_ratio=%.2f hard_limit_ratio=%.2f",
            policy.warning_ratio,
            policy.hard_limit_ratio,
        )

    @property
    def policy(self) -> QueuePolicy:
        """Return the current queue policy."""
        return self._policy

    def should_summarize(self, used_tokens: int, max_tokens: int) -> bool:
        """
        Return True if summarization should be considered.

        This is a soft signal: the system may continue to operate without
        immediately summarizing, but should plan to compress older context.

        Parameters
        ----------
        used_tokens:
            Current estimated token usage.
        max_tokens:
            Soft context window limit.

        Returns
        -------
        bool
        """
        if max_tokens <= 0:
            logger.warning(
                "QueueManager.should_summarize called with non-positive max_tokens=%d",
                max_tokens,
            )
            return False

        ratio = used_tokens / max_tokens
        decision = ratio >= self._policy.warning_ratio
        logger.debug(
            "QueueManager.should_summarize: used=%d max=%d ratio=%.3f -> %s",
            used_tokens,
            max_tokens,
            ratio,
            decision,
        )
        return decision

    def must_evict(self, used_tokens: int, max_tokens: int) -> bool:
        """
        Return True if summarization/eviction must be performed immediately.

        If True, the system should *not* add more raw messages without first
        summarizing/removing some older content.

        Parameters
        ----------
        used_tokens:
            Current estimated token usage.
        max_tokens:
            Soft context window limit.

        Returns
        -------
        bool
        """
        if max_tokens <= 0:
            logger.warning(
                "QueueManager.must_evict called with non-positive max_tokens=%d",
                max_tokens,
            )
            return False

        ratio = used_tokens / max_tokens
        decision = ratio >= self._policy.hard_limit_ratio
        logger.debug(
            "QueueManager.must_evict: used=%d max=%d ratio=%.3f -> %s",
            used_tokens,
            max_tokens,
            ratio,
            decision,
        )
        return decision