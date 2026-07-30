"""
Lifecycle hooks for UMA pipeline operations.

Hooks are async and optional. They fire at turn boundaries and fail in
isolation — a hook error never breaks the agent flow.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable


AsyncHook = Callable[..., Awaitable[None]]

logger = logging.getLogger(__name__)


class UMAHooks:
    """Async lifecycle hooks for UMA turn processing."""

    def __init__(self):
        self.before_turn: list[AsyncHook] = []
        self.after_turn: list[AsyncHook] = []

    async def run_hooks(self, group: list[AsyncHook], *args, **kwargs) -> None:
        for hook in group:
            try:
                await hook(*args, **kwargs)
            except Exception:
                logger.exception("UMA Hook failure")

    async def run_before_turn(self, *args, **kwargs) -> None:
        await self.run_hooks(self.before_turn, *args, **kwargs)

    async def run_after_turn(self, *args, **kwargs) -> None:
        await self.run_hooks(self.after_turn, *args, **kwargs)
