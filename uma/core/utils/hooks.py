"""
Lifecycle Hooks for UMA Agent

This module defines well-structured callback hooks that the UMA Orchestrator
uses to manage lifecycle events around each user turn.

Hooks add extensibility and clean separation of concerns:

- before_turn
- after_turn
- after_response
- after_memory_update

Coding agent instructions
-------------------------
- Hooks MUST be async.
- Keep hooks idempotent: running twice should cause no errors.
- Use hooks for plugin-based extensibility (e.g., metrics, analytics).
"""

from __future__ import annotations

from typing import Awaitable, Callable, Dict, List


AsyncHook = Callable[..., Awaitable[None]]


class UMAHooks:
    """
    Defines asynchronous lifecycle hooks for UMA operations.

    Hooks are OPTIONAL. They are lists of functions.
    """

    def __init__(self):
        self.before_turn: List[AsyncHook] = []
        self.after_turn: List[AsyncHook] = []
        self.after_response: List[AsyncHook] = []
        self.after_memory_update: List[AsyncHook] = []

    # Utility to call all hooks in a group
    async def run_hooks(self, group: List[AsyncHook], *args, **kwargs) -> None:
        for hook in group:
            try:
                await hook(*args, **kwargs)
            except Exception:
                # Never break agent flow — hooks fail in isolation.
                import logging
                logging.getLogger(__name__).exception("UMA Hook failure")

    async def run_before_turn(self, *args, **kwargs) -> None:
        await self.run_hooks(self.before_turn, *args, **kwargs)

    async def run_after_turn(self, *args, **kwargs) -> None:
        await self.run_hooks(self.after_turn, *args, **kwargs)

    async def run_after_response(self, *args, **kwargs) -> None:
        await self.run_hooks(self.after_response, *args, **kwargs)

    async def run_after_memory_update(self, *args, **kwargs) -> None:
        await self.run_hooks(self.after_memory_update, *args, **kwargs)
