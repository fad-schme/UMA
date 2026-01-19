"""
UMA-3 Orchestrator — Production Version
======================================

The orchestrator is the outward-facing "brain" of UMA-3.

It delegates all heavy logic to:
    - UMA3Memory (all memory subsystems)
    - MemoryPipeline (turn-level workflow)
    - Hooks (lifecycle management)

Coding Agent Instructions
-------------------------
- Do NOT embed business logic here.
- Keep this thin: orchestration only.
- Use this as the public interface for agent loops, HTTP handlers, CLI, etc.
"""

from __future__ import annotations

import logging
from typing import Any

from .pipeline import MemoryPipeline

logger = logging.getLogger(__name__)


class UMA3Orchestrator:
    """
    High-level agent orchestrator for UMA-3.

    Provides:
        async handle_turn(user_id: str, user_msg: str) -> str

    The orchestrator is responsible for:
        - Starting/stopping turn execution
        - Calling into the MemoryPipeline
        - Catching/logging failures
        - Exposing a clean API for external applications
    """

    def __init__(self, memory_client: Any) -> None:
        self.mem = memory_client
        self.pipeline = MemoryPipeline(memory_client, memory_client.hooks)
        logger.info("UMA3Orchestrator initialized.")

    async def handle_turn(self, user_id: str, user_msg: str) -> str:
        """
        Execute a full turn through the UMA-3 pipeline.

        Steps executed (inside pipeline):
            1. Before-turn hooks
            2. Working memory updates
            3. Hybrid retrieval (core)
            4. LLM reasoning
            5. Episodic memory store
            6. Semantic ingestion
            7. Graph updates (optional)
            8. After-turn hooks
            9. After-response hooks
        """
        logger.info("UMA3Orchestrator: starting turn for user=%s", user_id)

        try:
            reply = await self.pipeline.process_turn(user_id, user_msg)
        except Exception:
            logger.exception("UMA3Orchestrator: turn failed for user=%s", user_id)
            raise

        logger.info("UMA3Orchestrator: completed turn for user=%s", user_id)
        return reply