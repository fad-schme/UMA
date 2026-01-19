"""
pipeline.py
===========

UMA-3 Memory Pipeline (Memory-Only Mode)

This pipeline **does not use any LLM** and **does not perform retrieval**.
It receives both the user message and the final assistant reply from the
developer's agent and performs **memory management only**:

    1. before_turn hooks
    2. Working memory update (user + assistant)
    3. Working memory compaction
    4. Episodic memory storage
    5. Semantic ingestion (facts extracted from assistant reply)
    6. Graph update
    7. after_turn hooks

Coding Agent Instructions
-------------------------
- DO NOT add LLM calls here.
- DO NOT add any retrieval logic here.
- Assume UMA3Memory.initialize() has already been called.
- All operations must fail gracefully with logging.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MemoryPipeline:
    """
    UMA-3 internal memory pipeline (memory-only, no LLM).

    Expected UMA3Memory shape:
        - working_memory
        - episodic_core
        - semantic_core
        - graph_core (optional)
        - hooks

    Developers use UMA3Memory.get_user_context() or
    retrieval_service.retrieve() outside this pipeline.
    """

    def __init__(self, memory_client: Any, hooks: Any) -> None:
        self.mem = memory_client
        self.hooks = hooks
        logger.info("MemoryPipeline initialized (memory-only mode).")

    # ------------------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # ------------------------------------------------------------------

    async def process_turn(
        self,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Perform memory updates for a single turn using:

            user_msg          → final user input
            assistant_reply   → final agent output (external LLM)

        UMA-3 stores:
            - WM messages
            - Episodic memory
            - Semantic facts
            - Temporal graph edges (optional)

        No reply is returned.
        """

        # 1) Hooks
        await self._run_before_turn_hooks(user_id, user_msg)

        # 2) Working memory update
        self._update_working_memory(user_id, user_msg, assistant_reply)

        # 3) WM compaction
        await self._maybe_compact_working_memory(user_id)

        # 4) Episodic storage
        episode = await self._store_episode(user_id, user_msg, assistant_reply)

        # 5) Semantic ingestion
        await self._semantic_ingest(user_id, assistant_reply)

        # 6) Graph update
        await self._update_graph(episode)

        # 7) Hooks
        await self._run_after_turn_hooks(
            user_id=user_id,
            user_msg=user_msg,
            reply=assistant_reply,
            extra_meta=extra_meta or {},
        )

    # ------------------------------------------------------------------
    # HOOKS
    # ------------------------------------------------------------------

    async def _run_before_turn_hooks(self, user_id: str, user_msg: str) -> None:
        try:
            await self.hooks.run_before_turn(user_id=user_id, user_message=user_msg)
        except Exception:
            logger.exception("before_turn hooks failed; continuing.")

    async def _run_after_turn_hooks(
        self,
        user_id: str,
        user_msg: str,
        reply: str,
        extra_meta: Dict[str, Any],
    ) -> None:
        try:
            await self.hooks.run_after_turn(
                user_id=user_id,
                user_message=user_msg,
                assistant_reply=reply,
                extra_meta=extra_meta,
            )
        except Exception:
            logger.exception("after_turn hooks failed; continuing.")

    # ------------------------------------------------------------------
    # WORKING MEMORY
    # ------------------------------------------------------------------

    def _update_working_memory(
        self,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
    ) -> None:
        wm = getattr(self.mem, "working_memory", None)
        if wm is None:
            logger.warning("WorkingMemoryCore not initialized; skipping WM updates.")
            return

        try:
            wm.append(
                user_id=user_id,
                role="user",
                content=user_msg,
                metadata={"source": "user"},
            )
            wm.append(
                user_id=user_id,
                role="assistant",
                content=assistant_reply,
                metadata={"source": "assistant"},
            )
        except Exception:
            logger.exception("Failed to append messages to WorkingMemory; continuing.")

    async def _maybe_compact_working_memory(self, user_id: str) -> None:
        wm = getattr(self.mem, "working_memory", None)
        if wm is None:
            return
        try:
            await wm.compact(user_id=user_id)
        except Exception:
            logger.exception("WorkingMemory compact failed; continuing.")

    # ------------------------------------------------------------------
    # EPISODIC STORAGE
    # ------------------------------------------------------------------

    async def _store_episode(
        self,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
    ) -> Any:
        epi = getattr(self.mem, "episodic_core", None)
        wm = getattr(self.mem, "working_memory", None)

        if epi is None:
            logger.warning("EpisodicCore not initialized; skipping episode storage.")
            return None

        try:
            wm_context = wm.get_context(user_id) if wm else []
        except Exception:
            logger.exception("Failed to get WM context for episodic store.")
            wm_context = []

        try:
            return await epi.store_episode(
                user_id=user_id,
                user_message=user_msg,
                assistant_reply=assistant_reply,
                working_memory_context=wm_context,
            )
        except Exception:
            logger.exception("EpisodicCore.store_episode failed.")
            return None

    # ------------------------------------------------------------------
    # SEMANTIC INGESTION
    # ------------------------------------------------------------------

    async def _semantic_ingest(self, user_id: str, reply: str) -> None:
        sem = getattr(self.mem, "semantic_core", None)
        if sem is None:
            logger.warning("SemanticCore not initialized; skipping fact ingestion.")
            return
        try:
            await sem.ingest(subject=user_id, text=reply)
        except Exception:
            logger.exception("SemanticCore.ingest failed; continuing.")

    # ------------------------------------------------------------------
    # GRAPH UPDATE
    # ------------------------------------------------------------------

    async def _update_graph(self, episode: Any) -> None:
        graph = getattr(self.mem, "graph_core", None)
        if graph is None or episode is None:
            return
        try:
            graph.add_episode(episode)
        except Exception:
            logger.exception("GraphCore.add_episode failed; continuing.")