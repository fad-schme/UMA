"""
pipeline.py
===========

UMA Memory Pipeline (Memory-Only Mode)

This pipeline **does not perform retrieval** and **does not generate replies**.
It does use LLMs for summarization and semantic fact extraction.
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
- Assume UMAMemory.initialize() has already been called.
- All operations must fail gracefully with logging.
"""

from __future__ import annotations

import logging
import hashlib
from typing import Any, Dict, Optional, List
from .promotion import PromotionPolicy
from ...adapters.observability.context import request_context
from ...adapters.observability.metrics import increment, timed
logger = logging.getLogger(__name__)

def _get_fact_embedding(fact: Any) -> Optional[List[float]]:
    """Extract an embedding from fact.meta if present."""
    try:
        meta = getattr(fact, "meta", None) or {}
        emb = meta.get("embedding")
        if isinstance(emb, list) and emb:
            return [float(x) for x in emb]
    except Exception:
        return None
    return None


def _compute_turn_id(
    *,
    user_id: str,
    user_msg: str,
    assistant_reply: str,
    request_id: Optional[str] = None,
) -> str:
    """
    Deterministic idempotency key for a turn.

    Notes
    -----
    - Uses user_id + user_msg + assistant_reply.
    - Optionally includes request_id when present (to separate identical turns from distinct requests).
    """
    h = hashlib.sha256()
    h.update((user_id or "").encode("utf-8"))
    h.update(b"\n")
    h.update((user_msg or "").encode("utf-8"))
    h.update(b"\n")
    h.update((assistant_reply or "").encode("utf-8"))
    if request_id:
        h.update(b"\n")
        h.update(str(request_id).encode("utf-8"))
    return f"turn_{h.hexdigest()[:24]}"



class MemoryPipeline:
    """
    UMA internal memory pipeline (memory-only, no LLM).

    Expected UMAMemory shape:
        - working_memory
        - episodic_core
        - semantic_core
        - graph_core (optional)
        - hooks

    Developers use UMAMemory.get_structured_context() or
    retrieval_service.retrieve() outside this pipeline.
    """

    def __init__(self, memory_client: Any, hooks: Any, promotion_policy: Optional[PromotionPolicy] = None) -> None:
        self.mem = memory_client
        self.hooks = hooks
        self.promotion_policy = promotion_policy
        if promotion_policy is None:
            logger.info("PromotionPolicy disabled (none provided).")
        else:
            try:
                enabled = promotion_policy.is_enabled() if hasattr(promotion_policy, "is_enabled") else True
                logger.info("PromotionPolicy configured (enabled=%s, max_promotions_per_turn=%s).", enabled, getattr(promotion_policy, "max_promotions_per_turn", "<unset>"))
            except Exception:
                logger.exception("PromotionPolicy provided but failed during initialization checks; disabling promotion.")
                self.promotion_policy = None

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

        UMA stores:
            - WM messages
            - Episodic memory
            - Semantic facts
            - Temporal graph edges (optional)

        No reply is returned.
        """

        with request_context(extra_meta.get("request_id") if extra_meta else None):
            with timed("pipeline.process_turn.latency_s"):
                increment("pipeline.process_turn.count")

                request_id = (extra_meta or {}).get("request_id") if isinstance(extra_meta, dict) else None
                turn_id = _compute_turn_id(
                    user_id=user_id,
                    user_msg=user_msg,
                    assistant_reply=assistant_reply,
                    request_id=str(request_id) if request_id else None,
                )

                # 1) Hooks
                await self._run_before_turn_hooks(user_id, user_msg)

                # 2) Working memory update
                self._update_working_memory(user_id, user_msg, assistant_reply, turn_id=turn_id)

                # 3) WM compaction
                await self._maybe_compact_working_memory(user_id)

                # 4) Episodic storage
                episode = await self._store_episode(user_id, user_msg, assistant_reply, turn_id=turn_id)

                # 5) Semantic ingestion
                facts = await self._semantic_ingest(user_id, assistant_reply, turn_id=turn_id)
                if episode is not None and facts:
                    for f in facts:
                        try:
                            meta = getattr(f, "meta", None) or {}
                            meta.setdefault("source_episode_id", getattr(episode, "id", None))
                            f.meta = meta
                        except Exception:
                            continue

                # 5b) Optional promotion of eligible facts to agent KB
                await self._maybe_promote_facts(user_id=user_id, facts=facts)

                # 6) Graph update
                await self._update_graph(user_id, episode, facts)

                # 7) Hooks
                await self._run_after_turn_hooks(
                    user_id=user_id,
                    user_msg=user_msg,
                    reply=assistant_reply,
                    extra_meta=extra_meta or {},
                )

    # ------------------------------------------------------------------
    # PROMOTION (OPTIONAL)
    # ------------------------------------------------------------------
    async def _maybe_promote_facts(self, user_id: str, facts: Any) -> None:
        """Optionally promote eligible facts to agent-scoped knowledge.

        This is a best-effort stage:
        - NEVER raises to caller
        - bounded by policy.max_promotions_per_turn
        - only promotes if we can supply an embedding (so vector search stays correct)

        Requirements on UMAMemory shape:
        - semantic_core with async upsert_fact(fact, embedding)
        # NOTE:
        # Promotion requires embeddings to already exist on facts.
        # v1 does NOT compute embeddings here to avoid adding LLM / embedder calls
        # to the pipeline. Facts without embeddings are skipped intentionally.

        """
        policy = getattr(self, "promotion_policy", None)
        if policy is None:
            return

        try:
            if hasattr(policy, "is_enabled") and not policy.is_enabled():
                return
        except Exception:
            logger.exception("PromotionPolicy.is_enabled() failed; disabling promotion for this turn.")
            return

        if not facts:
            return

        # Ensure iterable
        try:
            fact_list = list(facts)
        except Exception:
            logger.exception("Promotion stage received non-iterable facts; skipping.")
            return

        sem_core = getattr(self.mem, "semantic_core", None)
        if sem_core is None:
            logger.warning("Promotion enabled but semantic_core is missing; skipping promotions.")
            return

        max_promotions = getattr(policy, "max_promotions_per_turn", 5)
        try:
            max_promotions = int(max_promotions)
        except Exception:
            max_promotions = 5
        if max_promotions <= 0:
            return

        promoted_count = 0

        for fact in fact_list:
            if promoted_count >= max_promotions:
                break

            try:
                if not policy.is_eligible(fact):
                    continue
            except Exception:
                logger.exception("PromotionPolicy.is_eligible failed; skipping fact.")
                continue

            # Require embedding to keep the agent KB searchable. If absent, skip.
            embedding = _get_fact_embedding(fact)
            if embedding is None:
                logger.debug(
                    "Skipping promotion for fact id=%r (no embedding present in fact.meta).",
                    getattr(fact, "id", None),
                )
                continue

            try:
                promoted = policy.promote(fact)
            except Exception:
                logger.exception("PromotionPolicy.promote failed; skipping fact.")
                continue

            # Persist promoted fact into agent KB.
            try:
                await sem_core.upsert_fact(promoted, embedding=embedding)
                promoted_count += 1
                logger.info(
                    "Promoted fact id=%r predicate=%r (user_id=%s)",
                    getattr(promoted, "id", getattr(fact, "id", None)),
                    getattr(promoted, "predicate", getattr(fact, "predicate", None)),
                    user_id,
                )
            except Exception:
                logger.exception("Failed to persist promoted fact; continuing.")

        if promoted_count:
            increment("pipeline.promotion.count")

    # ------------------------------------------------------------------
    # HOOKS
    # ------------------------------------------------------------------

    async def _run_before_turn_hooks(self, user_id: str, user_msg: str) -> None:
        if not self.hooks or not hasattr(self.hooks, "run_before_turn"):
            return
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
        if not self.hooks or not hasattr(self.hooks, "run_after_turn"):
            return
        
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
        *,
        turn_id: str,
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
                metadata={"source": "user", "turn_id": turn_id},
            )
            if assistant_reply and assistant_reply.strip():
                wm.append(
                    user_id=user_id,
                    role="assistant",
                    content=assistant_reply,
                    metadata={"source": "assistant", "turn_id": turn_id},
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
        *,
        turn_id: str,
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
                owner_type="user",
                owner_id=user_id,
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

    async def _semantic_ingest(self, user_id: str, reply: str, *, turn_id: str) -> Any:
        sem = getattr(self.mem, "semantic_core", None)
        if sem is None:
            logger.warning("SemanticCore not initialized; skipping fact ingestion.")
            return []

        # Canonical subject format in UMA-RLM v1: "user:<id>"
        from .identity import ensure_user_subject
        try:
            subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("SemanticCore.ingest failed; invalid subject user_id=%r", user_id)
            return []

        try:
            return await sem.ingest(subject=subject, text=reply, extra_meta={"turn_id": turn_id})
        except Exception:
            logger.exception("SemanticCore.ingest failed; continuing.")
            return []

    # ------------------------------------------------------------------
    # GRAPH UPDATE
    # ------------------------------------------------------------------

    async def _update_graph(self, user_id: str, episode: Any, facts: Any) -> None:
        graph = getattr(self.mem, "graph_core", None)
        if graph is None or episode is None:
            return
        try:
            graph.add_episode(episode)
        except Exception:
            logger.exception("GraphCore.add_episode failed; continuing.")

        if facts:
            try:
                graph.add_facts(list(facts))
                graph.link_episode_to_facts(episode, list(facts))
            except Exception:
                logger.exception("GraphCore fact linking failed; continuing.")

        # Link temporal sequence (previous episode -> current)
        try:
            core = getattr(self.mem, "episodic_core", None)
            if core is None:
                return
            recent = await core.list_recent(owner_type="user", owner_id=user_id, n=2)
            if len(recent) >= 2:
                prev = recent[1]
                graph.link_temporal(prev, episode)
        except Exception:
            logger.exception("GraphCore temporal link failed; continuing.")
