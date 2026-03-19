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
from ...stores.base_sql_store import DEFAULT_TENANT_ID
from ...types import RuntimeContext, SessionScope
from ..working_memory.core import legacy_session_scope_for_user
from .identity import normalize_user_id
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

    Developers use UMAMemory.get_structured_context() outside this pipeline.
    """

    def __init__(self, memory_client: Any, hooks: Any, promotion_policy: Optional[PromotionPolicy] = None) -> None:
        self.mem = memory_client
        self.hooks = hooks
        self.promotion_policy = promotion_policy
        self._post_turn_queue: List[Dict[str, Any]] = []
        if promotion_policy is None:
            logger.info("PromotionPolicy disabled (none provided).")
        else:
            try:
                enabled = promotion_policy.is_enabled() if hasattr(promotion_policy, "is_enabled") else True
                logger.info("PromotionPolicy configured (enabled=%s, max_promotions_per_turn=%s).", enabled, getattr(promotion_policy, "max_promotions_per_turn", "<unset>"))
            except Exception:
                logger.exception("PromotionPolicy provided but failed during initialization checks; disabling promotion.")
                self.promotion_policy = None

    def _post_turn_defer_enabled(self) -> bool:
        """
        Return True if post-turn maintenance should be queued for background execution.
        """
        try:
            cfg = getattr(self.mem, "pipeline_cfg", None)
            if cfg is None:
                return False
            return bool(getattr(cfg, "defer_post_turn", False))
        except Exception:
            return False

    def _post_turn_queue_limit(self) -> int:
        """
        Return max queued post-turn tasks before dropping new ones.
        """
        try:
            cfg = getattr(self.mem, "pipeline_cfg", None)
            limit = int(getattr(cfg, "post_turn_queue_max", 100))
            return max(1, limit)
        except Exception:
            return 100

    def _enqueue_post_turn(self, payload: Dict[str, Any]) -> bool:
        """
        Best-effort enqueue for deferred post-turn tasks.
        """
        limit = self._post_turn_queue_limit()
        if len(self._post_turn_queue) >= limit:
            logger.warning(
                "MemoryPipeline: post-turn queue full (size=%d limit=%d). Dropping task.",
                len(self._post_turn_queue),
                limit,
            )
            return False
        self._post_turn_queue.append(payload)
        return True

    async def process_post_turn_queue(self, *, max_items: Optional[int] = None) -> int:
        """
        Drain deferred post-turn tasks.
        Returns the number of tasks processed.
        """
        if not self._post_turn_queue:
            return 0
        count = 0
        limit = len(self._post_turn_queue) if max_items is None else max(0, int(max_items))
        while self._post_turn_queue and count < limit:
            payload = self._post_turn_queue.pop(0)
            try:
                await self._run_post_turn_tasks(**payload)
                count += 1
            except Exception:
                logger.exception("MemoryPipeline: deferred post-turn task failed; continuing.")
                count += 1
        return count

    async def _run_post_turn_tasks(
        self,
        *,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
        episode: Any,
        facts: Any,
        turn_context: RuntimeContext,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Execute post-turn tasks that can be deferred for performance:
        - semantic ingestion (facts)
        - optional promotion
        - graph updates
        - after_turn hooks
        """
        facts = facts or await self._semantic_ingest(
            user_id,
            assistant_reply,
            turn_id=None,
            turn_context=turn_context,
        )
        if episode is not None and facts:
            for f in facts:
                try:
                    meta = getattr(f, "meta", None) or {}
                    meta.setdefault("source_episode_id", getattr(episode, "id", None))
                    f.meta = meta
                except Exception:
                    continue

        await self._maybe_promote_facts(user_id=user_id, facts=facts)
        await self._update_graph(user_id, episode, facts, turn_context=turn_context)
        await self._run_after_turn_hooks(
            user_id=user_id,
            user_msg=user_msg,
            reply=assistant_reply,
            extra_meta=extra_meta or {},
        )

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
                turn_context = self._resolve_turn_context(
                    user_id=user_id,
                    turn_id=turn_id,
                    extra_meta=extra_meta,
                )
                wm_scope = self._resolve_working_memory_scope(turn_context=turn_context)

                # 1) Hooks
                await self._run_before_turn_hooks(user_id, user_msg)

                # 2) Working memory update
                self._update_working_memory(wm_scope, user_msg, assistant_reply, turn_id=turn_id)

                # 3) WM compaction
                await self._maybe_compact_working_memory(wm_scope)

                # 4) Episodic storage
                episode = await self._store_episode(
                    user_id,
                    user_msg,
                    assistant_reply,
                    turn_id=turn_id,
                    working_memory_scope=wm_scope,
                    turn_context=turn_context,
                )

                if self._post_turn_defer_enabled():
                    enqueued = self._enqueue_post_turn(
                        {
                            "user_id": user_id,
                            "user_msg": user_msg,
                            "assistant_reply": assistant_reply,
                            "episode": episode,
                            "facts": None,
                            "turn_context": turn_context,
                            "extra_meta": extra_meta or {},
                        }
                    )
                    if enqueued:
                        logger.info("MemoryPipeline: deferred post-turn tasks queued.")
                else:
                    # 5) Semantic ingestion
                    facts = await self._semantic_ingest(
                        user_id,
                        assistant_reply,
                        turn_id=turn_id,
                        turn_context=turn_context,
                    )
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
                    await self._update_graph(user_id, episode, facts, turn_context=turn_context)

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
                target_owner = policy.select_target_owner(fact)
                if target_owner is None:
                    continue
            except Exception:
                logger.exception("PromotionPolicy target selection failed; skipping fact.")
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
                promoted = policy.promote(
                    fact,
                    target_owner=target_owner,
                    reason="pipeline_policy",
                )
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

    def _resolve_turn_context(
        self,
        *,
        user_id: str,
        turn_id: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> RuntimeContext:
        agent_id = getattr(self.mem, "agent_id", None)
        if not agent_id:
            raise ValueError("MemoryPipeline.process_turn requires agent_id for scoped turn processing.")

        normalized_user_id = normalize_user_id(user_id)
        meta = extra_meta or {}
        tenant_id = str(meta.get("tenant_id") or DEFAULT_TENANT_ID)
        session_id = meta.get("session_id")
        if session_id:
            return RuntimeContext(
                tenant_id=tenant_id,
                agent_id=agent_id,
                request_id=str(meta.get("request_id") or turn_id),
                user_id=normalized_user_id,
                workspace_id=(str(meta["workspace_id"]) if meta.get("workspace_id") else None),
                session_id=str(session_id),
            )

        if not bool(meta.get("legacy_turn_write_mode", False)):
            raise ValueError(
                "MemoryPipeline.process_turn requires extra_meta['session_id'] for canonical turn writes. "
                "Use extra_meta['legacy_turn_write_mode']=True only for explicit transitional compatibility."
            )

        legacy_scope = legacy_session_scope_for_user(
            tenant_id=tenant_id,
            agent_id=agent_id,
            user_id=normalized_user_id,
        )
        logger.warning(
            "MemoryPipeline: using explicit legacy turn write mode without session_id. "
            "This compatibility path is non-canonical and should be removed after migration."
        )
        return RuntimeContext(
            tenant_id=legacy_scope.tenant_id,
            agent_id=legacy_scope.agent_id,
            request_id=str(meta.get("request_id") or turn_id),
            user_id=legacy_scope.user_id,
            workspace_id=legacy_scope.workspace_id,
            session_id=legacy_scope.session_id,
        )

    def _resolve_working_memory_scope(
        self,
        *,
        turn_context: RuntimeContext,
    ) -> SessionScope:
        return SessionScope(
            tenant_id=turn_context.tenant_id,
            agent_id=turn_context.agent_id,
            session_id=str(turn_context.session_id),
            user_id=turn_context.user_id,
            workspace_id=turn_context.workspace_id,
        )

    def _update_working_memory(
        self,
        scope: Optional[SessionScope],
        user_msg: str,
        assistant_reply: str,
        *,
        turn_id: str,
    ) -> None:
        wm = getattr(self.mem, "working_memory", None)
        if wm is None:
            logger.warning("WorkingMemoryCore not initialized; skipping WM updates.")
            return
        if scope is None:
            return

        try:
            wm.append(
                scope=scope,
                role="user",
                content=user_msg,
                metadata={"source": "user", "turn_id": turn_id},
            )
            if assistant_reply and assistant_reply.strip():
                wm.append(
                    scope=scope,
                    role="assistant",
                    content=assistant_reply,
                    metadata={"source": "assistant", "turn_id": turn_id},
                )
        except Exception:
            logger.exception("Failed to append messages to WorkingMemory; continuing.")

    async def _maybe_compact_working_memory(self, scope: Optional[SessionScope]) -> None:
        wm = getattr(self.mem, "working_memory", None)
        if wm is None or scope is None:
            return
        try:
            await wm.compact(scope=scope)
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
        working_memory_scope: Optional[SessionScope],
        turn_context: RuntimeContext,
    ) -> Any:
        epi = getattr(self.mem, "episodic_core", None)
        wm = getattr(self.mem, "working_memory", None)

        if epi is None:
            logger.warning("EpisodicCore not initialized; skipping episode storage.")
            return None

        try:
            wm_context = wm.get_context(working_memory_scope) if wm and working_memory_scope else []
        except Exception:
            logger.exception("Failed to get WM context for episodic store.")
            wm_context = []

        try:
            from .identity import normalize_user_id
            normalized_user_id = normalize_user_id(user_id)
            return await epi.store_episode(
                owner_type="user",
                owner_id=normalized_user_id,
                user_message=user_msg,
                assistant_reply=assistant_reply,
                working_memory_context=wm_context,
                turn_context=turn_context,
            )
        except Exception:
            logger.exception("EpisodicCore.store_episode failed.")
            return None

    # ------------------------------------------------------------------
    # SEMANTIC INGESTION
    # ------------------------------------------------------------------

    async def _semantic_ingest(
        self,
        user_id: str,
        reply: str,
        *,
        turn_id: str,
        turn_context: RuntimeContext,
    ) -> Any:
        sem = getattr(self.mem, "semantic_core", None)
        if sem is None:
            logger.warning("SemanticCore not initialized; skipping fact ingestion.")
            return []

        # Canonical subject format in UMA-RLM v1: "user:<id>"
        from .identity import normalize_user_id
        try:
            user_subject = normalize_user_id(user_id)
        except Exception:
            logger.exception("SemanticCore.ingest failed; invalid subject user_id=%r", user_id)
            return []

        try:
            return await sem.ingest(
                user_subject,
                reply,
                extra_meta={"turn_id": turn_id},
                turn_context=turn_context,
            )
        except Exception:
            logger.exception("SemanticCore.ingest failed; continuing.")
            return []

    # ------------------------------------------------------------------
    # GRAPH UPDATE
    # ------------------------------------------------------------------

    async def _update_graph(self, user_id: str, episode: Any, facts: Any, *, turn_context: RuntimeContext) -> None:
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
            from .identity import normalize_user_id
            normalized_user_id = normalize_user_id(user_id)
            recent = await core.list_recent(owner_type="user", owner_id=normalized_user_id, n=20)
            scoped_recent = [
                ep for ep in (recent or [])
                if getattr(ep, "session_id", None) == turn_context.session_id
                and getattr(ep, "origin_agent_id", None) == turn_context.agent_id
            ]
            if len(scoped_recent) >= 2:
                prev = scoped_recent[1]
                graph.link_temporal(prev, episode)
        except Exception:
            logger.exception("GraphCore temporal link failed; continuing.")
