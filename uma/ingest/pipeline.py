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
    5. Semantic ingestion (facts extracted from user message)
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

from collections import deque
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import hashlib
import threading
from typing import Any, Dict, Optional, List
from uma.memory.promotion import PromotionPolicy
from uma.adapters.observability.context import request_context
from uma.adapters.observability.metrics import increment, timed
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import RuntimeContext, SessionScope, Chunk
from uma.common.identity import normalize_user_id
from uma.common.trust import SourceDescriptor, score_source
from uma.common.injection_scan import scan_content, apply_scan, quarantine_enabled, scan_artifact_text
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DeferredPostTurnTask:
    user_id: str
    user_msg: str
    assistant_reply: str
    episode: Any
    facts: Any
    turn_context: RuntimeContext
    extra_meta: Dict[str, Any]


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

    Retrieval happens outside this pipeline through the bound runtime/request-handle path.
    """

    def __init__(self, memory_client: Any, hooks: Any, promotion_policy: Optional[PromotionPolicy] = None) -> None:
        self.mem = memory_client
        self.hooks = hooks
        self.promotion_policy = promotion_policy
        self._post_turn_queue: deque[_DeferredPostTurnTask] = deque()
        self._post_turn_queue_lock = threading.Lock()
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
        task = _DeferredPostTurnTask(
            user_id=payload["user_id"],
            user_msg=payload["user_msg"],
            assistant_reply=payload["assistant_reply"],
            episode=copy.deepcopy(payload["episode"]),
            facts=copy.deepcopy(payload["facts"]),
            turn_context=payload["turn_context"],
            extra_meta=dict(payload.get("extra_meta") or {}),
        )
        with self._post_turn_queue_lock:
            queue_size = len(self._post_turn_queue)
            if queue_size >= limit:
                logger.warning(
                    "MemoryPipeline: post-turn queue full (size=%d limit=%d). Dropping task.",
                    queue_size,
                    limit,
                )
                return False
            self._post_turn_queue.append(task)
            return True

    async def process_post_turn_queue(self, *, max_items: Optional[int] = None) -> int:
        """
        Drain deferred post-turn tasks.
        Returns the number of tasks processed.
        """
        with self._post_turn_queue_lock:
            queue_size = len(self._post_turn_queue)
        if queue_size == 0:
            return 0
        count = 0
        limit = queue_size if max_items is None else max(0, int(max_items))
        while count < limit:
            with self._post_turn_queue_lock:
                if not self._post_turn_queue:
                    break
                payload = self._post_turn_queue.popleft()
            try:
                await self._run_post_turn_tasks(
                    user_id=payload.user_id,
                    user_msg=payload.user_msg,
                    assistant_reply=payload.assistant_reply,
                    episode=payload.episode,
                    facts=payload.facts,
                    turn_context=payload.turn_context,
                    extra_meta=payload.extra_meta,
                )
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
        if not facts:
            if isinstance(user_msg, str) and user_msg.strip():
                facts = list(await self._semantic_ingest(
                    user_id,
                    user_msg,
                    turn_id=None,
                    turn_context=turn_context,
                    source_ids=list((extra_meta or {}).get("user_source_ids") or []),
                    source_kind="turn_user",
                ) or [])
            if isinstance(assistant_reply, str) and assistant_reply.strip():
                assistant_facts = list(await self._semantic_ingest(
                    user_id,
                    assistant_reply,
                    turn_id=None,
                    turn_context=turn_context,
                    source_ids=list((extra_meta or {}).get("assistant_source_ids") or []),
                    source_kind="turn_assistant",
                ) or [])
                facts = (facts or []) + assistant_facts
        facts = facts or []
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
        session_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        workspace_id: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Perform memory updates for a single turn using:

            user_msg          -> final user input
            assistant_reply   -> final agent output

        UMA stores:
            - WM messages
            - Episodic memory
            - Semantic facts
            - Temporal graph edges

        Required execution scope is passed explicitly through function parameters.
        `extra_meta` is optional non-scope metadata only.

        No reply is returned.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("MemoryPipeline.process_turn requires a non-empty session_id.")

        resolved_tenant_id = str(tenant_id or DEFAULT_TENANT_ID).strip()
        if not resolved_tenant_id:
            raise ValueError("MemoryPipeline.process_turn requires a non-empty tenant_id.")

        resolved_session_id = session_id.strip()
        resolved_workspace_id = str(workspace_id).strip() if workspace_id is not None else None
        meta = dict(extra_meta or {})

        with request_context(resolved_session_id):
            with timed("pipeline.process_turn.latency_s"):
                increment("pipeline.process_turn.count")

                turn_id = _compute_turn_id(
                    user_id=user_id,
                    user_msg=user_msg,
                    assistant_reply=assistant_reply,
                    request_id=resolved_session_id,
                )

                turn_context = self._resolve_turn_context(
                    user_id=user_id,
                    session_id=resolved_session_id,
                    tenant_id=resolved_tenant_id,
                    workspace_id=resolved_workspace_id,
                )
                wm_scope = self._resolve_working_memory_scope(turn_context=turn_context)

                await self._run_before_turn_hooks(user_id, user_msg)

                self._update_working_memory(
                    wm_scope,
                    user_msg,
                    assistant_reply,
                    turn_id=turn_id,
                )

                await self._maybe_compact_working_memory(wm_scope)

                chunk_refs = await self._store_turn_chunks(
                    user_id,
                    user_msg,
                    assistant_reply,
                    turn_id=turn_id,
                    turn_context=turn_context,
                    extra_meta=meta,
                )

                episode = await self._store_episode(
                    user_id,
                    user_msg,
                    assistant_reply,
                    turn_id=turn_id,
                    working_memory_scope=wm_scope,
                    turn_context=turn_context,
                    extra_meta=meta,
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
                            "extra_meta": {
                                **meta,
                                "user_source_ids": list(chunk_refs.get("user_source_ids") or []),
                                "assistant_source_ids": list(chunk_refs.get("assistant_source_ids") or []),
                            },
                        }
                    )
                    if enqueued:
                        logger.info("MemoryPipeline: deferred post-turn tasks queued.")
                    return

                facts = []
                if isinstance(user_msg, str) and user_msg.strip():
                    facts = list(await self._semantic_ingest(
                        user_id,
                        user_msg,
                        turn_id=turn_id,
                        turn_context=turn_context,
                        source_ids=list(chunk_refs.get("user_source_ids") or []),
                        source_kind="turn_user",
                    ) or [])

                if isinstance(assistant_reply, str) and assistant_reply.strip():
                    assistant_facts = list(await self._semantic_ingest(
                        user_id,
                        assistant_reply,
                        turn_id=turn_id,
                        turn_context=turn_context,
                        source_ids=list(chunk_refs.get("assistant_source_ids") or []),
                        source_kind="turn_assistant",
                    ) or [])
                    facts = facts + assistant_facts

                if episode is not None and facts:
                    for fact in facts:
                        try:
                            fact_meta = getattr(fact, "meta", None) or {}
                            fact_meta.setdefault("source_episode_id", getattr(episode, "id", None))
                            fact.meta = fact_meta
                        except Exception:
                            continue

                await self._maybe_promote_facts(user_id=user_id, facts=facts)

                await self._update_graph(
                    user_id,
                    episode,
                    facts,
                    turn_context=turn_context,
                )

                await self._run_after_turn_hooks(
                    user_id=user_id,
                    user_msg=user_msg,
                    reply=assistant_reply,
                    extra_meta=meta,
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
                target = policy.select_promotion_target(fact)
                if target is None:
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
                    tenant_id=target[0],
                    owner_type=target[1],
                    owner_id=target[2],
                    workspace_id=target[3],
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
        session_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        workspace_id: Optional[str] = None,
    ) -> RuntimeContext:
        agent_id = getattr(self.mem, "agent_id", None)
        if not agent_id:
            raise ValueError("MemoryPipeline.process_turn requires agent_id for scoped turn processing.")

        normalized_user_id = normalize_user_id(user_id)

        resolved_tenant_id = str(tenant_id or DEFAULT_TENANT_ID).strip()
        if not resolved_tenant_id:
            raise ValueError("MemoryPipeline.process_turn requires a non-empty tenant_id.")

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("MemoryPipeline.process_turn requires a non-empty session_id.")

        resolved_session_id = session_id.strip()
        resolved_workspace_id = str(workspace_id).strip() if workspace_id is not None else None

        return RuntimeContext(
            tenant_id=resolved_tenant_id,
            agent_id=agent_id,
            request_id=resolved_session_id,
            user_id=normalized_user_id,
            workspace_id=resolved_workspace_id,
            session_id=resolved_session_id,
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
        extra_meta: Optional[Dict[str, Any]] = None,
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
            from uma.common.identity import normalize_user_id
            normalized_user_id = normalize_user_id(user_id)
            return await epi.store_episode(
                owner_type="user",
                owner_id=normalized_user_id,
                user_message=user_msg,
                assistant_reply=assistant_reply,
                working_memory_context=wm_context,
                turn_context=turn_context,
                extra_meta=extra_meta,
            )
        except Exception:
            logger.exception("EpisodicCore.store_episode failed.")
            return None

    async def _store_turn_chunks(
        self,
        user_id: str,
        user_msg: str,
        assistant_reply: str,
        *,
        turn_id: str,
        turn_context: RuntimeContext,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        chunk_core = getattr(self.mem, "chunk_core", None)
        embedder = getattr(self.mem, "embedder", None)
        if chunk_core is None or embedder is None:
            logger.warning("ChunkCore or embedder not initialized; skipping turn chunk persistence.")
            return {"user_source_ids": [], "assistant_source_ids": []}

        owner_id = normalize_user_id(user_id)
        doc_id = f"turn:{turn_context.session_id}:{turn_id}"
        now = datetime.now(timezone.utc)
        caller_meta = {k: v for k, v in (extra_meta or {}).items() if k not in ("owner_type", "owner_id", "tenant_id", "session_id")} or None
        rows: List[tuple[str, Chunk]] = []
        for position, (role, text) in enumerate((("user", user_msg), ("assistant", assistant_reply))):
            if not isinstance(text, str) or not text.strip():
                continue
            chunk_id = hashlib.sha256(
                f"turn_chunk:{owner_id}:{turn_context.session_id}:{turn_id}:{role}".encode("utf-8")
            ).hexdigest()
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            trust_score = score_source(
                SourceDescriptor(
                    kind="turn_user" if role == "user" else "turn_assistant",
                    session_id=turn_context.session_id,
                )
            )
            chunk_meta: Dict[str, Any] = {
                "text_hash": text_hash,
                "source_kind": "turn",
                "source_role": role,
                "turn_id": turn_id,
            }
            if caller_meta:
                chunk_meta["caller"] = caller_meta
            trust_score, chunk_meta, chunk_quarantined_at = scan_artifact_text(
                text,
                trust_score,
                chunk_meta,
                log_context=f"turn_chunk/{role}:{turn_id}",
                now=now,
            )
            rows.append(
                (
                    role,
                    Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        text=text,
                        page_range=(1, 1),
                        position=position,
                        source_path=f"turn://{turn_context.session_id}/{turn_id}/{role}",
                        source_hash=text_hash,
                        created_at=now,
                        updated_at=now,
                        owner_type="user",
                        owner_id=owner_id,
                        tenant_id=turn_context.tenant_id,
                        workspace_id=turn_context.workspace_id,
                        origin_agent_id=turn_context.agent_id,
                        origin_user_id=owner_id,
                        origin_session_id=turn_context.session_id,
                        trust_score=trust_score,
                        quarantined_at=chunk_quarantined_at,
                        meta=chunk_meta,
                    ),
                )
            )
        if not rows:
            return {"user_source_ids": [], "assistant_source_ids": []}

        try:
            vectors = await embedder.embed([chunk.text for _, chunk in rows])
        except Exception:
            logger.exception("MemoryPipeline: turn chunk embedding failed.")
            return {"user_source_ids": [], "assistant_source_ids": []}

        persisted: Dict[str, List[str]] = {"user_source_ids": [], "assistant_source_ids": []}
        for (role, chunk), vec in zip(rows, vectors):
            try:
                ok = await chunk_core.upsert_chunk(chunk, vec)
                if ok:
                    persisted[f"{role}_source_ids"].append(chunk.id)
            except Exception:
                logger.exception(
                    "MemoryPipeline: turn chunk upsert failed role=%s chunk_id=%s",
                    role,
                    chunk.id,
                )
        return persisted

    # ------------------------------------------------------------------
    # SEMANTIC INGESTION
    # ------------------------------------------------------------------

    async def _semantic_ingest(
        self,
        user_id: str,
        text: str,
        *,
        turn_id: Optional[str],
        turn_context: RuntimeContext,
        source_ids: Optional[List[str]] = None,
        source_kind: str = "turn_user",
    ) -> Any:
        sem = getattr(self.mem, "semantic_core", None)
        if sem is None:
            logger.warning("SemanticCore not initialized; skipping fact ingestion.")
            return []

        # Canonical subject format in UMA v1: "user:<id>"
        from uma.common.identity import normalize_user_id
        try:
            user_subject = normalize_user_id(user_id)
        except Exception:
            logger.exception("SemanticCore.ingest failed; invalid subject user_id=%r", user_id)
            return []

        try:
            return await sem.ingest(
                user_subject,
                text,
                extra_meta=(
                    {
                        key: value
                        for key, value in {
                            "turn_id": turn_id,
                            "source_ids": list(source_ids or []),
                        }.items()
                        if value
                    }
                    or None
                ),
                turn_context=turn_context,
                source_kind=source_kind,
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
            from uma.common.identity import normalize_user_id
            normalized_user_id = normalize_user_id(user_id)
            recent = await core.list_recent(
                turn_context.tenant_id,
                owner_type="user",
                owner_id=normalized_user_id,
                n=20,
            )
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