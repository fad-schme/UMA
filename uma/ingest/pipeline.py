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

import asyncio
from datetime import datetime, timezone
import logging
import hashlib
from typing import Any, Optional
from uma.memory.promotion import PromotionPolicy
from uma.adapters.observability.context import request_context
from uma.adapters.observability.metrics import increment, timed
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import RuntimeContext, SessionScope, Chunk
from uma.common.identity import normalize_user_id
from uma.common.trust import SourceDescriptor, score_source
from uma.adapters.scanner.injection_scan import scan_artifact_text
logger = logging.getLogger(__name__)


def _get_fact_embedding(fact: Any) -> Optional[list[float]]:
    """Extract an embedding from fact.meta if present."""
    try:
        meta = getattr(fact, "meta", None) or {}
        emb = meta.get("embedding")
        if isinstance(emb, list) and emb:
            return [float(x) for x in emb]
    except Exception as exc:
        logger.debug("_get_fact_embedding: invalid fact embedding: %s", exc, exc_info=True)
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
        # Fire-and-forget background tasks (currently: promotion). We
        # keep strong references so tasks aren't garbage-collected
        # mid-flight (asyncio quirk) and so tests / shutdown paths can
        # await pending work via ``await_pending_background``. Each task
        # removes itself from the set via done_callback.
        self._background_tasks: set[asyncio.Task] = set()
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
        session_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        workspace_id: Optional[str] = None,
        extra_meta: Optional[dict[str, Any]] = None,
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
                        except Exception:  # nosec B112
                            logger.debug("pipeline: fact meta annotation failed, skipping fact", exc_info=True)
                            continue

                # Phase 5: promotion is fire-and-forget. The reply path
                # (and every downstream step in this pipeline) must not
                # wait on promotion latency. The scheduled task runs
                # after this function returns; tests and graceful
                # shutdown can await it via ``await_pending_background``.
                self._schedule_promotion(
                    user_id=user_id,
                    facts=facts,
                    tenant_id=resolved_tenant_id,
                )

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
    def _schedule_promotion(
        self,
        *,
        user_id: str,
        facts: Any,
        tenant_id: str,
    ) -> Optional[asyncio.Task]:
        """Schedule a fire-and-forget promotion pass.

        Returns the created task, or None when there's nothing to
        schedule (no policy bound or no facts to consider). Callers do
        NOT await the result — that would defeat the fire-and-forget
        contract.

        The task is tracked in ``self._background_tasks`` so:
          1. It isn't garbage-collected before it runs (an asyncio
             pitfall — `create_task` returns a reference that the
             caller must retain).
          2. Tests and shutdown paths can await pending work via
             :meth:`await_pending_background`.
        The done-callback removes the task from the set on completion.
        """
        if self.promotion_policy is None:
            return None
        if not facts:
            return None
        coro = self._safe_promotion_task(
            user_id=user_id,
            facts=facts,
            tenant_id=tenant_id,
        )
        task = asyncio.create_task(coro, name=f"promotion-{tenant_id}-{user_id}")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _safe_promotion_task(
        self,
        *,
        user_id: str,
        facts: Any,
        tenant_id: str,
    ) -> None:
        """Outer safety net for the fire-and-forget promotion task.

        ``_maybe_promote_facts`` is documented as never raising to the
        caller. This wrapper is defense-in-depth: if a future change to
        the promotion body breaks that contract, the exception would
        otherwise be silently swallowed by asyncio's default
        "Task exception was never retrieved" handling. Logging it
        explicitly here keeps the failure observable.
        """
        try:
            await self._maybe_promote_facts(
                user_id=user_id,
                facts=facts,
                tenant_id=tenant_id,
            )
        except Exception:
            logger.exception(
                "Background promotion task raised; swallowed to protect the reply path."
            )

    async def await_pending_background(self) -> None:
        """Await all pending fire-and-forget background tasks.

        Called by tests to observe the effect of promotion after
        ``process_turn`` returns. Production callers rarely need this,
        but a graceful-shutdown path can use it to drain in-flight work
        before closing stores. Safe to call when no tasks are pending.
        """
        if not self._background_tasks:
            return
        # Snapshot — the done_callback mutates the set during gather.
        pending = list(self._background_tasks)
        await asyncio.gather(*pending, return_exceptions=True)

    async def _maybe_promote_facts(
        self,
        user_id: str,
        facts: Any,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> None:
        """Promote qualifying facts into the agent's KB.

        Contract:
            - NEVER raises to caller (best-effort stage)
            - Bounded by ``policy.max_promotions_per_turn``
            - Requires:
                * a ``PromotionPolicy`` bound on the memory
                * an ``AgentProfile`` set via ``UMAMemory.set_agent_profile``
                * a semantic core with async ``upsert_fact(fact, embedding)``
                * embeddings already present on facts (populated by the
                  extractor; not computed here to keep the pipeline free
                  of LLM/embedder calls)
              Any missing requirement is a silent no-op for the turn.

        Each candidate fact goes through
        :meth:`PromotionPolicy.qualifies_for_agent_kb`, which composes
        quarantine + is_eligible + scope-match against the agent profile.
        There is no "no-profile" pathway — the scope-match gate is the
        only pathway to the agent KB.
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

        procedural_core = getattr(self.mem, "procedural_core", None)
        if procedural_core is None:
            logger.warning(
                "Promotion enabled but procedural_core is missing; skipping promotions."
            )
            return

        policy_agent_id = getattr(policy, "agent_id", None)
        if not policy_agent_id:
            logger.warning(
                "Promotion enabled but policy has no agent_id; skipping promotions."
            )
            return

        max_promotions = getattr(policy, "max_promotions_per_turn", 5)
        try:
            max_promotions = int(max_promotions)
        except Exception:
            max_promotions = 5
        if max_promotions <= 0:
            return

        # Fetch the agent profile once per turn. Required — no legacy
        # is_eligible-only pathway. If the profile is absent or the
        # fetch fails, promotion is a no-op for this turn (an operator
        # must call set_agent_profile to opt in).
        try:
            agent_profile = await procedural_core.get_agent_profile(
                agent_id=policy_agent_id,
                tenant_id=tenant_id,
            )
        except Exception:
            logger.exception(
                "Promotion: get_agent_profile failed for agent_id=%s; skipping promotions this turn.",
                policy_agent_id,
            )
            return
        if agent_profile is None:
            logger.debug(
                "Promotion: no agent_profile set for agent_id=%s; skipping promotions this turn.",
                policy_agent_id,
            )
            return

        promoted_count = 0

        for fact in fact_list:
            if promoted_count >= max_promotions:
                break

            # Embedding is required — promoted rows must be searchable,
            # and the qualifier's embedding branch needs the vector too.
            # Facts without embeddings can never be promoted, so we skip
            # them before running the qualifier.
            embedding = _get_fact_embedding(fact)
            if embedding is None:
                logger.debug(
                    "Skipping promotion for fact id=%r (no embedding present in fact.meta).",
                    getattr(fact, "id", None),
                )
                continue

            try:
                decision = policy.qualifies_for_agent_kb(
                    fact,
                    agent_profile,
                    fact_embedding=embedding,
                )
                if not decision.passed:
                    logger.debug(
                        "Promotion: dropped fact id=%r reasons=%s",
                        getattr(fact, "id", None),
                        decision.reasons,
                    )
                    continue
                target = policy.select_promotion_target(fact)
                if target is None:
                    continue
            except Exception:
                logger.exception("PromotionPolicy target selection failed; skipping fact.")
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
        extra_meta: dict[str, Any],
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
        extra_meta: Optional[dict[str, Any]] = None,
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
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[str]]:
        chunk_core = getattr(self.mem, "chunk_core", None)
        embedder = getattr(self.mem, "embedder", None)
        if chunk_core is None or embedder is None:
            logger.warning("ChunkCore or embedder not initialized; skipping turn chunk persistence.")
            return {"user_source_ids": [], "assistant_source_ids": []}

        owner_id = normalize_user_id(user_id)
        doc_id = f"turn:{turn_context.session_id}:{turn_id}"
        now = datetime.now(timezone.utc)
        caller_meta = {k: v for k, v in (extra_meta or {}).items() if k not in ("owner_type", "owner_id", "tenant_id", "session_id")} or None
        rows: list[tuple[str, Chunk]] = []
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
            chunk_meta: dict[str, Any] = {
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

        persisted: dict[str, list[str]] = {"user_source_ids": [], "assistant_source_ids": []}
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
        source_ids: Optional[list[str]] = None,
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
