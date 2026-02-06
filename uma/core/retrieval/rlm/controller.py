# uma/core/retrieval/rlm/controller.py

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from .context_pack import ContextPack
from .decisions import ControllerDecision, RetrievalAction
from .coverage import assess_coverage
from ..policy import RetrievalPolicy, should_stop
from ...utils.identity import ensure_user_subject
from ...utils.accessors import get_attr_or_key

try:
    from ...utils.user_query_helper import extract_query_terms
except Exception:  # pragma: no cover - optional
    extract_query_terms = None

logger = logging.getLogger(__name__)


class RLMController:
    """
    RLMController — Recursive, bounded *memory navigation* controller.

    IMPORTANT:
    - This controller NEVER answers questions.
    - It ONLY decides what memory to retrieve next.
    - All reasoning is deterministic unless LLM mode is enabled.

    Guarantees
    ----------
    - Store-native actions only (no 'retrieve again')
    - Bounded recursion (steps, env calls, timeout)
    - Deterministic stopping (coverage + novelty)
    """

    def __init__(
        self,
        llm: Any,
        env: Any,
    ) -> None:
        self.llm = llm
        self.env = env

        rlm_cfg = None
        try:
            memory = getattr(env, "_memory", None)
            retrieval_cfg = getattr(memory, "retrieval_cfg", None)
            rlm_cfg = getattr(retrieval_cfg, "rlm", None) if retrieval_cfg else None
        except Exception:
            rlm_cfg = None

        self.timeout_s = float(getattr(rlm_cfg, "timeout_s", 20.0))

        self.max_steps = int(getattr(rlm_cfg, "max_steps", 4))
        self.max_actions_per_step = int(getattr(rlm_cfg, "max_actions_per_step", 2))
        self.max_env_calls = int(getattr(rlm_cfg, "max_env_calls", 12))
        self.max_items_per_type = int(getattr(rlm_cfg, "max_items_per_type", 30))

        self.salience_threshold = float(getattr(rlm_cfg, "salience_threshold", 0.6))
        self.min_semantic_facts = max(1, int(getattr(rlm_cfg, "min_semantic_facts", 4)))
        self.min_high_salience_facts = max(0, int(getattr(rlm_cfg, "min_high_salience_facts", 2)))
        self.min_cluster_summaries = max(0, int(getattr(rlm_cfg, "min_cluster_summaries", 1)))

        self.cluster_k = max(1, int(getattr(rlm_cfg, "cluster_k", 3)))
        self.graph_predicate_limit = max(1, int(getattr(rlm_cfg, "graph_predicate_limit", 2)))
        self.predicate_weights = self._normalize_predicate_weights(
            getattr(rlm_cfg, "predicate_weights", None)
        )

        self.novelty_window = max(1, int(getattr(rlm_cfg, "novelty_window", 2)))
        self.min_recent_novelty = max(0, int(getattr(rlm_cfg, "min_recent_novelty", 1)))

        self.max_new_facts_per_step = max(0, int(getattr(rlm_cfg, "max_new_facts_per_step", 12)))
        self.max_new_chunks_per_step = max(0, int(getattr(rlm_cfg, "max_new_chunks_per_step", 8)))
        self.max_graph_expansions_per_step = max(0, int(getattr(rlm_cfg, "max_graph_expansions_per_step", 1)))

        self.max_state_chars = max(200, int(getattr(rlm_cfg, "max_state_chars", 1200)))
        self.test_mode = bool(getattr(rlm_cfg, "test_mode", False))
        self.semantic_first = bool(getattr(rlm_cfg, "semantic_first", True))
        self.clusters_first = bool(getattr(rlm_cfg, "clusters_first", True))

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def retrieve_context(self, user_id: str, query_text: str) -> ContextPack:
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not query_text or not isinstance(query_text, str):
            raise ValueError("query_text must be a non-empty string")

        policy = RetrievalPolicy(query_text)
        logger.info(
            "RLMController.retrieve_context: start user_id=%s query=%r",
            user_id,
            query_text,
        )
        start = time.time()
        user_subject = ensure_user_subject(user_id)

        # --- TRACE CONTEXT ---
        trace_id = f"rlm:{user_subject}:{int(time.time()*1000)}"
        logger.info(
            "RLM_START trace_id=%s user=%s recall_query=%s",
            trace_id,
            user_subject,
            any(k in query_text.lower() for k in ["remember", "recall", "previous", "earlier", "last time"]),
        )

        agent_id = getattr(self.env, "_agent_id", None)
        pack = ContextPack(
            user_id=user_subject,
            query_text=query_text,
            owner_type=None,
            owner_id=None,
            agent_id=agent_id,
        )

        pack.working_memory = []
        if hasattr(self.env, "_memory"):
            wm = getattr(getattr(self.env, "_memory", None), "working_memory", None)
            if wm is not None:
                pack.working_memory = wm.get_context(user_subject)
        query_embedding = await self.env.get_query_embedding(query_text)

        # Pass trace_id to baseline retrieval
        if policy.recall_score >= 0.75:
            owner_type = "user"
            owner_id = user_subject
        else:
            owner_type = "agent"
            agent_id = getattr(self.env, "_agent_id", None)
            if not agent_id:
                raise ValueError("RLMController.retrieve_context: agent_id is required for agent scope.")
            owner_id = agent_id
        pack.owner_type = owner_type
        pack.owner_id = owner_id
        logger.info("RLM_LANE owner_type=%s owner_id=%s", pack.owner_type, pack.owner_id)
        await self._baseline_retrieval(
            pack,
            query_embedding,
            trace_id=trace_id,
            owner_type=owner_type,
            owner_id=owner_id,
        )
        pack.record_seen()
        logger.debug(
            "RLMController: baseline counts facts=%d chunks=%d episodes=%d graph=%d",
            len(pack.facts),
            len(getattr(pack, "chunks", [])),
            len(pack.episodes),
            len(pack.graph),
        )
        logger.debug(
            "RLMController: step=0 facts preview=%s",
            [self._describe_fact(f)[:180] for f in pack.facts[:5]],
        )

        coverage = self._assess_coverage(pack)
        pack.coverage = coverage
        pack.steps.append(
            {
                "step": 0,
                "phase": "baseline",
                "event": "coverage",
                "counts": {
                    "facts": len(pack.facts),
                    "episodes": len(pack.episodes),
                    "graph": len(pack.graph),
                },
                "coverage": coverage.to_dict(),
            }
        )

        stop, reason = should_stop(
            recall_score=policy.recall_score,
            coverage=coverage.to_dict(),
            calls_made=0,
            max_calls=self.max_env_calls,
            tokens_used=0,  # baseline
            token_budget=self.max_state_chars,
            user_results_count=sum(1 for f in pack.facts if _is_user_owned(f)),
        )
        # --- COVERAGE + STOP DECISION TELEMETRY (baseline) ---
        logger.info(
            "RLM_COVERAGE trace_id=%s step=%d stop=%s reason=%s coverage=%s",
            trace_id,
            0,
            stop,
            reason,
            coverage.to_dict(),
        )
        if stop:
            pack.warnings.append(f"stop:{reason}")
            logger.info("RLMController: stop after baseline reason=%s", reason)
            await self._prune_facts_with_llm(pack)
            # --- TERMINATION TELEMETRY ---
            logger.info(
                "RLM_END trace_id=%s total_steps=%d total_calls=%d facts=%d episodes=%d graph=%d warnings=%s",
                trace_id,
                len(pack.steps),
                0,
                len(pack.facts),
                len(pack.episodes),
                len(pack.graph),
                pack.warnings,
            )
            return pack

        total_env_calls = 0

        for step in range(1, self.max_steps + 1):
            if (time.time() - start) > self.timeout_s:
                pack.warnings.append("stop:timeout")
                break
            if total_env_calls >= self.max_env_calls:
                pack.warnings.append("stop:max_env_calls")
                break

            coverage = self._assess_coverage(pack)
            pack.coverage = coverage
            pack.steps.append(
                {
                    "step": step,
                    "phase": "loop",
                    "event": "coverage",
                    "counts": {
                        "facts": len(pack.facts),
                        "episodes": len(pack.episodes),
                        "graph": len(pack.graph),
                    },
                    "coverage": coverage.to_dict(),
                }
            )
            logger.debug(
                "RLMController: step=%d coverage=%s",
                step,
                coverage.to_dict(),
            )

            stop, reason = should_stop(
                recall_score=policy.recall_score,
                coverage=coverage.to_dict(),
                calls_made=total_env_calls,
                max_calls=self.max_env_calls,
                tokens_used=len(json.dumps(pack.snapshot())),
                token_budget=self.max_state_chars,
                user_results_count=sum(1 for f in pack.facts if _is_user_owned(f)),
            )
            # --- COVERAGE + STOP DECISION TELEMETRY (loop) ---
            logger.info(
                "RLM_COVERAGE trace_id=%s step=%d stop=%s reason=%s coverage=%s",
                trace_id,
                step,
                stop,
                reason,
                coverage.to_dict(),
            )
            if stop:
                pack.warnings.append(f"stop:{reason}")
                logger.info("RLMController: stop step=%d reason=%s", step, reason)
                break

            if coverage.diminishing_returns:
                pack.warnings.append("stop:diminishing_returns")
                logger.info(
                    "RLMController: stop step=%d reason=diminishing_returns novelty_recent_sum=%d window=%d",
                    step,
                    coverage.novelty_recent_sum,
                    self.novelty_window,
                )
                break

            decision = self._deterministic_decision(pack, coverage)
            if not decision or not decision.actions:
                logger.info("RLMController: no actions at step=%d; stopping", step)
                break
            logger.debug(
                "RLMController: step=%d actions=%s",
                step,
                [a.action for a in decision.actions],
            )

            hard_budget_hit = False
            step_new_facts = 0
            step_new_chunks = 0
            step_graph_expansions = 0
            for action in decision.actions[: self.max_actions_per_step]:
                logger.debug(
                    "RLMController: executing action=%s k=%s",
                    action.action,
                    action.k,
                )
                # --- ACTION EXECUTION TELEMETRY ---
                logger.info(
                    "RLM_ACTION trace_id=%s step=%d action=%s k=%s",
                    trace_id,
                    step,
                    action.action,
                    action.k,
                )
                items = await self._execute_action(
                    user_subject=user_subject,
                    action=action,
                    query_embedding=query_embedding,
                    query_text=pack.query_text,
                    trace_id=trace_id,
                    step=step,
                    owner_type=str(pack.owner_type or owner_type),
                    owner_id=pack.owner_id or owner_id,
                )
                items = self._truncate_items(items)
                # --- ACTION RESULT TELEMETRY ---
                logger.info(
                    "RLM_ACTION_RESULT trace_id=%s step=%d action=%s returned=%d",
                    trace_id,
                    step,
                    action.action,
                    len(items or []),
                )

                if action.action in {
                    "search_semantic",
                    "fetch_more_facts",
                    "fetch_facts",
                }:
                    novelty = pack.compute_novelty(items, "facts")
                    pack.facts = _merge_unique(pack.facts, items, self.max_items_per_type)
                    pack.apply_novelty(items, "facts")
                    step_new_facts += novelty

                elif action.action in {
                    "episodic_clusters",
                    "search_episodic",
                    "fetch_episode_clusters",
                }:
                    pack.episodes = _merge_unique(pack.episodes, items, self.max_items_per_type)
                    pack.apply_novelty(items, "episodes")

                elif action.action in {"graph_neighbors", "expand_graph"}:
                    pack.graph = _merge_unique(pack.graph, items, self.max_items_per_type)
                    pack.apply_novelty(items, "graph")
                    step_graph_expansions += 1

                elif action.action in {"search_chunks", "fetch_chunks"}:
                    novelty = pack.compute_novelty(items, "chunks")
                    pack.chunks = _merge_unique(getattr(pack, "chunks", []), items, self.max_items_per_type)
                    pack.apply_novelty(items, "chunks")
                    step_new_chunks += novelty

                total_env_calls += 1
                if total_env_calls >= self.max_env_calls:
                    pack.warnings.append("stop:max_env_calls")
                    hard_budget_hit = True
                    break

                if self.max_new_facts_per_step and step_new_facts >= self.max_new_facts_per_step:
                    pack.warnings.append("stop:max_new_facts_per_step")
                    logger.info(
                        "RLMController: step=%d hit max_new_facts_per_step=%d",
                        step,
                        self.max_new_facts_per_step,
                    )
                    hard_budget_hit = True
                    break
                if self.max_new_chunks_per_step and step_new_chunks >= self.max_new_chunks_per_step:
                    pack.warnings.append("stop:max_new_chunks_per_step")
                    logger.info(
                        "RLMController: step=%d hit max_new_chunks_per_step=%d",
                        step,
                        self.max_new_chunks_per_step,
                    )
                    hard_budget_hit = True
                    break
                if self.max_graph_expansions_per_step and step_graph_expansions >= self.max_graph_expansions_per_step:
                    pack.warnings.append("stop:max_graph_expansions_per_step")
                    logger.info(
                        "RLMController: step=%d hit max_graph_expansions_per_step=%d",
                        step,
                        self.max_graph_expansions_per_step,
                    )
                    hard_budget_hit = True
                    break

            if hard_budget_hit:
                break
            logger.debug(
                "RLMController: step=%d facts preview=%s",
                step,
                [self._describe_fact(f)[:180] for f in pack.facts[:5]],
            )

        await self._prune_facts_with_llm(pack)
        logger.info(
            "RLMController.retrieve_context: done facts=%d episodes=%d graph=%d warnings=%s",
            len(pack.facts),
            len(pack.episodes),
            len(pack.graph),
            pack.warnings,
        )
        # --- TERMINATION TELEMETRY ---
        logger.info(
            "RLM_END trace_id=%s total_steps=%d total_calls=%d facts=%d episodes=%d graph=%d warnings=%s",
            trace_id,
            len(pack.steps),
            total_env_calls,
            len(pack.facts),
            len(pack.episodes),
            len(pack.graph),
            pack.warnings,
        )
        return pack

    # ------------------------------------------------------------------
    # BASELINE RETRIEVAL
    # ------------------------------------------------------------------

    async def _baseline_retrieval(
        self,
        pack: ContextPack,
        query_embedding: List[float],
        trace_id: str = None,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> None:
        if query_embedding:
            try:
                results = await self._search_semantic_core(
                    user_id=pack.user_id,
                    query_embedding=query_embedding,
                    k=self.max_items_per_type,
                    query_text=pack.query_text,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            except TypeError:
                results = await self._search_semantic_core(
                    user_id=pack.user_id,
                    query_embedding=query_embedding,
                    k=self.max_items_per_type,
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            pack.facts = _merge_unique(
                pack.facts,
                results,
                self.max_items_per_type,
            )
            # Evidence expansion: fetch chunks referenced by fact.source_ids (bounded).
            try:
                max_ev = int(getattr(getattr(self.env, "_memory", None), "retrieval_cfg", None).max_evidence_chunks)
            except Exception:
                max_ev = 6
            max_ev = max(0, max_ev)
            if max_ev and hasattr(self.env, "fetch_chunks"):
                cited: List[str] = []
                for f in pack.facts:
                    src = f.get("source_ids") if isinstance(f, dict) else getattr(f, "source_ids", None)
                    if isinstance(src, list):
                        for sid in src:
                            if sid:
                                cited.append(str(sid))
                cited = list(dict.fromkeys(cited))[:max_ev]
                if cited:
                    chunks_ev = await self.env.fetch_chunks(
                        user_id=pack.user_id,
                        ids=cited,
                        owner_type=owner_type,
                        owner_id=owner_id,
                    )
                    pack.chunks = _merge_unique(
                        getattr(pack, "chunks", []),
                        chunks_ev,
                        self.max_items_per_type,
                    )
            # Optional chunk retrieval via centralized ChunkCore search.
            try:
                chunk_core = getattr(self.env, "_chunk_core", None)
                if chunk_core is not None:
                    lexical_k = int(
                        getattr(getattr(self.env, "_memory", None), "retrieval_cfg", None).lexical_chunks_k
                    )
                else:
                    lexical_k = 0
            except Exception:
                lexical_k = 15
            if chunk_core is not None:
                try:
                    chunks = await chunk_core.search_chunks(
                        query_embedding=list(query_embedding),
                        owner_type=owner_type,
                        owner_id=owner_id,
                        k=self.max_items_per_type,
                        query_text=pack.query_text,
                        lexical_k=lexical_k,
                        filter_terms=bool(pack.query_text),
                    )
                    pack.chunks = _merge_unique(
                        getattr(pack, "chunks", []),
                        chunks,
                        self.max_items_per_type,
                    )
                except Exception:
                    logger.exception("RLMController: chunk_core.search_chunks failed")

        # Procedural baseline (skills)
        try:
            skills = await self._search_procedural_core(
                user_id=pack.user_id,
                query_embedding=query_embedding,
                k=self.max_items_per_type,
                owner_type=owner_type,
                owner_id=owner_id,
            )
            pack.skills = _merge_unique(pack.skills, skills, self.max_items_per_type)
        except Exception:
            logger.exception("RLMController: search_procedural failed")

        pack.episodes = _merge_unique(
            pack.episodes,
            await self.env.episodic_cluster_summaries(
                pack.user_id,
                owner_type=owner_type,
                owner_id=owner_id,
                k=self.cluster_k,
                max_episodes=self.max_items_per_type,
            ),
            self.max_items_per_type,
        )

        if owner_type == "user":
            pack.graph = _merge_unique(
                pack.graph,
                await self.env.graph_neighbors(
                    user_id=pack.user_id,
                    node_id=pack.user_id,
                    predicate_scope=self._next_predicate_scope(pack),
                    depth=1,
                    k=self.max_items_per_type,
                    owner_type=owner_type,
                    owner_id=owner_id,
                ),
                self.max_items_per_type,
            )
        # --- BASELINE RETRIEVAL TELEMETRY ---
        if trace_id is not None:
            logger.info(
                "RLM_BASELINE trace_id=%s facts=%d episodes=%d graph=%d",
                trace_id,
                len(pack.facts),
                len(pack.episodes),
                len(pack.graph),
            )

    # ------------------------------------------------------------------
    # DECISION LOGIC
    # ------------------------------------------------------------------
    async def _search_semantic_core(
        self,
        *,
        user_id: str,
        query_embedding: List[float],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        owner_type: str,
        owner_id: Optional[str],
    ) -> List[Any]:
        """
        Centralized semantic retrieval via SemanticCore (no environment wrapper).
        Resolves owner scope, validates subjects, and passes through filters.
        """
        semantic_core = getattr(self.env, "_semantic_core", None)
        if semantic_core is None:
            return []

        try:
            k = self.env._validate_k("RLMController._search_semantic_core", k)
        except Exception:
            k = max(1, int(k)) if k else 1
        try:
            offset = self.env._safe_offset(filters.get("offset") if isinstance(filters, dict) else None)
        except Exception:
            offset = max(0, int(filters.get("offset", 0))) if isinstance(filters, dict) else 0
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("RLMController._search_semantic_core: invalid subject=%r", user_id)
            return []

        subject: Optional[str] = None
        if isinstance(filters, dict) and filters.get("subject"):
            raw_subject = str(filters.get("subject")).strip()
            if raw_subject.startswith("user:"):
                try:
                    normalized = ensure_user_subject(raw_subject)
                except Exception:
                    normalized = None
                if normalized != user_subject:
                    logger.warning(
                        "RLMController._search_semantic_core: rejected cross-user subject=%r user=%s",
                        raw_subject,
                        user_subject,
                    )
                    subject = None
                else:
                    subject = normalized
            else:
                if raw_subject.startswith(("entity:", "doc:", "agent:")):
                    subject = raw_subject
                else:
                    logger.debug(
                        "RLMController._search_semantic_core: ignoring unknown subject namespace=%r",
                        raw_subject,
                    )
                    subject = None

        retrieval_cfg = getattr(self.env, "_memory", None)
        retrieval_cfg = getattr(retrieval_cfg, "retrieval_cfg", None)
        ctx_cfg = getattr(retrieval_cfg, "context", None) if retrieval_cfg else None
        allowed_topics = getattr(ctx_cfg, "allowed_topics", None) if ctx_cfg else None
        if isinstance(allowed_topics, list):
            allowed_topics = [t for t in allowed_topics if isinstance(t, str) and t.strip()]
        else:
            allowed_topics = None

        if owner_type == "agent":
            resolved_owner_id = owner_id or getattr(self.env, "_agent_id", None)
            if not resolved_owner_id:
                logger.warning("RLMController._search_semantic_core: missing agent_id for agent scope")
                return []
        elif owner_type == "user":
            resolved_owner_id = owner_id or user_subject
        else:
            logger.warning("RLMController._search_semantic_core: invalid owner_type=%r", owner_type)
            return []

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RLMController._search_semantic_core: owner_type=%s owner_id=%s subject=%s k=%d offset=%d",
                owner_type,
                resolved_owner_id,
                subject,
                k,
                offset,
            )
        return await semantic_core.search(
            subject=subject,
            query_embedding=list(query_embedding),
            owner_type=owner_type,
            owner_id=resolved_owner_id,
            k=int(k),
            offset=int(offset),
            filters=filters,
            query_text=query_text,
            allowed_topics=allowed_topics,
        )

    async def _search_episodic_core(
        self,
        *,
        user_id: str,
        query_embedding: List[float],
        k: int,
        time_range: Optional[Dict[str, Any]] = None,
        owner_type: str,
        owner_id: Optional[str],
    ) -> List[Any]:
        """
        Centralized episodic retrieval via EpisodicCore (no environment wrapper).
        Resolves owner scope and applies time_range filtering.
        """
        episodic_core = getattr(self.env, "_episodic_core", None)
        if episodic_core is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("RLMController._search_episodic_core: invalid subject=%r", user_id)
            return []

        try:
            k = self.env._validate_k("RLMController._search_episodic_core", k)
        except Exception:
            k = max(1, int(k)) if k else 1
        try:
            offset = self.env._safe_offset(time_range.get("offset") if isinstance(time_range, dict) else None)
        except Exception:
            offset = max(0, int(time_range.get("offset", 0))) if isinstance(time_range, dict) else 0

        if owner_type == "agent":
            resolved_owner_id = owner_id or getattr(self.env, "_agent_id", None)
            if not resolved_owner_id:
                logger.warning("RLMController._search_episodic_core: missing agent_id for agent scope")
                return []
        else:
            resolved_owner_id = owner_id or user_subject

        episodes = await episodic_core.search(
            user_id=user_subject,
            query_embedding=list(query_embedding),
            owner_type=owner_type,
            owner_id=resolved_owner_id,
            k=int(k),
            offset=int(offset),
        )
        try:
            episodes = self.env._filter_time_range(episodes or [], time_range)
        except Exception:
            pass
        return episodes

    async def _search_procedural_core(
        self,
        *,
        user_id: str,
        query_embedding: List[float],
        k: int,
        owner_type: str,
        owner_id: Optional[str],
    ) -> List[Any]:
        """
        Centralized procedural retrieval via ProceduralCore (no environment wrapper).
        Resolves owner scope and returns raw procedural matches.
        """
        procedural_core = getattr(self.env, "_procedural_core", None)
        if procedural_core is None:
            return []
        try:
            user_subject = ensure_user_subject(user_id)
        except Exception:
            logger.exception("RLMController._search_procedural_core: invalid subject=%r", user_id)
            return []

        try:
            k = self.env._validate_k("RLMController._search_procedural_core", k)
        except Exception:
            k = max(1, int(k)) if k else 1

        if owner_type == "agent":
            resolved_owner_id = owner_id or getattr(self.env, "_agent_id", None)
            if not resolved_owner_id:
                logger.warning("RLMController._search_procedural_core: missing agent_id for agent scope")
                return []
        else:
            resolved_owner_id = owner_id or user_subject

        return await procedural_core.search(
            user_id=user_subject,
            query_embedding=list(query_embedding),
            owner_type=owner_type,
            owner_id=resolved_owner_id,
            k=int(k),
        )

    def _assess_coverage(self, pack: ContextPack):
        return assess_coverage(
            facts=pack.facts,
            episodes=pack.episodes,
            graph=pack.graph,
            salience_threshold=self.salience_threshold,
            min_semantic_facts=self.min_semantic_facts,
            min_high_salience_facts=self.min_high_salience_facts,
            min_cluster_summaries=self.min_cluster_summaries,
            require_semantic=True,
            prefer_clusters=True,
            novelty_history=pack.novelty_history,
            novelty_window=self.novelty_window,
            min_recent_novelty=self.min_recent_novelty,
        )

    def _deterministic_decision(self, pack: ContextPack, coverage) -> Optional[ControllerDecision]:
        actions: List[RetrievalAction] = []

        if coverage.needs_semantic:
            if pack.facts:
                preds = self._next_predicate_scope(pack)
                if preds:
                    predicate = preds[0]
                    offset = pack.get_predicate_offset(predicate)
                    actions.append(
                        RetrievalAction(
                            action="fetch_more_facts",
                            predicate=predicate,
                            k=self.max_items_per_type,
                            filters={"offset": offset},
                            owner_type=pack.owner_type,
                        )
                    )
                    pack.bump_predicate_offset(predicate, self.max_items_per_type)
            else:
                filters = {"subject": pack.user_id} if pack.owner_type == "user" else None
                actions.append(
                    RetrievalAction(
                        action="search_semantic",
                        k=self.max_items_per_type,
                        filters=filters,
                        owner_type=pack.owner_type,
                    )
                )

        if coverage.needs_clusters:
            has_cluster = any(
                isinstance(ep, dict) and "episode_ids" in ep for ep in (pack.episodes or [])
            )
            actions.append(
                RetrievalAction(
                    action="fetch_episode_clusters",
                    k=self.cluster_k,
                    time_range=None,
                    min_salience=self.salience_threshold,
                    owner_type=pack.owner_type,
                )
            )
            if not has_cluster and len(pack.steps) >= 2:
                actions.append(
                    RetrievalAction(
                        action="search_episodic",
                        k=self.max_items_per_type,
                        owner_type=pack.owner_type,
                    )
                )

        # Graph is navigation, not truth: only expand as a last-mile step.
        if not pack.graph and pack.facts and not coverage.needs_semantic and not coverage.needs_clusters:
            predicate_scope = self._next_predicate_scope(pack)
            if predicate_scope and pack.owner_type == "user":
                actions.append(
                    RetrievalAction(
                        action="expand_graph",
                        subject=pack.user_id,
                        predicate=predicate_scope[0],
                        hops=1,
                        direction="outbound",
                        k=min(self.max_items_per_type, 20),
                        owner_type=pack.owner_type,
                    )
                )

        return ControllerDecision(actions=actions) if actions else None


    # ------------------------------------------------------------------
    # ACTION EXECUTION
    # ------------------------------------------------------------------

    async def _execute_action(
        self,
        *,
        user_subject: str,
        action: RetrievalAction,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        trace_id: str = None,
        step: int = None,
        owner_type: str,
        owner_id: Optional[str],
    ) -> List[Any]:

        k = int(action.k) if action.k else self.max_items_per_type
        lane_owner_type = action.owner_type or owner_type
        lane_owner_id = owner_id
        if lane_owner_type == "agent":
            lane_owner_id = getattr(self.env, "_agent_id", None) if lane_owner_id is None else lane_owner_id

        if action.action == "search_semantic":
            results = await self._search_semantic_core(
                user_id=user_subject,
                query_embedding=query_embedding,
                k=k,
                filters=action.filters,
                query_text=query_text,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: search_semantic returned %d", len(results or []))
            return results

        if action.action == "search_episodic":
            results = await self._search_episodic_core(
                user_id=user_subject,
                query_embedding=query_embedding,
                k=k,
                time_range=action.time_range,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: search_episodic returned %d", len(results or []))
            return results

        if action.action == "search_procedural":
            results = await self._search_procedural_core(
                user_id=user_subject,
                query_embedding=query_embedding,
                k=k,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: search_procedural returned %d", len(results or []))
            return results

        if action.action == "fetch_more_facts":
            offset = int(action.filters.get("offset", 0)) if action.filters else 0
            # --- FETCH_MORE_FACTS-SPECIFIC TELEMETRY ---
            if trace_id is not None:
                logger.info(
                    "RLM_FETCH_MORE_FACTS trace_id=%s predicate=%s offset=%s k=%s",
                    trace_id,
                    action.predicate,
                    offset,
                    k,
                )
            results = await self.env.fetch_more_facts(
                user_id=user_subject,
                predicate=action.predicate,
                k=k,
                offset=offset,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: fetch_more_facts returned %d", len(results or []))
            return results

        if action.action == "fetch_facts":
            results = await self.env.fetch_facts_by_ids(
                user_id=user_subject,
                ids=action.ids or [],
            )
            logger.debug("RLMController: fetch_facts returned %d", len(results or []))
            return results

        if action.action == "fetch_chunks":
            results = await self.env.fetch_chunks(
                user_id=user_subject,
                ids=action.ids or [],
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: fetch_chunks returned %d", len(results or []))
            return results

        if action.action == "episodic_clusters":
            results = await self.env.episodic_cluster_summaries(
                user_id=user_subject,
                k=k,
                max_episodes=self.max_items_per_type,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: episodic_clusters returned %d", len(results or []))
            return results

        if action.action == "fetch_episode_clusters":
            results = await self.env.fetch_episode_clusters(
                user_id=user_subject,
                k=k,
                max_episodes=self.max_items_per_type,
                time_range=action.time_range,
                min_salience=action.min_salience,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: fetch_episode_clusters returned %d", len(results or []))
            return results

        if action.action == "graph_neighbors":
            results = await self.env.graph_neighbors(
                user_id=user_subject,
                node_id=action.node_id,
                predicate_scope=action.predicate_scope,
                depth=int(action.depth or 1),
                k=k,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: graph_neighbors returned %d", len(results or []))
            return results

        if action.action == "expand_graph":
            results = await self.env.expand_graph(
                user_id=user_subject,
                subject=action.subject,
                predicate=action.predicate,
                hops=int(action.hops or 1),
                direction=action.direction,
                k=k,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: expand_graph returned %d", len(results or []))
            return results

        return []

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _next_predicate_scope(self, pack: ContextPack) -> List[str]:
        ordered: List[str] = []

        if self.predicate_weights:
            for p, _ in sorted(self.predicate_weights.items(), key=lambda kv: (-kv[1], kv[0])):
                ordered.append(p)

        counts = {}
        for f in pack.facts:
            pred = f.get("predicate") if isinstance(f, dict) else None
            if pred:
                counts[pred.upper()] = counts.get(pred.upper(), 0) + 1

        for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if p not in ordered:
                ordered.append(p)

        if not ordered:
            ordered = ["RELATED_TO"]

        return ordered[: self.graph_predicate_limit]

    def _normalize_predicate_weights(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not weights:
            return {}
        return {str(k).upper(): float(v) for k, v in weights.items() if isinstance(v, (int, float))}

    def _truncate_items(self, items: List[Any]) -> List[Any]:
        if not items:
            return items
        return [
            {k: (v[:2000] if isinstance(v, str) else v) for k, v in it.items()}
            if isinstance(it, dict)
            else it
            for it in items
        ]

    async def _prune_facts_with_llm(self, pack: ContextPack) -> None:
        if not self.llm or not pack.facts:
            logger.debug("RLMController: prune skipped (llm=%s facts=%d)", bool(self.llm), len(pack.facts))
            return

        logger.info("RLMController: pruning facts with LLM (count=%d)", len(pack.facts))
        descriptions = []
        for idx, fact in enumerate(pack.facts, start=1):
            desc = self._describe_fact(fact)
            descriptions.append(f"{idx}. {desc}")

        if not descriptions:
            logger.debug("RLMController: prune skipped (no descriptions)")
            return
        logger.debug("RLMController: prune input facts=%s", descriptions)

        system_prompt = (
            "You are a retrieval assistant that filters facts. Given a question and "
            "a numbered list of facts, return ONLY valid JSON with a key `keep` whose value "
            "is an array of 1-based indices of the facts that directly answer the question. "
            "Return an empty array if none apply."
        )
        user_prompt = f"Question: {pack.query_text}\nFacts:\n" + "\n".join(descriptions)

        try:
            response = await self.llm.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=128,
                temperature=0.0,
            )
        except Exception:
            logger.exception("RLMController: final pruning request failed.")
            return

        logger.debug("RLMController: prune response=%r", response)

        selected = self._parse_keep_list(response)
        if not selected:
            logger.info("RLMController: prune response empty; applying fallback filter")
            selected = self._fallback_keep_by_query(pack.query_text, pack.facts)
            if not selected:
                logger.info("RLMController: prune fallback kept 0 facts")
                return

        filtered = []
        for idx in sorted(selected):
            if 1 <= idx <= len(pack.facts):
                filtered.append(pack.facts[idx - 1])
        if filtered:
            pack.facts = filtered
            logger.info("RLMController: prune kept %d facts", len(pack.facts))

    def _describe_fact(self, fact: Any) -> str:
        meta = get_attr_or_key(fact, "meta", {})
        if isinstance(meta, dict):
            excerpt = meta.get("excerpt") or meta.get("description")
            if excerpt:
                return str(excerpt).replace("\n", " ").strip()
        obj = get_attr_or_key(fact, "object")
        if isinstance(obj, dict):
            text = obj.get("text") or ""
        else:
            text = str(obj)
        if text and text.strip():
            return text.strip()
        sub = get_attr_or_key(fact, "subject", "user")
        pred = get_attr_or_key(fact, "predicate", "related_to")
        return f"{sub} {pred}"

    def _parse_keep_list(self, response: str) -> List[int]:
        response = response.strip()
        values: List[int] = []
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and isinstance(parsed.get("keep"), list):
                values = [int(x) for x in parsed["keep"] if isinstance(x, int)]
            elif isinstance(parsed, list):
                values = [int(x) for x in parsed if isinstance(x, int)]
        except Exception:
            tokens = response.replace(",", " ").split()
            for tok in tokens:
                if tok.isdigit():
                    values.append(int(tok))
        return sorted(set(values))

    def _fallback_keep_by_query(self, query: str, facts: List[Any]) -> List[int]:
        """
        Heuristic fallback if LLM pruning fails: keep facts that mention any query term.
        """
        stop = {
            "what", "do", "you", "know", "about", "the", "a", "an", "of", "to",
            "and", "or", "is", "are", "in", "on", "for", "with", "that", "this",
        }
        terms = []
        if extract_query_terms:
            try:
                terms = [t for t in (extract_query_terms(query) or []) if t and t not in stop]
            except Exception:
                terms = []
        if not terms:
            terms = [
                t for t in re.split(r"\W+", query.lower())
                if t and len(t) >= 4 and t not in stop
            ]
        if not terms:
            return []
        logger.debug("RLMController: prune fallback terms=%s", terms)
        kept = []
        for idx, fact in enumerate(facts, start=1):
            text = self._describe_fact(fact).lower()
            if any(t in text for t in terms):
                kept.append(idx)
        return kept


def _merge_unique(existing: List[Any], research: List[Any], limit: int) -> List[Any]:
    from uma.core.utils.dedupe import dedupe_by_id

    merged = dedupe_by_id(list(existing or []) + list(research or []))
    if limit <= 0:
        return []
    return merged[:limit]


def _get_owner_type(item: Any) -> str:
    if isinstance(item, dict):
        return (item.get("owner_type") or "").lower()
    return (getattr(item, "owner_type", None) or "").lower()


def _is_user_owned(item: Any) -> bool:
    return _get_owner_type(item) == "user"
