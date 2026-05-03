# uma/retrieve/rlm/controller.py

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .context_pack import ContextPack
from .decisions import RetrievalAction
from . import decisions
from .intent import QueryIntent, classify_query_intent
from .domain import (
    PREFERENCE_PREDICATES,
    ensure_domains_for_chunks,
    ensure_domains_for_facts,
    ensure_domains_for_skills,
    filter_facts_by_domains,
)
from .coverage import assess_coverage, compute_confidence
from ..policy import RetrievalPolicy, should_stop
from uma.memory.chunk.core import merge_chunks_with_precedence, partition_chunks_by_route
from uma.memory.semantic.query_pruner import prune_facts_for_query
from .evidence import expand_evidence_chunks_from_facts

from ..ranking import Ranker
from .request import RetrievalRequest, RetrievalScope
from uma.memory.working_memory.core import session_scope_from_runtime_context

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
        debug_scores = False
        try:
            memory = getattr(env, "_memory", None)
            retrieval_cfg = getattr(memory, "retrieval_cfg", None)
            rlm_cfg = getattr(retrieval_cfg, "rlm", None) if retrieval_cfg else None
            debug_scores = bool(getattr(retrieval_cfg, "debug_scores", False)) if retrieval_cfg else False
        except Exception:
            rlm_cfg = None
            debug_scores = False

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
        self.predicate_allowlist = getattr(rlm_cfg, "predicate_allowlist", None)
        self.ranker = Ranker(debug_scores=debug_scores)

        self.novelty_window = max(1, int(getattr(rlm_cfg, "novelty_window", 2)))
        self.min_recent_novelty = max(0, int(getattr(rlm_cfg, "min_recent_novelty", 1)))

        self.max_new_facts_per_step = max(0, int(getattr(rlm_cfg, "max_new_facts_per_step", 12)))
        self.max_new_chunks_per_step = max(0, int(getattr(rlm_cfg, "max_new_chunks_per_step", 8)))
        self.max_graph_expansions_per_step = max(0, int(getattr(rlm_cfg, "max_graph_expansions_per_step", 1)))
        self.chunk_fallback_enabled = bool(getattr(rlm_cfg, "chunk_fallback_enabled", True))
        self.chunk_fallback_k_multiplier = max(1, int(getattr(rlm_cfg, "chunk_fallback_k_multiplier", 2)))

        self.max_state_chars = max(200, int(getattr(rlm_cfg, "max_state_chars", 1200)))
        self.test_mode = bool(getattr(rlm_cfg, "test_mode", False))
        self.semantic_first = bool(getattr(rlm_cfg, "semantic_first", True))
        self.clusters_first = bool(getattr(rlm_cfg, "clusters_first", True))

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def retrieve_context(self, request: RetrievalRequest, query_text: str) -> ContextPack:
        if not isinstance(request, RetrievalRequest):
            logger.error("RLMController.retrieve_context: request must be a RetrievalRequest")
            raise TypeError("request must be a RetrievalRequest")
        if not query_text or not isinstance(query_text, str):
            logger.error("RLMController.retrieve_context: query_text must be a non-empty string")
            raise ValueError("query_text must be a non-empty string")

        policy = RetrievalPolicy(query_text)
        intent = classify_query_intent(query_text)
        logger.info(
            "RLMController.retrieve_context: start user_id=%s query=%r",
            request.normalized_user_id,
            query_text,
        )
        start = time.time()
        normalized_user_id = request.normalized_user_id

        # --- TRACE CONTEXT ---
        trace_root = request.trace_id or request.context.request_id or normalized_user_id
        trace_id = f"rlm:{trace_root}:{int(time.time()*1000)}"
        logger.info(
            "RLM_START trace_id=%s user=%s recall_query=%s",
            trace_id,
            normalized_user_id,
            any(k in query_text.lower() for k in ["remember", "recall", "previous", "earlier", "last time"]),
        )
        logger.info("RLM_INTENT trace_id=%s intent=%s", trace_id, intent.value)

        plan = getattr(request, "plan", None)
        if plan is None:
            raise ValueError("RLMController.retrieve_context requires request.plan")
        active_lanes = list(plan.participating_lanes)
        active_domains = list(plan.active_domains)
        pack = ContextPack(
            user_id=normalized_user_id,
            query_text=query_text,
            owner_type=None,
            owner_id=None,
            agent_id=request.context.agent_id,
            intent=intent.value,
            active_lanes=list(active_lanes),
            active_domains=list(active_domains),
            lane_plan=plan.to_trace(),
        )
        pack.steps.append(
            {
                "step": 0,
                "phase": "plan",
                **plan.to_trace(),
            }
        )

        pack.working_memory = []
        if hasattr(self.env, "_memory"):
            wm = getattr(getattr(self.env, "_memory", None), "working_memory", None)
            if wm is not None:
                session_scope = session_scope_from_runtime_context(request.context)
                if session_scope is not None:
                    pack.working_memory = wm.get_context(session_scope)
        query_embedding = await self.env.get_query_embedding(query_text)

        # Owner scopes and lane participation are separate. Recall narrows owner
        # scopes; the retrieval plan still controls which canonical lanes run.
        is_recall = policy.recall_score >= 0.75

        if is_recall:
            scopes = list(request.scopes_for_owner_type("user"))
        else:
            scopes = list(request.scopes)
        if not scopes:
            logger.error("RLMController.retrieve_context: no retrieval scopes available")
            raise ValueError("RLMController.retrieve_context: no retrieval scopes available")

        # Keep these fields for telemetry/back-compat; primary execution uses `scopes`.
        pack.owner_type, pack.owner_id = scopes[0].owner_type, scopes[0].owner_id
        logger.info("RLM_LANE scopes=%s", [(scope.owner_type, scope.owner_id) for scope in scopes])
        logger.info(
            "RLM_PLAN trace_id=%s product=%s lanes=%s excluded=%s",
            trace_id,
            getattr(plan, "product", "context"),
            active_lanes,
            [dict(item) for item in getattr(plan, "excluded_lanes", ()) or ()],
        )
        logger.info("RLM_DOMAINS trace_id=%s active_domains=%s", trace_id, pack.active_domains)

        # Baseline retrieval per-scope, then merge into the pack.
        for idx, scope in enumerate(scopes):
            await self._baseline_retrieval(
                request=request,
                pack=pack,
                query_embedding=query_embedding,
                trace_id=f"{trace_id}:{idx}",
                owner_type=scope.owner_type,
                owner_id=scope.owner_id,
            )
            if idx == 0:
                # Preserve first-scope telemetry for tests and downstream expectations.
                pack.owner_type, pack.owner_id = scope.owner_type, scope.owner_id

        # Deterministic merge cleanup after multi-scope baseline.
        try:
            from uma.common.dedupe import dedupe_by_id as _dedupe_by_id
        except Exception:
            _dedupe_by_id = None

        if _dedupe_by_id:
            try:
                pack.facts = _dedupe_by_id(getattr(pack, "facts", []) or [])
            except Exception:
                pass
            try:
                pack.chunks = _dedupe_by_id(getattr(pack, "chunks", []) or [])
            except Exception:
                pass
            try:
                pack.episodes = _dedupe_by_id(getattr(pack, "episodes", []) or [])
            except Exception:
                pass
            try:
                pack.graph = _dedupe_by_id(getattr(pack, "graph", []) or [])
            except Exception:
                pass

        # Phase 0: default domain when missing (metadata-only; no migrations).
        try:
            ensure_domains_for_facts(getattr(pack, "facts", []) or [])
        except Exception:
            pass
        try:
            ensure_domains_for_chunks(getattr(pack, "chunks", []) or [])
        except Exception:
            pass
        try:
            ensure_domains_for_skills(getattr(pack, "skills", []) or [])
        except Exception:
            pass

        # Apply domain routing: filter out disallowed fact domains for this query intent.
        allowed_fact_domains = set(pack.active_domains or [])
        try:
            pack.facts = filter_facts_by_domains(getattr(pack, "facts", []) or [], allowed_fact_domains)
        except Exception:
            pass


        # Tighten evidence expansion: prune facts before expanding cited chunks.
        await self._prune_facts_with_llm(pack)
        await self._expand_evidence_chunks_from_facts(
            request=request,
            pack=pack,
        )
        self._rebuild_chunk_buckets(pack)
        pack.record_seen()
        logger.debug(
            "RLMController: baseline counts facts=%d chunks=%d episodes=%d graph=%d",
            len(pack.facts),
            len(getattr(pack, "chunks", [])),
            len(pack.episodes),
            len(pack.graph),
        )
        try:
            from uma.memory.semantic.query_pruner import describe_fact as _describe_fact
        except Exception:
            _describe_fact = None
        if _describe_fact:
            logger.debug(
                "RLMController: step=0 facts preview=%s",
                [_describe_fact(f)[:180] for f in pack.facts[:5]],
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

        cov = coverage.to_dict()
        try:
            cov["confidence"] = float(compute_confidence(coverage).get("score", 0.0))
        except Exception:
            cov["confidence"] = 0.0
        stop, reason = should_stop(
            recall_score=policy.recall_score,
            coverage=cov,
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
            cov,
        )
        if stop:
            pack.warnings.append(f"stop:{reason}")
            logger.info("RLMController: stop after baseline reason=%s", reason)
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

            cov = coverage.to_dict()
            try:
                cov["confidence"] = float(compute_confidence(coverage).get("score", 0.0))
            except Exception:
                cov["confidence"] = 0.0
            stop, reason = should_stop(
                recall_score=policy.recall_score,
                coverage=cov,
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
                cov,
            )
            if stop:
                pack.warnings.append(f"stop:{reason}")
                logger.info("RLMController: stop step=%d reason=%s", step, reason)
                break

            if coverage.diminishing_returns:
                if _two_distinct_zero_yield_lanes(getattr(pack, "steps", []) or []):
                    pack.warnings.append("stop:diminishing_returns")
                    logger.info(
                        "RLMController: stop step=%d reason=diminishing_returns novelty_recent_sum=%d window=%d",
                        step,
                        coverage.novelty_recent_sum,
                        self.novelty_window,
                    )
                    break
                logger.info(
                    "RLMController: diminishing_returns at step=%d but continuing (fallback ladder not exhausted)",
                    step,
                )

            decision = decisions.deterministic_decision(
                pack,
                coverage,
                cfg={
                    "trace_id": trace_id,
                    "max_items_per_type": self.max_items_per_type,
                    "cluster_k": self.cluster_k,
                    "salience_threshold": self.salience_threshold,
                    "graph_predicate_limit": self.graph_predicate_limit,
                    "chunk_fallback_enabled": self.chunk_fallback_enabled,
                    "chunk_fallback_k_multiplier": self.chunk_fallback_k_multiplier,
                    "predicate_allowlist": self.predicate_allowlist,
                    "next_predicate_scope": lambda p, limit: _filter_predicates_for_domains(
                        decisions.next_predicate_scope(
                            facts=getattr(p, "facts", []) or [],
                            predicate_weights=getattr(self, "predicate_weights", None),
                            graph_predicate_limit=getattr(self, "graph_predicate_limit", 2),
                        ),
                        active_domains=set(getattr(p, "active_domains", []) or []),
                    ),
                },
            )
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
                # Lane plan is authoritative; domain routing remains subordinate.
                active = set(getattr(pack, "active_domains", []) or [])
                if action.action in {"search_chunks", "fetch_chunks"} and not self._lane_active(pack, "raw"):
                    logger.debug("RLMController: skipping %s (raw lane not active)", action.action)
                    continue
                if action.action in {"search_semantic", "fetch_more_facts", "fetch_facts"} and not self._semantic_lanes_active(pack):
                    logger.debug("RLMController: skipping %s (semantic/profile lanes not active)", action.action)
                    continue
                if action.action in {"episodic_clusters", "search_episodic", "fetch_episode_clusters"} and not self._lane_active(pack, "episodic"):
                    logger.debug("RLMController: skipping %s (episodic lane not active)", action.action)
                    continue
                if action.action in {"search_procedural"} and not self._lane_active(pack, "procedural"):
                    logger.debug("RLMController: skipping %s (procedural lane not active)", action.action)
                    continue
                if action.action in {"graph_neighbors", "expand_graph"} and not (
                    ("kb_doc" in active) or ("user_profile" in active)
                ):
                    logger.debug("RLMController: skipping %s (no graph domain active)", action.action)
                    continue
                if action.action in {"graph_neighbors", "expand_graph"} and getattr(getattr(self.env, "_memory", None), "graph_core", None) is None:
                    logger.debug("RLMController: skipping %s (graph_core not available)", action.action)
                    continue
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
                action_start = time.time()
                if action.action == "search_chunks":
                    logger.debug(
                        "RLMController: dispatching search_chunks step=%d scopes=%s k=%s",
                        step,
                        [(scope.owner_type, scope.owner_id) for scope in self._scopes_for_action(scopes, action)],
                        action.k,
                    )
                scope_results: List[Any] = []
                action_scopes = self._scopes_for_action(scopes, action)
                for scope in action_scopes:
                    scope_items = await self.env.execute_action(
                        request=request,
                        action=action,
                        query_embedding=list(query_embedding),
                        query_text=pack.query_text,
                        owner_type=scope.owner_type,
                        owner_id=scope.owner_id,
                        default_k=self.max_items_per_type,
                        trace_id=trace_id,
                    )
                    scope_results.extend(list(scope_items or []))
                items = scope_results
                a = str(getattr(action, "action", "") or "")
                if a in {"search_semantic", "fetch_more_facts", "fetch_facts"}:
                    items = self.ranker.rank_facts(items or [], query_text=pack.query_text)
                elif a in {"fetch_chunks", "search_chunks"}:
                    items = self.ranker.rank_chunks(items or [], query_text=pack.query_text)
                elif a in {"episodic_clusters", "search_episodic", "fetch_episode_clusters"}:
                    items = self.ranker.rank_episodes(items or [], query_text=pack.query_text)
                elif a in {"search_procedural"}:
                    items = self.ranker.rank_skills(items or [], query_text=pack.query_text)
                elapsed_ms = int((time.time() - action_start) * 1000)
                items = self._truncate_items(items)
                if action.action in {"search_semantic", "fetch_more_facts", "fetch_facts"}:
                    try:
                        items = filter_facts_by_domains(list(items or []), set(getattr(pack, "active_domains", []) or []))
                        items = self._filter_items_by_active_lanes(list(items or []), pack)
                    except Exception:
                        pass
                if action.action in {"search_chunks", "fetch_chunks"}:
                    try:
                        ensure_domains_for_chunks(list(items or []))
                        items = self._filter_items_by_active_lanes(list(items or []), pack)
                    except Exception:
                        pass
                if action.action in {"search_procedural"}:
                    try:
                        ensure_domains_for_skills(list(items or []))
                        items = self._filter_items_by_active_lanes(list(items or []), pack)
                    except Exception:
                        pass
                store = _store_for_action(action.action)
                returned = len(items or [])
                novelty = 0
                if store:
                    try:
                        novelty = pack.compute_novelty(items, store)
                    except Exception:
                        novelty = 0
                # --- ACTION RESULT TELEMETRY ---
                logger.info(
                    "RLM_ACTION_RESULT trace_id=%s step=%d action=%s k=%s predicate=%s offset=%s owner=%s:%s action_owner_type=%s result_count=%d novelty=%d elapsed_ms=%d",
                    trace_id,
                    step,
                    action.action,
                    action.k,
                    getattr(action, "predicate", None),
                    (action.filters or {}).get("offset") if getattr(action, "filters", None) else None,
                    str(getattr(pack, "owner_type", None) or scopes[0].owner_type),
                    getattr(pack, "owner_id", None) or scopes[0].owner_id,
                    getattr(action, "owner_type", None),
                    returned,
                    novelty,
                    elapsed_ms,
                )

                if action.action in {
                    "search_semantic",
                    "fetch_more_facts",
                    "fetch_facts",
                }:
                    pack.facts = _merge_unique(pack.facts, items, self.max_items_per_type)
                    pack.facts = self._dedupe_facts_by_signature(pack.facts)
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
                    pack.chunks = _merge_unique(getattr(pack, "chunks", []), items, self.max_items_per_type)
                    pack.apply_novelty(items, "chunks")
                    step_new_chunks += novelty
                    self._rebuild_chunk_buckets(pack)

                elif action.action in {"search_procedural"}:
                    pack.skills = _merge_unique(getattr(pack, "skills", []), items, self.max_items_per_type)
                    pack.apply_novelty(items, "skills")

                if store:
                    pack.steps.append(
                        {
                            "step": step,
                            "phase": "loop",
                            "event": "action_result",
                            "action": action.action,
                            "store": store,
                            "returned": returned,
                            "novelty": novelty,
                            "predicate": getattr(action, "predicate", None),
                            "subject": getattr(action, "subject", None),
                            "node_id": getattr(action, "node_id", None),
                            "filters": getattr(action, "filters", None),
                            "intent": getattr(pack, "intent", None),
                            "active_domains": list(getattr(pack, "active_domains", []) or []),
                            "lane_scopes": [(scope.owner_type, scope.owner_id) for scope in action_scopes],
                            "lane_owner_type": str(getattr(pack, "owner_type", None) or scopes[0].owner_type),
                            "lane_owner_id": getattr(pack, "owner_id", None) or scopes[0].owner_id,
                            "action_owner_type": getattr(action, "owner_type", None),
                        }
                    )

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
            try:
                from uma.memory.semantic.query_pruner import describe_fact as _describe_fact
            except Exception:
                _describe_fact = None
            if _describe_fact:
                logger.debug(
                    "RLMController: step=%d facts preview=%s",
                    step,
                    [_describe_fact(f)[:180] for f in pack.facts[:5]],
                )

        # Facts are already pruned post-baseline; prune again to account for any
        # new facts added during the loop, then expand evidence based on the final set.
        await self._prune_facts_with_llm(pack)
        await self._expand_evidence_chunks_from_facts(
            request=request,
            pack=pack,
        )
        self._rebuild_chunk_buckets(pack)
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
        request: RetrievalRequest,
        pack: ContextPack,
        query_embedding: List[float],
        trace_id: str = None,
        owner_type: str = "agent",
        owner_id: Optional[str] = None,
    ) -> None:
        start_facts = len(pack.facts)
        start_chunks = len(getattr(pack, "chunks", []))
        start_episodes = len(pack.episodes)
        start_graph = len(pack.graph)
        if query_embedding:
            if self._semantic_lanes_active(pack):
                results = await self.env.execute_action(
                    request=request,
                    action=RetrievalAction(action="search_semantic", k=self.max_items_per_type, reason="baseline"),
                    query_embedding=list(query_embedding),
                    query_text=pack.query_text,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    default_k=self.max_items_per_type,
                    trace_id=trace_id,
                )
                results = self.ranker.rank_facts(results or [], query_text=pack.query_text)
                # Ensure + filter by active domains (defaults domain for older data).
                try:
                    ensure_domains_for_facts(list(results or []))
                    results = filter_facts_by_domains(list(results or []), set(getattr(pack, "active_domains", []) or []))
                    results = self._filter_items_by_active_lanes(list(results or []), pack)
                except Exception:
                    pass
                pack.facts = _merge_unique(
                    pack.facts,
                    results,
                    self.max_items_per_type,
                )
                pack.facts = self._dedupe_facts_by_signature(pack.facts)
                logger.debug(
                    "RLMController._baseline_retrieval: semantic facts=%d owner=%s:%s",
                    len(pack.facts),
                    owner_type,
                    owner_id,
                )
            else:
                logger.debug(
                    "RLMController._baseline_retrieval: semantic/profile lanes not active; skipping fact search"
                )
            if self._lane_active(pack, "raw"):
                chunks = await self.env.execute_action(
                    request=request,
                    action=RetrievalAction(action="search_chunks", k=self.max_items_per_type, reason="baseline"),
                    query_embedding=list(query_embedding),
                    query_text=pack.query_text,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    default_k=self.max_items_per_type,
                    trace_id=trace_id,
                )
                chunks = self.ranker.rank_chunks(chunks or [], query_text=pack.query_text)
                try:
                    ensure_domains_for_chunks(list(chunks or []))
                    chunks = self._filter_items_by_active_lanes(list(chunks or []), pack)
                except Exception:
                    pass
                pack.chunks = _merge_unique(
                    getattr(pack, "chunks", []),
                    chunks,
                    self.max_items_per_type,
                )
                self._rebuild_chunk_buckets(pack)
                logger.debug(
                    "RLMController._baseline_retrieval: chunk search returned=%d merged_chunks=%d",
                    len(chunks or []),
                    len(getattr(pack, "chunks", [])),
                )
            else:
                logger.debug("RLMController._baseline_retrieval: raw lane not active; skipping chunk search")

        # Procedural baseline (skills)
        if not self._lane_active(pack, "procedural"):
            skills = []
        else:
            skills = await self.env.execute_action(
                request=request,
                action=RetrievalAction(action="search_procedural", k=self.max_items_per_type, reason="baseline"),
                query_embedding=list(query_embedding),
                query_text=pack.query_text,
                owner_type=owner_type,
                owner_id=owner_id,
                default_k=self.max_items_per_type,
                trace_id=trace_id,
            )
        skills = self.ranker.rank_skills(skills or [], query_text=pack.query_text)
        try:
            ensure_domains_for_skills(list(skills or []))
            skills = self._filter_items_by_active_lanes(list(skills or []), pack)
        except Exception:
            pass
        pack.skills = _merge_unique(pack.skills, skills, self.max_items_per_type)

        episodes = []
        if self._lane_active(pack, "episodic"):
            episodes = await self.env.execute_action(
                request=request,
                action=RetrievalAction(action="episodic_clusters", k=self.cluster_k, reason="baseline"),
                query_embedding=list(query_embedding),
                query_text=pack.query_text,
                owner_type=owner_type,
                owner_id=owner_id,
                default_k=self.max_items_per_type,
                trace_id=trace_id,
            )
        episodes = self.ranker.rank_episodes(episodes or [], query_text=pack.query_text)
        episodes = self._filter_items_by_active_lanes(list(episodes or []), pack)
        pack.episodes = _merge_unique(
            pack.episodes,
            episodes,
            self.max_items_per_type,
        )

        # User-profile graph baseline: only for PERSONAL intent.
        if (
            owner_type == "user"
            and getattr(pack, "intent", None) == "personal"
            and "user_profile" in set(getattr(pack, "active_domains", []) or [])
            and getattr(getattr(self.env, "_memory", None), "graph_core", None) is not None
        ):
            pack.graph = _merge_unique(
                pack.graph,
                await self.env.graph_neighbors(
                    request=request,
                    node_id=pack.user_id,
                    predicate_scope=_filter_predicates_for_domains(
                        decisions.next_predicate_scope(
                            facts=getattr(pack, "facts", []) or [],
                            predicate_weights=getattr(self, "predicate_weights", None),
                            graph_predicate_limit=getattr(self, "graph_predicate_limit", 2),
                        ),
                        active_domains=set(getattr(pack, "active_domains", []) or []),
                    ),
                    domain_scope=["user_profile"],
                    depth=1,
                    k=self.max_items_per_type,
                    owner_type=owner_type,
                    owner_id=owner_id,
                ),
                self.max_items_per_type,
            )
        logger.debug(
            "RLMController._baseline_retrieval: scope owner=%s:%s facts=%d chunks=%d episodes=%d graph=%d",
            owner_type,
            owner_id,
            max(0, len(pack.facts) - start_facts),
            max(0, len(getattr(pack, "chunks", [])) - start_chunks),
            max(0, len(pack.episodes) - start_episodes),
            max(0, len(pack.graph) - start_graph),
        )
        # --- BASELINE RETRIEVAL TELEMETRY ---
        if trace_id is not None:
            logger.info(
                "RLM_BASELINE trace_id=%s facts=%d chunks=%d episodes=%d graph=%d",
                trace_id,
                len(pack.facts),
                len(getattr(pack, "chunks", [])),
                len(pack.episodes),
                len(pack.graph),
            )


    def _filter_predicates_for_domains(predicates: List[str], *, active_domains: set[str]) -> List[str]:
        """
        Deterministically filter predicate candidates based on active domains.

        Phase 0/1 scope: prevent user_profile predicates from entering topical scope.
        """
        preds = [str(p).upper() for p in (predicates or []) if p]
        if "user_profile" not in (active_domains or set()):
            preds = [p for p in preds if p not in PREFERENCE_PREDICATES]
        return preds

    # Retrieval execution is centralized in UMAMemoryEnvironment.execute_action.

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

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _normalize_predicate_weights(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not weights:
            return {}
        return {str(k).upper(): float(v) for k, v in weights.items() if isinstance(v, (int, float))}

    @staticmethod
    def _lane_active(pack: ContextPack, lane: str) -> bool:
        active_lanes = set(getattr(pack, "active_lanes", []) or [])
        return str(lane or "").strip().lower() in active_lanes

    @classmethod
    def _semantic_lanes_active(cls, pack: ContextPack) -> bool:
        return cls._lane_active(pack, "semantic") or cls._lane_active(pack, "profile")

    @classmethod
    def _filter_items_by_active_lanes(cls, items: List[Any], pack: ContextPack) -> List[Any]:
        active_lanes = set(getattr(pack, "active_lanes", []) or [])
        if not active_lanes:
            return list(items or [])
        filtered: List[Any] = []
        for item in items or []:
            lane = cls._item_lane(item)
            if lane is None or lane in active_lanes:
                filtered.append(item)
        return filtered

    @staticmethod
    def _item_lane(item: Any) -> Optional[str]:
        meta = item.get("meta") if isinstance(item, dict) else getattr(item, "meta", None)
        if not isinstance(meta, dict):
            return None
        lane = str(meta.get("kb_lane") or "").strip().lower()
        return lane or None

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
        if pack.facts:
            pack.facts = self._dedupe_facts_by_signature(pack.facts)
        if not self.llm or not pack.facts:
            logger.debug("RLMController: prune skipped (llm=%s facts=%d)", bool(self.llm), len(pack.facts))
            return

        logger.info("RLMController: pruning facts with LLM (count=%d)", len(pack.facts))
        pack.facts = await prune_facts_for_query(
            llm=self.llm,
            query_text=pack.query_text,
            facts=list(pack.facts),
            threshold=0.6,
            max_keep=12,
            max_candidates=20,
        )
        pack.facts = self._dedupe_facts_by_signature(pack.facts)
        logger.info("RLMController: prune kept %d facts", len(pack.facts))

    @staticmethod
    def _dedupe_facts_by_signature(facts: List[Any]) -> List[Any]:
        """
        Deduplicate Fact-like objects by semantic signature while preserving grounding.

        Signature: (owner_type, owner_id, subject, predicate, object_text)
        Merge: union `source_ids`, keep max(confidence/salience).
        """
        from uma.common.accessors import get_attr_or_key

        def _norm_text(x: Any) -> str:
            """
            Normalize text for dedupe keys.

            Intentionally slightly lossy to collapse near-identical facts extracted
            from adjacent chunks (e.g. "defines X" vs "defines the X").
            """
            import re

            s = str(x or "").strip().lower()
            # Drop punctuation to avoid PDF/tokenization artifacts affecting keys.
            s = re.sub(r"[^a-z0-9]+", " ", s)
            # Drop common determiners that frequently vary across chunk boundaries.
            s = re.sub(r"\b(the|a|an)\b", " ", s)
            return " ".join(s.split()).strip()

        def _object_text(obj: Any) -> str:
            if isinstance(obj, dict):
                for k in ("text", "value", "name", "id"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v
                try:
                    import json

                    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
                except Exception:
                    return str(obj)
            return str(obj or "")

        def _get_source_ids(f: Any) -> List[str]:
            src = get_attr_or_key(f, "source_ids", []) or []
            if isinstance(src, list):
                return [str(x) for x in src if x]
            return []

        def _set_source_ids(f: Any, ids: List[str]) -> None:
            if isinstance(f, dict):
                f["source_ids"] = ids
            else:
                try:
                    setattr(f, "source_ids", ids)
                except Exception:
                    pass

        def _get_num(f: Any, field: str) -> Optional[float]:
            v = get_attr_or_key(f, field, None)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        def _set_num_max(dst: Any, src: Any, field: str) -> None:
            a = _get_num(dst, field)
            b = _get_num(src, field)
            if b is None:
                return
            val = b if a is None else max(a, b)
            if isinstance(dst, dict):
                dst[field] = val
            else:
                try:
                    setattr(dst, field, val)
                except Exception:
                    pass

        if not facts:
            return []

        seen: dict[tuple[str, str, str, str, str], Any] = {}
        out: List[Any] = []
        for f in list(facts or []):
            owner_type = _norm_text(get_attr_or_key(f, "owner_type", "") or "")
            owner_id = _norm_text(get_attr_or_key(f, "owner_id", "") or "")
            subj = _norm_text(get_attr_or_key(f, "subject", "") or "")
            pred = _norm_text(get_attr_or_key(f, "predicate", "") or "")
            obj = _norm_text(_object_text(get_attr_or_key(f, "object", "") or ""))
            key = (owner_type, owner_id, subj, pred, obj)

            if key not in seen:
                seen[key] = f
                out.append(f)
                continue

            keep = seen[key]
            merged_ids: List[str] = []
            seen_ids: set[str] = set()
            for sid in _get_source_ids(keep) + _get_source_ids(f):
                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    merged_ids.append(sid)
            if merged_ids:
                _set_source_ids(keep, merged_ids)

            _set_num_max(keep, f, "confidence")
            _set_num_max(keep, f, "salience")

        return out

    async def _expand_evidence_chunks_from_facts(
        self,
        request: RetrievalRequest,
        pack: ContextPack,
    ) -> None:
        chunks_ev = await expand_evidence_chunks_from_facts(
            env=self.env,
            request=request,
            pack=pack,
            max_items_per_type=self.max_items_per_type,
        )
        if chunks_ev:
            self._rebuild_chunk_buckets(pack)

    @staticmethod
    def _scopes_for_action(
        scopes: List[RetrievalScope],
        action: RetrievalAction,
    ) -> List[RetrievalScope]:
        requested_owner_type = getattr(action, "owner_type", None)
        if not requested_owner_type:
            return list(scopes)
        filtered = [scope for scope in scopes if scope.owner_type == requested_owner_type]
        return filtered or []

    @staticmethod
    def _rebuild_chunk_buckets(pack: ContextPack) -> None:
        """
        Separate chunk roles into conceptual buckets and rebuild `pack.chunks`
        with lane-aware deterministic precedence.

        Lane strategy:
        - KB lane (agent scope): query hits first (topical Q&A: most relevant doc wins).
        - User lane (user scope): evidence first (recall/grounding: citations win).
        """
        from uma.common.dedupe import dedupe_by_id

        evidence, query_hits, neighbors = partition_chunks_by_route(getattr(pack, "chunks", []) or [])
        pack.evidence_chunks = evidence
        pack.query_chunks = query_hits
        pack.neighbor_chunks = neighbors
        lane = str(getattr(pack, "owner_type", "") or "").strip().lower()
        if lane == "agent":
            pack.chunks = dedupe_by_id(list(query_hits or []) + list(evidence or []) + list(neighbors or []))
        else:
            pack.chunks = merge_chunks_with_precedence(evidence, query_hits, neighbors)

    # Fact description / parsing helpers live in semantic.query_pruner.


def _merge_unique(existing: List[Any], research: List[Any], limit: int) -> List[Any]:
    from uma.common.dedupe import dedupe_by_id

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


def _store_for_action(action_name: str) -> Optional[str]:
    a = str(action_name or "").strip()
    if a in {"search_semantic", "fetch_more_facts", "fetch_facts"}:
        return "facts"
    if a in {"search_chunks", "fetch_chunks"}:
        return "chunks"
    if a in {"episodic_clusters", "search_episodic", "fetch_episode_clusters"}:
        return "episodes"
    if a in {"graph_neighbors", "expand_graph"}:
        return "graph"
    if a in {"search_procedural"}:
        return "skills"
    return None


def _two_distinct_zero_yield_lanes(steps: List[Dict[str, Any]]) -> bool:
    """
    Only stop when two distinct stores have yielded 0 novelty consecutively.
    """
    last: List[tuple[str, int]] = []
    for s in reversed(steps or []):
        if not isinstance(s, dict) or s.get("event") != "action_result":
            continue
        store = str(s.get("store") or "").strip()
        if not store:
            continue
        try:
            novelty = int(s.get("novelty") or 0)
        except Exception:
            novelty = 0
        last.append((store, novelty))
        if len(last) >= 2:
            break
    if len(last) < 2:
        return False
    (s1, n1), (s2, n2) = last[0], last[1]
    return n1 == 0 and n2 == 0 and s1 != s2
