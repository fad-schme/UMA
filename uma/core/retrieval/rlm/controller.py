# uma/core/retrieval/rlm/controller.py

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .context_pack import ContextPack
from .decisions import RetrievalAction
from . import decisions
from .coverage import assess_coverage, compute_confidence
from ..policy import RetrievalPolicy, should_stop
from ...utils.identity import normalize_user_id
from ...chunk.core import merge_chunks_with_precedence, partition_chunks_by_route
from ...semantic.query_pruner import prune_facts_for_query
from .evidence import expand_evidence_chunks_from_facts

from ..ranking import Ranker

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

    async def retrieve_context(self, user_id: str, query_text: str) -> ContextPack:
        if not user_id or not isinstance(user_id, str):
            logger.error("RLMController.retrieve_context: user_id must be a non-empty string")
            raise ValueError("user_id must be a non-empty string")
        if not query_text or not isinstance(query_text, str):
            logger.error("RLMController.retrieve_context: query_text must be a non-empty string")
            raise ValueError("query_text must be a non-empty string")

        policy = RetrievalPolicy(query_text)
        logger.info(
            "RLMController.retrieve_context: start user_id=%s query=%r",
            user_id,
            query_text,
        )
        start = time.time()
        normalized_user_id = normalize_user_id(user_id)

        # --- TRACE CONTEXT ---
        trace_id = f"rlm:{user_id}:{int(time.time()*1000)}"
        logger.info(
            "RLM_START trace_id=%s user=%s recall_query=%s",
            trace_id,
            user_id,
            any(k in query_text.lower() for k in ["remember", "recall", "previous", "earlier", "last time"]),
        )

        agent_id = getattr(self.env, "_agent_id", None)
        pack = ContextPack(
            user_id=user_id,
            query_text=query_text,
            owner_type=None,
            owner_id=None,
            agent_id=agent_id,
        )

        pack.working_memory = []
        if hasattr(self.env, "_memory"):
            wm = getattr(getattr(self.env, "_memory", None), "working_memory", None)
            if wm is not None:
                pack.working_memory = wm.get_context(user_id)
        query_embedding = await self.env.get_query_embedding(query_text)

        # Lane decision: recall = user-only; KB = agent KB + user-owned KB docs.
        is_recall = policy.recall_score >= 0.75

        agent_id = getattr(self.env, "_agent_id", None)
        if not agent_id:
            logger.error("RLMController.retrieve_context: agent_id is required")
            raise ValueError("RLMController.retrieve_context: agent_id is required")

        if is_recall:
            scopes = [("user", normalized_user_id)]
        else:
            # KB lane must include BOTH agent scope and user-owned documents scope.
            scopes = [("agent", agent_id), ("user", normalized_user_id)]

        # Keep these fields for telemetry/back-compat; primary execution uses `scopes`.
        pack.owner_type, pack.owner_id = scopes[0]
        logger.info("RLM_LANE scopes=%s", scopes)

        # Baseline retrieval per-scope, then merge into the pack.
        for idx, (owner_type, owner_id) in enumerate(scopes):
            await self._baseline_retrieval(
                pack,
                query_embedding,
                trace_id=f"{trace_id}:{idx}",
                owner_type=owner_type,
                owner_id=owner_id,
            )
            if idx == 0:
                # Preserve first-scope telemetry for tests and downstream expectations.
                pack.owner_type, pack.owner_id = owner_type, owner_id

        # Deterministic merge cleanup after multi-scope baseline.
        try:
            from ...utils.dedupe import dedupe_by_id as _dedupe_by_id
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


        # Tighten evidence expansion: prune facts before expanding cited chunks.
        await self._prune_facts_with_llm(pack)
        await self._expand_evidence_chunks_from_facts(
            pack,
            owner_type=str(pack.owner_type or scopes[0][0]),
            owner_id=pack.owner_id or scopes[0][1],
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
            from ...semantic.query_pruner import describe_fact as _describe_fact
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
                pack.warnings.append("stop:diminishing_returns")
                logger.info(
                    "RLMController: stop step=%d reason=diminishing_returns novelty_recent_sum=%d window=%d",
                    step,
                    coverage.novelty_recent_sum,
                    self.novelty_window,
                )
                break

            decision = decisions.deterministic_decision(
                pack,
                coverage,
                cfg={
                    "max_items_per_type": self.max_items_per_type,
                    "cluster_k": self.cluster_k,
                    "salience_threshold": self.salience_threshold,
                    "graph_predicate_limit": self.graph_predicate_limit,
                    "chunk_fallback_enabled": self.chunk_fallback_enabled,
                    "chunk_fallback_k_multiplier": self.chunk_fallback_k_multiplier,
                    "next_predicate_scope": lambda p, limit: decisions.next_predicate_scope(
                        facts=getattr(p, "facts", []) or [],
                        predicate_weights=getattr(self, "predicate_weights", None),
                        graph_predicate_limit=getattr(self, "graph_predicate_limit", 2),
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
                if action.action == "search_chunks":
                    logger.debug(
                        "RLMController: dispatching search_chunks step=%d owner=%s:%s k=%s",
                        step,
                        str(pack.owner_type or owner_type),
                        pack.owner_id or owner_id,
                        action.k,
                    )
                items = await self._execute_action(
                    user_id=user_id,
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
                    self._rebuild_chunk_buckets(pack)

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
                from ...semantic.query_pruner import describe_fact as _describe_fact
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
            pack,
            owner_type=str(pack.owner_type or scopes[0][0]),
            owner_id=pack.owner_id or scopes[0][1],
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
            results = await self._search_semantic_core(
                user_id=pack.user_id,
                query_embedding=query_embedding,
                k=self.max_items_per_type,
                query_text=pack.query_text,
                owner_type=owner_type,
                owner_id=owner_id,
                filters=None,
            )
            results = self.ranker.rank_facts(results or [], query_text=pack.query_text)
            pack.facts = _merge_unique(
                pack.facts,
                results,
                self.max_items_per_type,
            )
            logger.debug(
                "RLMController._baseline_retrieval: semantic facts=%d owner=%s:%s",
                len(pack.facts),
                owner_type,
                owner_id,
            )
            # Optional chunk retrieval via centralized ChunkCore search.
            chunk_core = getattr(getattr(self.env, "_memory", None), "chunk_core", None)
            if chunk_core is None:
                chunk_core = getattr(self.env, "_chunk_core", None)
            if chunk_core is None:
                logger.debug("RLMController._baseline_retrieval: chunk_core missing on env; skipping chunk search")
            else:
                try:
                    search_fn = getattr(chunk_core, "search_chunks_for_rlm", None) or getattr(
                        chunk_core, "search_chunks", None
                    )
                    if search_fn is None:
                        chunks = []
                    else:
                        kwargs = {
                            "query_embedding": list(query_embedding),
                            "owner_type": owner_type,
                            "owner_id": owner_id,
                            "k": self.max_items_per_type,
                            "query_text": pack.query_text,
                        }
                        chunks = await search_fn(**kwargs)
                    chunks = self.ranker.rank_chunks(chunks or [], query_text=pack.query_text)
                    # Neighbor expansion happens inside ChunkCore.search_chunks(expand_neighbors=True).
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
            skills = self.ranker.rank_skills(skills or [], query_text=pack.query_text)
            pack.skills = _merge_unique(pack.skills, skills, self.max_items_per_type)
        except Exception:
            logger.exception("RLMController: search_procedural failed")

        episodes = await self.env.episodic_cluster_summaries(
            pack.user_id,
            owner_type=owner_type,
            owner_id=owner_id,
            k=self.cluster_k,
            max_episodes=self.max_items_per_type,
        )
        episodes = self.ranker.rank_episodes(episodes or [], query_text=pack.query_text)
        pack.episodes = _merge_unique(
            pack.episodes,
            episodes,
            self.max_items_per_type,
        )

        if owner_type == "user":
            pack.graph = _merge_unique(
                pack.graph,
                await self.env.graph_neighbors(
                    user_id=pack.user_id,
                    node_id=pack.user_id,
                    predicate_scope=decisions.next_predicate_scope(
                        facts=getattr(pack, "facts", []) or [],
                        predicate_weights=getattr(self, "predicate_weights", None),
                        graph_predicate_limit=getattr(self, "graph_predicate_limit", 2),
                    ),
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
        Resolves owner scope and passes through filters.

        IMPORTANT
        ---------
        Retrieval is ownership-scoped only via (owner_type, owner_id). Fact.subject
        is metadata and must not gate retrieval.
        """
        semantic_core = getattr(getattr(self.env, "_memory", None), "semantic_core", None)
        if semantic_core is None:
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
            normalized_user_id = normalize_user_id(user_id)
        except Exception:
            logger.exception("RLMController._search_semantic_core: invalid subject=%r", user_id)
            return []

        retrieval_cfg = getattr(self.env, "_memory", None)
        retrieval_cfg = getattr(retrieval_cfg, "retrieval_cfg", None)
       #ctx_cfg = getattr(retrieval_cfg, "context", None) if retrieval_cfg else None

        if owner_type == "agent":
            resolved_owner_id = owner_id or getattr(self.env, "_agent_id", None)
            if not resolved_owner_id:
                logger.warning("RLMController._search_semantic_core: missing agent_id for agent scope")
                return []
        elif owner_type == "user":
            resolved_owner_id = owner_id or normalized_user_id
        else:
            logger.warning("RLMController._search_semantic_core: invalid owner_type=%r", owner_type)
            return []

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RLMController._search_semantic_core: owner_type=%s owner_id=%s k=%d offset=%d",
                owner_type,
                resolved_owner_id,
                k,
                offset,
            )
        return await semantic_core.search(
            query_embedding=list(query_embedding),
            owner_type=owner_type,
            owner_id=resolved_owner_id,
            k=int(k),
            offset=int(offset),
            filters=filters,
            query_text=query_text,
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
        episodic_core = getattr(getattr(self.env, "_memory", None), "episodic_core", None)
        if episodic_core is None:
            return []
        try:
            normalized_user_id = normalize_user_id(user_id)
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
            resolved_owner_id = owner_id or normalized_user_id

        episodes = await episodic_core.search(
            user_id=normalized_user_id,
            query_embedding=list(query_embedding),
            owner_type=owner_type,
            owner_id=resolved_owner_id,
            k=int(k),
            offset=int(offset),
        )
        try:
            episodes = self.env._filter_time_range(episodes or [], time_range)
        except Exception:
            logger.exception("RLMController._search_episodic_core: failed to filter time range")
            raise
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
        procedural_core = getattr(getattr(self.env, "_memory", None), "procedural_core", None)
        if procedural_core is None:
            return []
        try:
            normalized_user_id = normalize_user_id(user_id)
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
            resolved_owner_id = owner_id or normalized_user_id

        return await procedural_core.search(
            user_id=normalized_user_id,
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

    # ------------------------------------------------------------------
    # ACTION EXECUTION
    # ------------------------------------------------------------------

    async def _execute_action(
        self,
        *,
        user_id: str,
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
                user_id=user_id,
                query_embedding=query_embedding,
                k=k,
                filters=action.filters,
                query_text=query_text,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: search_semantic returned %d", len(results or []))
            return self.ranker.rank_facts(results or [], query_text=query_text or "")

        if action.action == "search_episodic":
            results = await self._search_episodic_core(
                user_id=user_id,
                query_embedding=query_embedding,
                k=k,
                time_range=action.time_range,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: search_episodic returned %d", len(results or []))
            return self.ranker.rank_episodes(results or [], query_text=query_text or "")

        if action.action == "search_procedural":
            results = await self._search_procedural_core(
                user_id=user_id,
                query_embedding=query_embedding,
                k=k,
                owner_type=lane_owner_type,
                owner_id=lane_owner_id,
            )
            logger.debug("RLMController: search_procedural returned %d", len(results or []))
            return self.ranker.rank_skills(results or [], query_text=query_text or "")

        try:
            user_subject = normalize_user_id(user_id)
        except Exception:
            user_subject = user_id

        results = await decisions.execute_action(
            env=self.env,
            user_subject=user_subject,
            action=action,
            query_embedding=list(query_embedding),
            query_text=query_text,
            owner_type=lane_owner_type,
            owner_id=lane_owner_id,
            default_k=self.max_items_per_type,
            trace_id=trace_id,
        )
        logger.debug("RLMController: %s returned %d", getattr(action, "action", "action"), len(results or []))
        a = str(getattr(action, "action", "") or "")
        if a in {"fetch_more_facts", "fetch_facts"}:
            return self.ranker.rank_facts(results or [], query_text=query_text or "")
        if a in {"fetch_chunks", "search_chunks"}:
            return self.ranker.rank_chunks(results or [], query_text=query_text or "")
        if a in {"episodic_clusters", "fetch_episode_clusters"}:
            return self.ranker.rank_episodes(results or [], query_text=query_text or "")
        if a in {"search_procedural"}:
            return self.ranker.rank_skills(results or [], query_text=query_text or "")
        return results

        # (Unreachable)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

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
        pack.facts = await prune_facts_for_query(
            llm=self.llm,
            query_text=pack.query_text,
            facts=list(pack.facts),
            threshold=0.6,
            max_keep=12,
            max_candidates=20,
        )
        logger.info("RLMController: prune kept %d facts", len(pack.facts))

    async def _expand_evidence_chunks_from_facts(
        self,
        pack: ContextPack,
        *,
        owner_type: str,
        owner_id: Optional[str],
    ) -> None:
        chunks_ev = await expand_evidence_chunks_from_facts(
            env=self.env,
            pack=pack,
            owner_type=owner_type,
            owner_id=owner_id,
            max_items_per_type=self.max_items_per_type,
        )
        if chunks_ev:
            self._rebuild_chunk_buckets(pack)

    @staticmethod
    def _rebuild_chunk_buckets(pack: ContextPack) -> None:
        """
        Separate chunk roles into conceptual buckets and rebuild `pack.chunks`
        with deterministic precedence:
        1) evidence, 2) query hits, 3) neighbors.
        """
        evidence, query_hits, neighbors = partition_chunks_by_route(getattr(pack, "chunks", []) or [])
        pack.evidence_chunks = evidence
        pack.query_chunks = query_hits
        pack.neighbor_chunks = neighbors
        pack.chunks = merge_chunks_with_precedence(evidence, query_hits, neighbors)

    # Fact description / parsing helpers live in semantic.query_pruner.


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
