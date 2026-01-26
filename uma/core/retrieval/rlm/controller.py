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
from .environment import MemoryEnvironment
from .policy import assess_coverage, should_stop
from ...utils.identity import ensure_user_subject

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
        env: MemoryEnvironment,
        *,
        deterministic_only: bool = True,
        timeout_s: float = 20.0,
        max_steps: int = 4,
        max_actions_per_step: int = 2,
        max_env_calls: int = 12,
        max_items_per_type: int = 30,
        llm_max_tokens: int = 300,
        salience_threshold: float = 0.6,
        min_semantic_facts: int = 4,
        min_high_salience_facts: int = 2,
        min_cluster_summaries: int = 1,
        cluster_k: int = 3,
        graph_predicate_limit: int = 2,
        predicate_weights: Optional[Dict[str, float]] = None,
        novelty_window: int = 2,
        min_recent_novelty: int = 1,
        max_state_chars: int = 1200,
    ) -> None:
        self.llm = llm
        self.env = env

        self.deterministic_only = bool(deterministic_only)
        self.timeout_s = float(timeout_s)

        self.max_steps = int(max_steps)
        self.max_actions_per_step = int(max_actions_per_step)
        self.max_env_calls = int(max_env_calls)
        self.max_items_per_type = int(max_items_per_type)
        self.llm_max_tokens = int(llm_max_tokens)

        self.salience_threshold = float(salience_threshold)
        self.min_semantic_facts = max(1, int(min_semantic_facts))
        self.min_high_salience_facts = max(0, int(min_high_salience_facts))
        self.min_cluster_summaries = max(0, int(min_cluster_summaries))

        self.cluster_k = max(1, int(cluster_k))
        self.graph_predicate_limit = max(1, int(graph_predicate_limit))
        self.predicate_weights = self._normalize_predicate_weights(predicate_weights)

        self.novelty_window = max(1, int(novelty_window))
        self.min_recent_novelty = max(0, int(min_recent_novelty))

        self.max_state_chars = max(200, int(max_state_chars))

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def retrieve_context(self, user_id: str, query_text: str) -> ContextPack:
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not query_text or not isinstance(query_text, str):
            raise ValueError("query_text must be a non-empty string")

        start = time.time()
        user_subject = ensure_user_subject(user_id)

        pack = ContextPack(user_id=user_subject, query_text=query_text)
        pack.record_seen()

        pack.working_memory = await self.env.get_working_memory(user_subject)
        query_embedding = await self.env.get_query_embedding(query_text)

        await self._baseline_retrieval(pack, query_embedding)
        pack.record_seen()

        coverage = assess_coverage(
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
        pack.steps.append({"step": 0, "coverage": coverage.to_dict()})

        stop, reason = should_stop(
            coverage=coverage,
            hard_budget_hit=False,
            prefer_clusters=True,
        )
        if stop:
            pack.warnings.append(f"stop:{reason}")
            return pack

        total_env_calls = 0

        for step in range(1, self.max_steps + 1):
            if (time.time() - start) > self.timeout_s:
                pack.warnings.append("stop:timeout")
                break
            if total_env_calls >= self.max_env_calls:
                pack.warnings.append("stop:max_env_calls")
                break

            coverage = assess_coverage(
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
            pack.steps.append({"step": step, "coverage": coverage.to_dict()})

            stop, reason = should_stop(
                coverage=coverage,
                hard_budget_hit=False,
                prefer_clusters=True,
            )
            if stop:
                pack.warnings.append(f"stop:{reason}")
                break

            decision = self._deterministic_decision(pack, coverage)
            if not decision or not decision.actions:
                break

            hard_budget_hit = False
            for action in decision.actions[: self.max_actions_per_step]:
                items = await self._execute_action(
                    user_subject=user_subject,
                    action=action,
                    query_embedding=query_embedding,
                )
                items = self._truncate_items(items)

                if action.action in {
                    "search_semantic",
                    "fetch_more_facts",
                    "fetch_facts",
                    "resolve_conflicts",
                }:
                    pack.facts = _merge_unique(pack.facts, items, self.max_items_per_type)
                    pack.apply_novelty(items, "facts")

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

                total_env_calls += 1
                if total_env_calls >= self.max_env_calls:
                    pack.warnings.append("stop:max_env_calls")
                    hard_budget_hit = True
                    break

            if hard_budget_hit:
                break

        return pack

    # ------------------------------------------------------------------
    # BASELINE RETRIEVAL
    # ------------------------------------------------------------------

    async def _baseline_retrieval(self, pack: ContextPack, query_embedding: List[float]) -> None:
        if query_embedding:
            pack.facts = _merge_unique(
                pack.facts,
                await self.env.search_semantic(pack.user_id, query_embedding, k=self.max_items_per_type),
                self.max_items_per_type,
            )

        pack.episodes = _merge_unique(
            pack.episodes,
            await self.env.episodic_cluster_summaries(
                pack.user_id,
                k=self.cluster_k,
                max_episodes=self.max_items_per_type,
            ),
            self.max_items_per_type,
        )

        pack.graph = _merge_unique(
            pack.graph,
            await self.env.graph_neighbors(
                user_id=pack.user_id,
                node_id=pack.user_id,
                predicate_scope=self._next_predicate_scope(pack),
                depth=1,
                k=self.max_items_per_type,
            ),
            self.max_items_per_type,
        )

    # ------------------------------------------------------------------
    # DECISION LOGIC
    # ------------------------------------------------------------------

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
                        )
                    )
                    pack.bump_predicate_offset(predicate, self.max_items_per_type)
            else:
                actions.append(RetrievalAction(action="search_semantic", k=self.max_items_per_type))

        if coverage.needs_clusters:
            actions.append(
                RetrievalAction(
                    action="fetch_episode_clusters",
                    k=self.cluster_k,
                    time_range=None,
                    min_salience=self.salience_threshold,
                )
            )

        if coverage.has_contradictions:
            fact_ids = list(pack.seen_fact_ids)[: self.max_items_per_type]
            if fact_ids:
                actions.append(
                    RetrievalAction(
                        action="resolve_conflicts",
                        fact_ids=fact_ids,
                        k=self.max_items_per_type,
                        reason="contradictions",
                    )
                )

        if not pack.graph and pack.facts:
            predicate_scope = self._next_predicate_scope(pack)
            actions.append(
                RetrievalAction(
                    action="expand_graph",
                    subject=pack.user_id,
                    predicate=predicate_scope[0] if predicate_scope else None,
                    hops=1,
                    direction="outbound",
                    k=min(self.max_items_per_type, 20),
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
    ) -> List[Any]:

        k = int(action.k) if action.k else self.max_items_per_type

        if action.action == "search_semantic":
            return await self.env.search_semantic(
                user_id=user_subject,
                query_embedding=query_embedding,
                k=k,
                filters=action.filters,
            )

        if action.action == "fetch_more_facts":
            offset = int(action.filters.get("offset", 0)) if action.filters else 0
            return await self.env.fetch_more_facts(
                user_id=user_subject,
                predicate=action.predicate,
                k=k,
                offset=offset,
                owner_scope=action.owner_scope,
            )

        if action.action == "episodic_clusters":
            return await self.env.episodic_cluster_summaries(
                user_id=user_subject,
                k=k,
                max_episodes=self.max_items_per_type,
            )

        if action.action == "fetch_episode_clusters":
            return await self.env.fetch_episode_clusters(
                user_id=user_subject,
                k=k,
                max_episodes=self.max_items_per_type,
                time_range=action.time_range,
                min_salience=action.min_salience,
            )

        if action.action == "graph_neighbors":
            return await self.env.graph_neighbors(
                user_id=user_subject,
                node_id=action.node_id,
                predicate_scope=action.predicate_scope,
                depth=int(action.depth or 1),
                k=k,
            )

        if action.action == "expand_graph":
            return await self.env.expand_graph(
                user_id=user_subject,
                subject=action.subject,
                predicate=action.predicate,
                hops=int(action.hops or 1),
                direction=action.direction,
                k=k,
            )

        if action.action == "resolve_conflicts":
            return await self.env.resolve_conflicts(user_id=user_subject, fact_ids=action.fact_ids or [])

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


def _merge_unique(existing: List[Any], research: List[Any], limit: int) -> List[Any]:
    seen = set()
    out = []

    def _id(x):
        if isinstance(x, dict):
            return x.get("id")
        return getattr(x, "id", None)

    for it in existing + research:
        key = _id(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break

    return out
