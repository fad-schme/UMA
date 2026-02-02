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
from .policy import assess_coverage
from ..policy import RetrievalPolicy, should_stop
from ...utils.identity import ensure_user_subject

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
        self.llm_max_tokens = int(getattr(rlm_cfg, "llm_max_tokens", 300))

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

        self.max_state_chars = max(200, int(getattr(rlm_cfg, "max_state_chars", 1200)))
        self.test_mode = bool(getattr(rlm_cfg, "test_mode", False))
        self.extract_snippets = bool(getattr(rlm_cfg, "extract_snippets", True))
        self.max_return_chars = int(getattr(rlm_cfg, "max_return_chars", 1200))
        self.max_eval_rounds = int(getattr(rlm_cfg, "max_eval_rounds", 2))
        self.max_eval_chunks = int(getattr(rlm_cfg, "max_eval_chunks", 12))
        self.max_snippet_chars = int(getattr(rlm_cfg, "max_snippet_chars", 320))
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

        pack = ContextPack(user_id=user_subject, query_text=query_text, owner_type="user")

        pack.working_memory = []
        if hasattr(self.env, "_memory"):
            wm = getattr(getattr(self.env, "_memory", None), "working_memory", None)
            if wm is not None:
                pack.working_memory = wm.get_context(user_subject)
        query_embedding = await self.env.get_query_embedding(query_text)

        # Pass trace_id to baseline retrieval
        await self._baseline_retrieval(pack, query_embedding, trace_id=trace_id)
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
        pack.steps.append({"step": 0, "coverage": coverage.to_dict()})

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
            pack.steps.append({"step": step, "coverage": coverage.to_dict()})
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
            logger.debug(
                "RLMController: step=%d facts preview=%s",
                step,
                [self._describe_fact(f)[:180] for f in pack.facts[:5]],
            )

        await self._prune_facts_with_llm(pack)
        self._apply_scope_bias(pack, policy)
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

    async def _baseline_retrieval(self, pack: ContextPack, query_embedding: List[float], trace_id: str = None) -> None:
        if query_embedding:
            try:
                results = await self.env.search_semantic(
                    pack.user_id,
                    query_embedding,
                    k=self.max_items_per_type,
                    query_text=pack.query_text,
                )
            except TypeError:
                # Backward compatibility: older envs may not accept query_text
                results = await self.env.search_semantic(
                    pack.user_id,
                    query_embedding,
                    k=self.max_items_per_type,
                )
            pack.facts = _merge_unique(
                pack.facts,
                results,
                self.max_items_per_type,
            )
            # Optional chunk retrieval if environment supports it
            if hasattr(self.env, "search_chunks"):
                try:
                    chunks = await self.env.search_chunks(
                        pack.user_id,
                        query_embedding,
                        k=self.max_items_per_type,
                    )
                    pack.chunks = _merge_unique(
                        getattr(pack, "chunks", []),
                        chunks,
                        self.max_items_per_type,
                    )
                except Exception:
                    logger.exception("RLMController: search_chunks failed")

        # Procedural baseline (skills)
        try:
            skills = await self.env.search_procedural(
                pack.user_id,
                query_embedding,
                k=self.max_items_per_type,
            )
            pack.skills = _merge_unique(pack.skills, skills, self.max_items_per_type)
        except Exception:
            logger.exception("RLMController: search_procedural failed")

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
                        )
                    )
                    pack.bump_predicate_offset(predicate, self.max_items_per_type)
            else:
                actions.append(RetrievalAction(action="search_semantic", k=self.max_items_per_type))

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
                )
            )
            if not has_cluster and len(pack.steps) >= 2:
                actions.append(
                    RetrievalAction(
                        action="search_episodic",
                        k=self.max_items_per_type,
                    )
                )

        if not pack.graph and pack.facts:
            predicate_scope = self._next_predicate_scope(pack)
            if predicate_scope:
                actions.append(
                    RetrievalAction(
                        action="expand_graph",
                        subject=pack.user_id,
                        predicate=predicate_scope[0],
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
        query_text: Optional[str] = None,
        trace_id: str = None,
        step: int = None,
    ) -> List[Any]:

        k = int(action.k) if action.k else self.max_items_per_type

        if action.action == "search_semantic":
            results = await self.env.search_semantic(
                user_id=user_subject,
                query_embedding=query_embedding,
                k=k,
                filters=action.filters,
                query_text=query_text,
            )
            logger.debug("RLMController: search_semantic returned %d", len(results or []))
            return results

        if action.action == "search_episodic":
            results = await self.env.search_episodic(
                user_id=user_subject,
                query_embedding=query_embedding,
                k=k,
                time_range=action.time_range,
            )
            logger.debug("RLMController: search_episodic returned %d", len(results or []))
            return results

        if action.action == "search_procedural":
            results = await self.env.search_procedural(
                user_id=user_subject,
                query_embedding=query_embedding,
                k=k,
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
                owner_scope=action.owner_scope,
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

        if action.action == "episodic_clusters":
            results = await self.env.episodic_cluster_summaries(
                user_id=user_subject,
                k=k,
                max_episodes=self.max_items_per_type,
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

    def _apply_scope_bias(self, pack: ContextPack, policy: RetrievalPolicy) -> None:
        """
        Apply recall/scope bias to retrieved items to prefer user memory for recall intent.
        """
        if not policy:
            return

        def _score_fact(f: Any) -> float:
            base = 0.0
            if isinstance(f, dict):
                sal = (f.get("meta") or {}).get("salience")
                conf = f.get("confidence")
            else:
                sal = getattr(f, "salience", None)
                conf = getattr(f, "confidence", None)
            try:
                base = (float(sal or 0.0) + float(conf or 0.5)) / 2.0
            except Exception:
                base = 0.0
            return base * policy.scope_weight(_get_owner_scope(f))

        def _score_chunk(c: Any) -> float:
            try:
                pos = getattr(c, "position", None)
                if pos is None and isinstance(c, dict):
                    pos = c.get("position")
                base = 1.0 / max(1, int(pos or 1))
            except Exception:
                base = 0.0
            return base * policy.scope_weight(_get_owner_scope(c))

        if pack.facts:
            pack.facts = sorted(pack.facts, key=_score_fact, reverse=True)
        if getattr(pack, "chunks", None):
            pack.chunks = sorted(pack.chunks, key=_score_chunk, reverse=True)

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
        meta = self._get_attr_or_key(fact, "meta", {})
        if isinstance(meta, dict):
            excerpt = meta.get("excerpt") or meta.get("description")
            if excerpt:
                return str(excerpt).replace("\n", " ").strip()
        obj = self._get_attr_or_key(fact, "object")
        if isinstance(obj, dict):
            text = obj.get("text") or ""
        else:
            text = str(obj)
        if text and text.strip():
            return text.strip()
        sub = self._get_attr_or_key(fact, "subject", "user")
        pred = self._get_attr_or_key(fact, "predicate", "related_to")
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

    def _get_attr_or_key(self, obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        if hasattr(obj, key):
            return getattr(obj, key, default)
        return default


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


def _get_owner_scope(item: Any) -> str:
    if isinstance(item, dict):
        return (item.get("owner_type") or item.get("owner_scope") or "").lower()
    return (getattr(item, "owner_type", None) or getattr(item, "owner_scope", None) or "").lower()


def _is_user_owned(item: Any) -> bool:
    return _get_owner_scope(item) == "user"
