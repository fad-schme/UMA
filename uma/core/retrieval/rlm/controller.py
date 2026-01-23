# uma/core/retrieval/rlm/controller.py

from __future__ import annotations
import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

try:
    from ...utils.text import extract_query_terms
except Exception:  # pragma: no cover
    extract_query_terms = None

from .context_pack import ContextPack
from .environment import MemoryEnvironment
from .decisions import ControllerDecision, RetrievalAction
from .policy import assess_coverage, merge_unique
from ...utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)


class RLMController:
    """
    RLMController — Recursive, Bounded Memory Retrieval Controller.

    This controller implements **Recursive Language Model (RLM)-style retrieval**
    for UMA. It allows the system to explore long-term memory iteratively
    instead of relying on a single retrieval pass.

    IMPORTANT:
    ----------
    This controller is **not an agent** and **does not answer user queries**.
    This controller performs **memory navigation only**.
    It never plans tasks, reasons about solutions, or decides what to say.
    All agent reasoning and response generation remain outside UMA.

    Its sole responsibility is memory navigation:
    deciding which memory stores to query next in order to improve context coverage.

    Role in UMA
    -------------
    • Used exclusively by `UMAMemory.get_user_context()`
    • Activated only when enabled via `retrieval.rlm.enabled`
    • Operates entirely in read-only mode

    What this controller DOES
    -------------------------
    • Starts with baseline retrieval (via RetrievalService)
    • Uses an LLM *only* as a control model to decide next retrieval steps
    • Performs bounded, iterative memory retrieval via MemoryEnvironment
    • Merges and deduplicates results across steps
    • Stops deterministically based on:
        - max_steps
        - max_actions_per_step
        - per-type item caps
        - wall-clock timeout
    • Returns a structured ContextPack

    What this controller DOES NOT do
    --------------------------------
    • Does not generate natural language answers
    • Does not perform task reasoning or planning
    • Does not construct prompts
    • Does not mutate or write memory
    • Does not access raw databases or adapters directly

    Safety and determinism
    ----------------------
    • All memory access is mediated by MemoryEnvironment
    • All LLM outputs are strictly parsed as JSON
    • Invalid decisions result in safe early termination
    • Failures automatically fall back to classic retrieval

    Relationship to RetrievalService
    --------------------------------
    • RetrievalService = single-shot, deterministic retrieval
    • RLMController = iterative, decision-driven retrieval

    The RLMController **builds on top of** RetrievalService; it does not replace it.

    Design philosophy
    -----------------
    • Treat long-term memory as an external environment
    • Keep working context small and relevant
    • Trade time (iterations) for space (context window)
    • Preserve UMA's role as a memory system, not an agent
    """

    def __init__(
        self,
        llm: Any,
        env: MemoryEnvironment,
        test_mode: bool = False,
        extract_snippets: bool = True,
        max_eval_rounds: int = 2,
        max_eval_chunks: int = 12,
        max_snippet_chars: int = 320,
        max_steps: int = 4,
        max_actions_per_step: int = 2,
        max_items_per_type: int = 30,
        llm_max_tokens: int = 300,
        timeout_s: float = 20.0,
        max_env_calls: int = 12,
        max_return_chars: int = 1200,
        semantic_first: bool = True,
        clusters_first: bool = True,
        salience_threshold: float = 0.6,
        min_semantic_facts: int = 4,
        min_high_salience_facts: int = 2,
        min_cluster_summaries: int = 1,
        cluster_k: int = 3,
        graph_predicate_limit: int = 2,
        predicate_weights: Optional[Dict[str, float]] = None,
        deterministic_only: bool = True,
    ) -> None:
        self.llm = llm
        self.env = env
        self.test_mode = test_mode
        self.extract_snippets = extract_snippets
        self.max_eval_rounds = max_eval_rounds
        self.max_eval_chunks = max_eval_chunks
        self.max_snippet_chars = max_snippet_chars

        self.max_steps = max_steps
        self.max_actions_per_step = max_actions_per_step
        self.max_items_per_type = max_items_per_type
        self.llm_max_tokens = llm_max_tokens
        self.timeout_s = timeout_s
        self.max_env_calls = max_env_calls
        self.max_return_chars = max_return_chars
        self.max_parse_failures = 2
        self.semantic_first = bool(semantic_first)
        self.clusters_first = bool(clusters_first)
        self.salience_threshold = float(salience_threshold)
        self.min_semantic_facts = max(1, int(min_semantic_facts))
        self.min_high_salience_facts = max(0, int(min_high_salience_facts))
        self.min_cluster_summaries = max(0, int(min_cluster_summaries))
        self.cluster_k = max(1, int(cluster_k))
        self.graph_predicate_limit = max(1, int(graph_predicate_limit))
        self.predicate_weights = self._normalize_predicate_weights(predicate_weights)
        self.deterministic_only = bool(deterministic_only)

        logger.info(
            "RLMController initialized steps=%d actions/step=%d max_items=%d max_env_calls=%d",
            max_steps,
            max_actions_per_step,
            max_items_per_type,
            max_env_calls,
        )

    async def retrieve_context(self, user_id: str, query_text: str) -> ContextPack:
        if not user_id or not query_text:
            raise ValueError("RLMController.retrieve_context: invalid input")

        start = time.time()
        user_subject = ensure_user_subject(user_id)
        pack = ContextPack(user_id=user_subject, query_text=query_text)

        # Always include WM
        pack.working_memory = await self.env.get_working_memory(user_subject)

        # Baseline retrieval
        baseline = await self._safe(
            self.env.retrieve_all(user_subject, query_text),
            default={},
        )
        pack.episodes = baseline.get("episodes", [])
        pack.facts = baseline.get("facts", [])
        pack.skills = baseline.get("skills", [])
        pack.graph = baseline.get("graph", [])

        if self.clusters_first:
            clusters = await self._safe(
                self.env.episodic_cluster_summaries(
                    user_id=user_subject,
                    k=self.cluster_k,
                    max_episodes=self.max_items_per_type,
                ),
                default=[],
            )
            if clusters:
                pack.episodes = clusters
            else:
                pack.warnings.append("clusters_unavailable")
            pack.steps.append(
                {
                    "step": 0,
                    "clusters_first": True,
                    "cluster_count": len(clusters),
                }
            )

        pack.steps.append({"step": 0, "baseline": pack.snapshot()})

        if self.extract_snippets:
            extracted = await self._evaluate_and_extract(pack, query_text, eval_round=0)
            if extracted is not None:
                return extracted

        coverage = assess_coverage(
            facts=pack.facts,
            episodes=pack.episodes,
            graph=pack.graph,
            salience_threshold=self.salience_threshold,
            min_semantic_facts=self.min_semantic_facts,
            min_high_salience_facts=self.min_high_salience_facts,
            min_cluster_summaries=self.min_cluster_summaries,
            require_semantic=self.semantic_first,
            prefer_clusters=self.clusters_first,
        )
        pack.steps.append({"step": 0, "coverage": coverage.to_dict()})
        if coverage.enough:
            return pack

        # Precompute embedding once for semantic/episodic actions
        query_embedding = await self.env.get_query_embedding(query_text)

        total_env_calls = 0
        parse_failures = 0
        for step in range(1, self.max_steps + 1):
            if time.time() - start > self.timeout_s:
                pack.warnings.append("timeout")
                break

            coverage = assess_coverage(
                facts=pack.facts,
                episodes=pack.episodes,
                graph=pack.graph,
                salience_threshold=self.salience_threshold,
                min_semantic_facts=self.min_semantic_facts,
                min_high_salience_facts=self.min_high_salience_facts,
                min_cluster_summaries=self.min_cluster_summaries,
                require_semantic=self.semantic_first,
                prefer_clusters=self.clusters_first,
            )
            pack.steps.append({"step": step, "coverage": coverage.to_dict()})

            if self.deterministic_only:
                decision = self._deterministic_decision(user_subject, pack, coverage)
                if decision is None:
                    decision = self._fallback_graph_action(user_subject, pack)
                    if not decision.actions or decision.done:
                        break
            else:
                decision = await self._decide(pack, coverage)
                if decision is None:
                    parse_failures += 1
                    pack.warnings.append("controller_parse_failed")
                    if parse_failures >= self.max_parse_failures:
                        break
                    continue
                if not decision.actions or decision.done:
                    decision = self._deterministic_decision(user_subject, pack, coverage)
                    if decision is None:
                        decision = self._fallback_graph_action(user_subject, pack)
                        if not decision.actions or decision.done:
                            break
            pack.steps.append({"step": step, "decision": decision.model_dump()})

            env_calls = 0
            for action in decision.actions[: self.max_actions_per_step]:
                if total_env_calls >= self.max_env_calls:
                    pack.warnings.append("max_env_calls")
                    break
                if action.action == "stop":
                    break

                items: List[Any] = []
                action_type = action.action

                if action_type in {"retrieve", "expand_graph"}:
                    mem_type = "graph" if action_type == "expand_graph" else action.memory_type
                    items = await self._safe(
                        self.env.retrieve_slice(user_subject, mem_type, query_text),
                        default=[],
                    )
                elif action_type == "search_semantic":
                    if query_embedding:
                        items = await self._safe(
                            self.env.search_semantic(
                                user_id=user_subject,
                                query_embedding=query_embedding,
                                k=action.k or self.max_items_per_type,
                                filters=action.filters,
                            ),
                            default=[],
                        )
                elif action_type == "search_episodic":
                    if query_embedding:
                        items = await self._safe(
                            self.env.search_episodic(
                                user_id=user_subject,
                                query_embedding=query_embedding,
                                k=action.k or self.max_items_per_type,
                                time_range=action.time_range,
                            ),
                            default=[],
                        )
                elif action_type == "fetch_facts":
                    items = await self._safe(
                        self.env.fetch_facts_by_ids(user_id=user_subject, ids=action.ids or []),
                        default=[],
                    )
                elif action_type == "fetch_episode_summaries":
                    items = await self._safe(
                        self.env.fetch_episode_summaries(ids=action.ids or []),
                        default=[],
                    )
                elif action_type == "fetch_episode_transcripts":
                    items = await self._safe(
                        self.env.fetch_episode_transcripts(ids=action.ids or []),
                        default=[],
                    )
                elif action_type == "graph_neighbors":
                    items = await self._safe(
                        self.env.graph_neighbors(
                            user_id=user_subject,
                            node_id=action.node_id or "",
                            predicate_scope=action.predicate_scope,
                            depth=1,
                            k=action.k or self.max_items_per_type,
                        ),
                        default=[],
                    )
                elif action_type == "episodic_clusters":
                    items = await self._safe(
                        self.env.episodic_cluster_summaries(
                            user_id=user_subject,
                            k=action.k or 5,
                            max_episodes=self.max_items_per_type,
                        ),
                        default=[],
                    )

                items = self._truncate_items(items)

                if action_type in {"search_episodic", "fetch_episode_summaries", "fetch_episode_transcripts", "episodic_clusters"}:
                    pack.episodes = merge_unique(pack.episodes, items, self.max_items_per_type)
                elif action_type in {"search_semantic", "fetch_facts"}:
                    pack.facts = merge_unique(pack.facts, items, self.max_items_per_type)
                elif action_type == "graph_neighbors":
                    pack.graph = merge_unique(pack.graph, items, self.max_items_per_type)
                elif action_type in {"retrieve", "expand_graph"}:
                    mem_type = "graph" if action_type == "expand_graph" else action.memory_type
                    if mem_type == "episodic":
                        pack.episodes = merge_unique(pack.episodes, items, self.max_items_per_type)
                    elif mem_type == "semantic":
                        pack.facts = merge_unique(pack.facts, items, self.max_items_per_type)
                    elif mem_type == "procedural":
                        pack.skills = merge_unique(pack.skills, items, self.max_items_per_type)
                    elif mem_type == "graph":
                        pack.graph = merge_unique(pack.graph, items, self.max_items_per_type)

                env_calls += 1
                total_env_calls += 1
                pack.steps.append(
                    {
                        "step": step,
                        "env_call": env_calls,
                        "action": action.model_dump(),
                        "counts": pack.snapshot()["counts"],
                    }
                )

            coverage = assess_coverage(
                facts=pack.facts,
                episodes=pack.episodes,
                graph=pack.graph,
                salience_threshold=self.salience_threshold,
                min_semantic_facts=self.min_semantic_facts,
                min_high_salience_facts=self.min_high_salience_facts,
                min_cluster_summaries=self.min_cluster_summaries,
                require_semantic=self.semantic_first,
                prefer_clusters=self.clusters_first,
            )
            if coverage.enough:
                if self.extract_snippets:
                    extracted = await self._evaluate_and_extract(pack, query_text, eval_round=step)
                    if extracted is not None:
                        return extracted
                break

        if self.extract_snippets:
            extracted = await self._evaluate_and_extract(pack, query_text, eval_round=self.max_steps + 1)
            if extracted is not None:
                return extracted

            pack.working_memory = []
            pack.episodes = []
            pack.facts = []
            pack.skills = []
            pack.graph = []
            pack.warnings.append("insufficient_evidence")
        return pack

    async def _decide(
        self,
        pack: ContextPack,
        coverage,
    ) -> ControllerDecision | None:
        """
        Ask LLM what to fetch next. Must return strict JSON.
        """
        state = pack.snapshot()
        state["snippets"] = self._build_snippet_summary(pack)
        state["predicate_counts"] = self._predicate_counts(pack)
        if coverage is not None:
            state["coverage"] = coverage.to_dict()
        state_json = json.dumps(state)
        if self.max_return_chars and len(state_json) > self.max_return_chars:
            state_json = state_json[: self.max_return_chars]
            pack.warnings.append("state_truncated")

        prompt = (
            "You are UMA Retrieval Controller.\n\n"
            "Goal: choose the minimal, most relevant memory retrieval actions needed "
            "to answer the user's query.\n\n"
            "Think silently using this checklist (do NOT reveal reasoning):\n"
            "- UNDERSTAND the core question and what a useful answer should enable.\n"
            "- ANALYZE which memory types matter (episodic, semantic, procedural, graph).\n"
            "- EVALUATE coverage: high-salience facts, graph support, and episodic cluster signals.\n"
            "- If coverage flags contradictions, request targeted episodic follow-ups.\n"
            "- REASON about gaps in the current snippets and what to fetch next.\n"
            "- SYNTHESIZE the smallest safe set of retrieval actions.\n"
            "- CONCLUDE by selecting actions or stopping.\n\n"
            "Return JSON ONLY matching ControllerDecision schema. No prose.\n\n"
            "Allowed actions:\n"
            "- search_semantic(k, filters)\n"
            "- search_episodic(k, time_range)\n"
            "- episodic_clusters(k)\n"
            "- fetch_facts(ids)\n"
            "- fetch_episode_summaries(ids)\n"
            "- fetch_episode_transcripts(ids)\n"
            "- graph_neighbors(node_id, predicate_scope, k)\n"
            "- retrieve(memory_type)\n"
            "- expand_graph(memory_type='graph')\n"
            "- stop\n\n"
            f"STATE:\n{state_json}\n"
        )
        if self.test_mode:
            prompt += (
                "\nSTRICT JSON MODE:\n"
                "- Output a single JSON object.\n"
                "- Do not include function-like strings.\n"
                "- filters must be null or an object with only 'subject' or 'topic' as strings.\n"
                "- If unsure, set filters to null.\n"
                "- Example:\n"
                "{\"actions\":[{\"action\":\"search_semantic\",\"k\":5,"
                "\"filters\":null,\"reason\":\"need facts\"}],\"done\":false}\n"
            )

        try:
            raw = await asyncio.wait_for(
                self.llm.generate(
                    [{"role": "system", "content": prompt}],
                    max_tokens=self.llm_max_tokens,
                    temperature=0.0,
                    format="json",
                ),
                timeout=self.timeout_s,
            )
            return ControllerDecision.from_json(raw)
        except Exception as exc:
            logger.warning("RLMController decision failed; using fallback (%s)", exc)
            return ControllerDecision(
                actions=[RetrievalAction(action="retrieve", memory_type="semantic", reason="fallback")],
                done=False,
            )

    def _deterministic_decision(
        self,
        user_id: str,
        pack: ContextPack,
        coverage,
    ) -> Optional[ControllerDecision]:
        if self.semantic_first and coverage.needs_semantic:
            try:
                subject = ensure_user_subject(user_id)
            except Exception:
                subject = user_id
            action = RetrievalAction(
                action="search_semantic",
                k=self.max_items_per_type,
                filters={"subject": subject},
                reason="coverage_semantic",
            )
            return ControllerDecision(actions=[action], done=False)

        if self.clusters_first and coverage.needs_clusters:
            if "clusters_unavailable" in pack.warnings:
                return None
            action = RetrievalAction(
                action="episodic_clusters",
                k=self.cluster_k,
                reason="coverage_clusters",
            )
            return ControllerDecision(actions=[action], done=False)

        if self.clusters_first and coverage.needs_episode_summaries:
            ids = self._cluster_episode_ids(pack)
            if ids:
                action = RetrievalAction(
                    action="fetch_episode_summaries",
                    ids=ids,
                    reason="coverage_episode_summaries",
                )
                return ControllerDecision(actions=[action], done=False)

        if self.semantic_first and not pack.graph:
            scope = self._next_predicate_scope(pack)
            try:
                node_id = ensure_user_subject(user_id)
            except Exception:
                node_id = user_id
            if scope:
                action = RetrievalAction(
                    action="graph_neighbors",
                    node_id=node_id,
                    predicate_scope=scope,
                    k=self.max_items_per_type,
                    reason="graph_predicate_scope",
                )
                return ControllerDecision(actions=[action], done=False)
            if not self._has_graph_action(pack, scoped=False):
                action = RetrievalAction(
                    action="graph_neighbors",
                    node_id=node_id,
                    predicate_scope=None,
                    k=self.max_items_per_type,
                    reason="graph_broaden",
                )
                return ControllerDecision(actions=[action], done=False)

        return None

    def _cluster_episode_ids(self, pack: ContextPack) -> List[str]:
        ids: List[str] = []
        seen = set()
        for item in pack.episodes:
            if not self._is_cluster_summary(item):
                continue
            for eid in item.get("episode_ids", [])[: self.max_items_per_type]:
                if eid in seen:
                    continue
                seen.add(eid)
                ids.append(eid)
                if len(ids) >= self.max_items_per_type:
                    return ids
        return ids

    @staticmethod
    def _is_cluster_summary(item: Any) -> bool:
        return isinstance(item, dict) and "episode_ids" in item and "latest_timestamp" in item

    def _next_predicate_scope(self, pack: ContextPack) -> List[str]:
        candidates = self._predicate_candidates(pack)
        if not candidates:
            return []
        used = set()
        for step in pack.steps:
            action = step.get("action") if isinstance(step, dict) else None
            if not isinstance(action, dict):
                continue
            if action.get("action") != "graph_neighbors":
                continue
            scope = action.get("predicate_scope") or []
            for pred in scope:
                used.add(str(pred))
        remaining = [p for p in candidates if p not in used]
        return remaining[: self.graph_predicate_limit]

    def _predicate_candidates(self, pack: ContextPack) -> List[str]:
        weighted = sorted(
            self.predicate_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )
        ordered: List[str] = []
        seen = set()
        for pred, _weight in weighted:
            if pred in seen:
                continue
            seen.add(pred)
            ordered.append(pred)

        counts: Dict[str, int] = {}
        for fact in pack.facts:
            pred = None
            if isinstance(fact, dict):
                pred = fact.get("predicate")
            else:
                pred = getattr(fact, "predicate", None)
            pred = self._sanitize_predicate(str(pred or ""))
            if not pred:
                continue
            counts[pred] = counts.get(pred, 0) + 1
        if counts:
            for pred, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                if pred in seen:
                    continue
                seen.add(pred)
                ordered.append(pred)

        terms = extract_query_terms(pack.query_text) if extract_query_terms else []
        terms = [str(t).lower() for t in terms if t]
        if not terms or not ordered:
            return ordered

        def match_score(pred: str) -> int:
            pl = pred.lower()
            return 1 if any(t in pl for t in terms) else 0

        ordered = sorted(ordered, key=lambda p: (-match_score(p), ordered.index(p)))
        return ordered

    def _normalize_predicate_weights(
        self,
        weights: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        if not weights:
            return {}
        out: Dict[str, float] = {}
        for key, val in weights.items():
            pred = self._sanitize_predicate(str(key or ""))
            if not pred:
                continue
            try:
                out[pred] = float(val)
            except Exception:
                continue
        return out

    def _sanitize_predicate(self, predicate: str) -> str:
        cleaned = re.sub(r"[^A-Z0-9_]", "_", predicate.upper()).strip("_")
        if not cleaned:
            return ""
        if not cleaned[0].isalpha():
            cleaned = f"REL_{cleaned}"
        return cleaned

    @staticmethod
    def _has_graph_action(pack: ContextPack, scoped: Optional[bool] = None) -> bool:
        for step in pack.steps:
            action = step.get("action") if isinstance(step, dict) else None
            if not isinstance(action, dict):
                continue
            if action.get("action") != "graph_neighbors":
                continue
            if scoped is None:
                return True
            has_scope = bool(action.get("predicate_scope"))
            if scoped == has_scope:
                return True
        return False

    async def _evaluate_and_extract(
        self,
        pack: ContextPack,
        query_text: str,
        eval_round: int,
    ) -> Optional[ContextPack]:
        if not self.extract_snippets:
            return None

        chunks = self._collect_fact_chunks(pack)
        if not chunks:
            return None

        chunks = chunks[: self.max_eval_chunks]
        decision = await self._evaluate_evidence(query_text, chunks)
        pack.steps.append({"step": eval_round, "evidence": decision.model_dump()})

        if not decision.has_sufficient_evidence:
            heuristic = self._extract_snippets_rule_based(query_text, chunks)
            if heuristic:
                return self._pack_snippets(pack, heuristic)
            if decision.next_query and eval_round < self.max_eval_rounds:
                await self._retrieve_with_query(pack, decision.next_query)
                return None
            empty_pack = ContextPack(user_id=pack.user_id, query_text=pack.query_text)
            empty_pack.warnings.append("insufficient_evidence")
            return empty_pack

        relevant_chunks = [
            c for c in chunks if not decision.relevant_chunk_ids or c["id"] in decision.relevant_chunk_ids
        ]
        if not relevant_chunks:
            empty_pack = ContextPack(user_id=pack.user_id, query_text=pack.query_text)
            empty_pack.warnings.append("insufficient_evidence")
            return empty_pack

        if self.extract_snippets:
            relevant_chunks = await self._expand_chunks_by_document(pack, relevant_chunks)

        snippets = await self._extract_snippets(query_text, relevant_chunks)
        if not snippets:
            heuristic = self._extract_snippets_rule_based(query_text, relevant_chunks)
            if heuristic:
                return self._pack_snippets(pack, heuristic)
            empty_pack = ContextPack(user_id=pack.user_id, query_text=pack.query_text)
            empty_pack.warnings.append("insufficient_evidence")
            return empty_pack

        return self._pack_snippets(pack, snippets)

    async def _expand_chunks_by_document(
        self,
        pack: ContextPack,
        relevant_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        If any chunk is relevant, include other chunks from the same document path.
        """
        paths = {c.get("path") for c in relevant_chunks if c.get("path")}
        if not paths:
            return relevant_chunks
        extras = []
        for fact in pack.facts:
            obj = fact.get("object") if isinstance(fact, dict) else getattr(fact, "object", None)
            if not isinstance(obj, dict):
                continue
            path = obj.get("path")
            if path not in paths:
                continue
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            extras.append(
                {
                    "id": fact.get("id") if isinstance(fact, dict) else getattr(fact, "id", None),
                    "title": obj.get("title"),
                    "path": path,
                    "text": text,
                }
            )
        combined = {c["id"]: c for c in relevant_chunks}
        for extra in extras:
            if extra.get("id") not in combined:
                combined[extra.get("id")] = extra
        return list(combined.values())

    async def _retrieve_with_query(self, pack: ContextPack, query_text: str) -> None:
        query_embedding = await self.env.get_query_embedding(query_text)
        if not query_embedding:
            return
        items = await self._safe(
            self.env.search_semantic(
                user_id=pack.user_id,
                query_embedding=query_embedding,
                k=self.max_items_per_type,
                filters=None,
            ),
            default=[],
        )
        if items:
            pack.facts = merge_unique(pack.facts, items, self.max_items_per_type)
        pack.steps.append({"step": "eval_retrieve", "query": query_text, "counts": pack.snapshot()["counts"]})

    def _collect_fact_chunks(self, pack: ContextPack) -> List[Dict[str, Any]]:
        chunks = []
        for fact in pack.facts:
            obj = fact.get("object") if isinstance(fact, dict) else getattr(fact, "object", None)
            if not isinstance(obj, dict):
                continue
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            chunks.append(
                {
                    "id": fact.get("id") if isinstance(fact, dict) else getattr(fact, "id", None),
                    "title": obj.get("title"),
                    "path": obj.get("path"),
                    "text": text,
                }
            )
        return chunks

    async def _evaluate_evidence(self, query_text: str, chunks: List[Dict[str, Any]]) -> "EvidenceEvaluation":
        payload = json.dumps({"query": query_text, "chunks": chunks})[: self.max_return_chars]
        prompt = (
            "You are UMA RLM evidence evaluator.\n"
            "Decide if the provided chunks contain enough evidence to answer the query.\n"
            "Return JSON only.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "has_sufficient_evidence": true|false,\n'
            '  "relevant_chunk_ids": ["..."],\n'
            '  "next_query": "string or null",\n'
            '  "reason": "short"\n'
            "}\n\n"
            f"DATA:\n{payload}\n"
        )
        raw = await asyncio.wait_for(
            self.llm.generate(
                [{"role": "system", "content": prompt}],
                max_tokens=self.llm_max_tokens,
                temperature=0.0,
                format="json",
            ),
            timeout=self.timeout_s,
        )
        return EvidenceEvaluation.from_json(raw)

    async def _extract_snippets(self, query_text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload = json.dumps({"query": query_text, "chunks": chunks})[: self.max_return_chars]
        prompt = (
            "You are UMA RLM snippet extractor.\n"
            "Extract only exact spans from chunks that answer the query.\n"
            "Return JSON only. Do not rewrite. If nothing, return empty list.\n\n"
            "JSON schema:\n"
            "{ \"snippets\": [ {\"chunk_id\":\"...\",\"text\":\"...\"} ] }\n\n"
            f"DATA:\n{payload}\n"
        )
        raw = await asyncio.wait_for(
            self.llm.generate(
                [{"role": "system", "content": prompt}],
                max_tokens=self.llm_max_tokens,
                temperature=0.0,
                format="json",
            ),
            timeout=self.timeout_s,
        )
        result = ExtractionResult.from_json(raw)
        if not result.snippets:
            return []
        chunk_map = {c["id"]: c for c in chunks}
        cleaned = []
        seen = set()
        for snip in result.snippets:
            chunk = chunk_map.get(snip.get("chunk_id"))
            text = (snip.get("text") or "").strip()
            if not chunk or not text:
                continue
            key = (snip.get("chunk_id"), text[: self.max_snippet_chars])
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(
                {
                    "chunk_id": snip.get("chunk_id"),
                    "text": text[: self.max_snippet_chars],
                    "title": chunk.get("title"),
                    "path": chunk.get("path"),
                }
            )
        return cleaned

    def _extract_snippets_rule_based(
        self,
        query_text: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not extract_query_terms:
            return []
        terms = extract_query_terms(query_text)
        if not terms:
            return []
        snippets = []
        seen = set()
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            lowered = text.lower()
            if not any(t in lowered for t in terms):
                continue
            for line in text.splitlines():
                if any(t in line.lower() for t in terms):
                    snippet = line.strip()
                    if not snippet:
                        continue
                    snippet = snippet[: self.max_snippet_chars]
                    key = (chunk.get("id"), snippet)
                    if key in seen:
                        continue
                    seen.add(key)
                    snippets.append(
                        {
                            "chunk_id": chunk.get("id"),
                            "text": snippet,
                            "title": chunk.get("title"),
                            "path": chunk.get("path"),
                        }
                    )
                    break
        return snippets

    def _pack_snippets(self, pack: ContextPack, snippets: List[Dict[str, Any]]) -> ContextPack:
        pack.facts = [
            {
                "subject": pack.user_id,
                "predicate": "document_snippet",
                "object": {
                    "title": s.get("title"),
                    "path": s.get("path"),
                    "text": s.get("text"),
                },
                "meta": {"source_chunk_id": s.get("chunk_id")},
            }
            for s in snippets
        ]
        pack.episodes = []
        pack.skills = []
        pack.graph = []
        return pack

    async def _safe(self, coro, default):
        try:
            return await asyncio.wait_for(coro, timeout=self.timeout_s)
        except Exception:
            logger.exception("RLMController env call failed")
            return default

    def _build_snippet_summary(self, pack: ContextPack) -> Dict[str, str]:
        """
        Build a compact, token-safe summary of retrieved items to ground decisions.
        """
        return {
            "episodes": self._summarize_items(pack.episodes, "episodic", 4, 800),
            "facts": self._summarize_items(pack.facts, "semantic", 6, 600),
            "skills": self._summarize_items(pack.skills, "procedural", 4, 400),
            "graph": self._summarize_items(pack.graph, "graph", 6, 600),
        }

    def _summarize_items(self, items: List[Any], kind: str, max_items: int, max_chars: int) -> str:
        snippets: List[str] = []
        for it in items[:max_items]:
            try:
                if kind == "episodic":
                    if isinstance(it, dict):
                        text = it.get("summary") or repr(it)
                    else:
                        text = getattr(it, "summary", "") or repr(it)
                elif kind == "semantic":
                    if isinstance(it, dict):
                        pred = it.get("predicate", "")
                        obj = it.get("object", "")
                    else:
                        pred = getattr(it, "predicate", "")
                        obj = getattr(it, "object", "")
                    text = f"{pred} {obj}".strip() or repr(it)
                elif kind == "procedural":
                    text = getattr(it, "name", "") or repr(it)
                elif kind == "graph":
                    if isinstance(it, dict):
                        labels = ",".join(it.get("labels", []) or [])
                        props = it.get("properties", {}) or {}
                        text = f"{labels} {props}".strip()
                    else:
                        text = repr(it)
                else:
                    text = repr(it)
                text = (text or "").strip()
                if text:
                    snippets.append(text)
            except Exception:
                logger.exception("RLMController: failed to summarize %s item", kind)
        joined = "; ".join(snippets)
        return joined[:max_chars]

    def _predicate_counts(self, pack: ContextPack) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for fact in pack.facts:
            try:
                pred = getattr(fact, "predicate", None)
                if not pred and isinstance(fact, dict):
                    pred = fact.get("predicate")
                if not pred:
                    continue
                pred = str(pred).upper()
                counts[pred] = counts.get(pred, 0) + 1
            except Exception:
                logger.exception("RLMController: predicate count failed")
        return counts

    def _fallback_graph_action(self, user_id: str, pack: ContextPack) -> ControllerDecision:
        """
        Deterministic fallback: if graph is empty but predicates exist,
        expand graph around the user node with top predicates.
        """
        if pack.graph:
            return ControllerDecision(done=False)

        preds = self._predicate_counts(pack)
        if not preds:
            return ControllerDecision(done=False)

        top_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:2]
        predicate_scope = [p for p, _ in top_preds]
        try:
            node_id = ensure_user_subject(user_id)
        except Exception:
            node_id = user_id

        action = RetrievalAction(
            action="graph_neighbors",
            node_id=node_id,
            predicate_scope=predicate_scope,
            k=min(8, self.max_items_per_type),
            reason="fallback_graph_expansion",
        )
        return ControllerDecision(actions=[action], done=False)

    def _truncate_items(self, items: List[Any]) -> List[Any]:
        """
        Enforce per-item text size limits to keep environment returns bounded.
        """
        if not items or not self.max_return_chars:
            return items
        return [self._truncate_item(it) for it in items]

    def _truncate_item(self, item: Any) -> Any:
        max_chars = int(self.max_return_chars)
        if isinstance(item, str):
            return item[:max_chars]
        if isinstance(item, dict):
            return self._truncate_dict(item, max_chars)
        # Best-effort: truncate common text fields on objects.
        for attr in ("summary", "raw", "content", "text"):
            if hasattr(item, attr):
                val = getattr(item, attr)
                if isinstance(val, str) and len(val) > max_chars:
                    setattr(item, attr, val[:max_chars])
        return item

    def _truncate_dict(self, data: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str):
                out[k] = v[:max_chars]
            elif isinstance(v, dict):
                out[k] = self._truncate_dict(v, max_chars)
            else:
                out[k] = v
        return out

    

    


class EvidenceEvaluation(BaseModel):
    has_sufficient_evidence: bool = False
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    next_query: Optional[str] = None
    reason: str = ""

    @classmethod
    def from_json(cls, raw: str) -> "EvidenceEvaluation":
        try:
            return cls.model_validate_json(raw)
        except ValidationError as exc:
            cleaned = raw.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                    return cls.model_validate(parsed)
                except Exception:
                    pass
            raise ValueError(f"Invalid evidence JSON: {exc}")


class ExtractionResult(BaseModel):
    snippets: List[Dict[str, str]] = Field(default_factory=list)

    @classmethod
    def from_json(cls, raw: str) -> "ExtractionResult":
        try:
            return cls.model_validate_json(raw)
        except ValidationError as exc:
            cleaned = raw.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                    return cls.model_validate(parsed)
                except Exception:
                    pass
            raise ValueError(f"Invalid extraction JSON: {exc}")
