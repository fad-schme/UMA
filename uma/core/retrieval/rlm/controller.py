# uma3/core/retrieval/rlm/controller.py

from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from .context_pack import ContextPack
from .environment import MemoryEnvironment
from .decisions import ControllerDecision, RetrievalAction
from .policy import good_enough, merge_unique
from ...utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)


class RLMController:
    """
    RLMController — Recursive, Bounded Memory Retrieval Controller.

    This controller implements **Recursive Language Model (RLM)-style retrieval**
    for UMA-3. It allows the system to explore long-term memory iteratively
    instead of relying on a single retrieval pass.

    IMPORTANT:
    ----------
    This controller is **not an agent** and **does not answer user queries**.
    This controller performs **memory navigation only**.
    It never plans tasks, reasons about solutions, or decides what to say.
    All agent reasoning and response generation remain outside UMA-3.

    Its sole responsibility is memory navigation:
    deciding which memory stores to query next in order to improve context coverage.

    Role in UMA-3
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
    • Preserve UMA-3's role as a memory system, not an agent
    """

    def __init__(
        self,
        llm: Any,
        env: MemoryEnvironment,
        max_steps: int = 4,
        max_actions_per_step: int = 2,
        max_items_per_type: int = 30,
        llm_max_tokens: int = 300,
        timeout_s: float = 20.0,
        max_env_calls: int = 12,
        max_return_chars: int = 1200,
    ) -> None:
        self.llm = llm
        self.env = env

        self.max_steps = max_steps
        self.max_actions_per_step = max_actions_per_step
        self.max_items_per_type = max_items_per_type
        self.llm_max_tokens = llm_max_tokens
        self.timeout_s = timeout_s
        self.max_env_calls = max_env_calls
        self.max_return_chars = max_return_chars
        self.max_parse_failures = 2

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
        pack = ContextPack(user_id=user_id, query_text=query_text)

        # Always include WM
        pack.working_memory = await self.env.get_working_memory(user_id)

        # Baseline retrieval
        baseline = await self._safe(self.env.retrieve_all(user_id, query_text), default={})
        pack.episodes = baseline.get("episodes", [])
        pack.facts = baseline.get("facts", [])
        pack.skills = baseline.get("skills", [])
        pack.graph = baseline.get("graph", [])

        pack.steps.append({"step": 0, "baseline": pack.snapshot()})

        if good_enough(pack.snapshot()["counts"]):
            return pack

        # Precompute embedding once for semantic/episodic actions
        query_embedding = await self.env.get_query_embedding(query_text)

        total_env_calls = 0
        parse_failures = 0
        for step in range(1, self.max_steps + 1):
            if time.time() - start > self.timeout_s:
                pack.warnings.append("timeout")
                break

            decision = await self._decide(pack)
            if decision is None:
                parse_failures += 1
                pack.warnings.append("controller_parse_failed")
                if parse_failures >= self.max_parse_failures:
                    break
                continue
            pack.steps.append({"step": step, "decision": decision.model_dump()})

            if (decision.done or not decision.actions) and not decision.actions:
                decision = self._fallback_graph_action(user_id, pack)
                if not decision.actions or decision.done:
                    break

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
                        self.env.retrieve_slice(user_id, mem_type, query_text),
                        default=[],
                    )
                elif action_type == "search_semantic":
                    if query_embedding:
                        items = await self._safe(
                            self.env.search_semantic(
                                user_id=user_id,
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
                                user_id=user_id,
                                query_embedding=query_embedding,
                                k=action.k or self.max_items_per_type,
                                time_range=action.time_range,
                            ),
                            default=[],
                        )
                elif action_type == "fetch_facts":
                    items = await self._safe(
                        self.env.fetch_facts_by_ids(user_id=user_id, ids=action.ids or []),
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
                            user_id=user_id,
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
                            user_id=user_id,
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

            if good_enough(pack.snapshot()["counts"]):
                break

        return pack

    async def _decide(self, pack: ContextPack) -> ControllerDecision | None:
        """
        Ask LLM what to fetch next. Must return strict JSON.
        """
        state = pack.snapshot()
        state["snippets"] = self._build_snippet_summary(pack)
        state["predicate_counts"] = self._predicate_counts(pack)
        prompt = (
            "You are UMA-3 Retrieval Controller.\n"
            "Decide which memory to retrieve next using bounded, safe actions.\n"
            "Return JSON ONLY matching ControllerDecision schema.\n\n"
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
            f"STATE:\n{json.dumps(state)}\n"
        )

        try:
            raw = await self.llm.generate(
                [{"role": "system", "content": prompt}],
                max_tokens=self.llm_max_tokens,
                temperature=0.0,
            )
            return ControllerDecision.from_json(raw)
        except Exception:
            logger.exception("RLMController decision failed; stopping")
            return None

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
                    text = getattr(it, "summary", "") or repr(it)
                elif kind == "semantic":
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
