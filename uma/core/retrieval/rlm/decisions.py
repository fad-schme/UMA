from __future__ import annotations

"""
uma.core.retrieval.rlm.decisions
================================

This module defines the *only* action space the RLM controller is allowed to use.

Design goals
------------
- Store-native: actions map directly to safe, bounded memory store operations.
- Production-safe: strict validation of parameters, no arbitrary queries.
- No backwards-compat: UMA-RLM is v1 in active development.

Action space
------------
The controller may:
- vector-search semantic/episodic/procedural (bounded k)
- fetch by ids (bounded list length)
- graph neighbor expansion (bounded k, bounded depth)
- episodic cluster summaries (bounded k)
- stop
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional, NoReturn

from pydantic import BaseModel, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)


MAX_FACT_IDS = 50

ActionType = Literal[
    "search_semantic",
    "fetch_more_facts",              # ← NEW
    "search_episodic",
    "episodic_clusters",
    "fetch_episode_clusters",
    "search_procedural",
    "fetch_facts",
    "fetch_chunks",
    "search_chunks",
    "graph_neighbors",
    "expand_graph",
    "stop",
]


class RetrievalAction(BaseModel):
    """
    A single bounded retrieval action.

    IMPORTANT:
    - Each action has a strict contract.
    - Invalid combinations are rejected.
    """

    action: ActionType
    reason: str = ""

    # Common bounds
    k: Optional[int] = Field(default=None, ge=1, le=500)

    # Semantic / procedural
    filters: Optional[Dict[str, Any]] = None

    # Semantic refinement (NEW)
    predicate: Optional[str] = None
    owner_type: Optional[Literal["user", "agent"]] = None

    # Episodic
    time_range: Optional[Dict[str, Any]] = None
    min_salience: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Fetch-by-id
    ids: Optional[List[str]] = None

    # Graph
    node_id: Optional[str] = None
    predicate_scope: Optional[List[str]] = None
    depth: Optional[int] = Field(default=None, ge=1, le=3)
    subject: Optional[str] = None
    direction: Optional[str] = None
    hops: Optional[int] = Field(default=None, ge=1, le=3)

    # Conflict resolution
    fact_ids: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_action(self) -> "RetrievalAction":
        """
        Enforce per-action contracts.

        This prevents the controller from emitting ambiguous or unsafe actions.
        """
        def _raise(msg: str) -> "NoReturn":
            logger.error("RetrievalAction.validate_action failed: %s", msg)
            raise ValueError(msg)

        a = self.action

        # --- STOP ---
        if a == "stop":
            if any(
                [
                    self.k,
                    self.filters,
                    self.time_range,
                    self.ids,
                    self.node_id,
                    self.predicate_scope,
                    self.depth,
                ]
            ):
                _raise("stop action must not include any parameters")
            return self

        # --- SEARCH SEMANTIC ---
        if a == "search_semantic":
            if self.k is None:
                _raise("search_semantic requires k")
            if self.ids or self.node_id:
                _raise("search_semantic does not accept ids or node_id")
            return self

        # --- SEARCH EPISODIC ---
        if a == "search_episodic":
            if self.k is None:
                _raise("search_episodic requires k")
            if self.ids or self.node_id:
                _raise("search_episodic does not accept ids or node_id")
            return self

        # --- EPISODIC CLUSTERS ---
        if a == "episodic_clusters":
            if self.k is None:
                _raise("episodic_clusters requires k")
            if self.filters or self.ids or self.node_id:
                _raise("episodic_clusters does not accept filters, ids, or node_id")
            return self

        if a == "fetch_episode_clusters":
            if self.k is None:
                _raise("fetch_episode_clusters requires k")
            if self.filters or self.ids or self.node_id:
                _raise("fetch_episode_clusters does not accept filters, ids, or node_id")
            if self.min_salience is not None and not (0 <= self.min_salience <= 1):
                _raise("min_salience must be between 0 and 1")
            return self

        # --- SEARCH PROCEDURAL ---
        if a == "search_procedural":
            if self.k is None:
                _raise("search_procedural requires k")
            if self.ids or self.node_id:
                _raise("search_procedural does not accept ids or node_id")
            return self

        # --- FETCH FACTS ---
        if a == "fetch_facts":
            if not self.ids:
                _raise("fetch_facts requires ids")
            if self.k or self.filters or self.node_id:
                _raise("fetch_facts does not accept k, filters, or node_id")
            return self

        # --- FETCH CHUNKS ---
        if a == "fetch_chunks":
            if not self.ids:
                _raise("fetch_chunks requires ids")
            if self.k or self.filters or self.node_id or self.time_range:
                _raise("fetch_chunks does not accept k, filters, node_id, or time_range")
            return self

        # --- SEARCH CHUNKS ---
        if a == "search_chunks":
            if self.k is None:
                _raise("search_chunks requires k")
            if self.ids or self.node_id or self.time_range:
                _raise("search_chunks does not accept ids, node_id, or time_range")
            return self
        
        # --- FETCH MORE FACTS (predicate-scoped semantic expansion) ---
        if a == "fetch_more_facts":
            if self.k is None:
                _raise("fetch_more_facts requires k")
            if not self.predicate or not isinstance(self.predicate, str):
                _raise("fetch_more_facts requires a non-empty predicate")
            if self.ids or self.node_id or self.time_range:
                _raise("fetch_more_facts does not accept ids, node_id, or time_range")
            if self.owner_type and self.owner_type not in {"user", "agent"}:
                _raise("owner_type must be one of: user, agent")
            return self
        
        if a == "expand_graph":
            if not self.subject or not isinstance(self.subject, str):
                _raise("expand_graph requires a non-empty subject")
            if self.k is None:
                _raise("expand_graph requires k")
            if self.filters or self.ids or self.time_range:
                _raise("expand_graph does not accept filters, ids, or time_range")
            if self.node_id:
                _raise("expand_graph does not accept node_id")
            if self.direction:
                dir_val = str(self.direction).lower()
                if dir_val not in {"inbound", "outbound", "both"}:
                    _raise("direction must be one of: inbound, outbound, both")
                self.direction = dir_val
            if self.hops is None:
                self.hops = 1
            return self


        # --- GRAPH NEIGHBORS ---
        if a == "graph_neighbors":
            if not self.node_id:
                _raise("graph_neighbors requires node_id")
            if self.k is None:
                _raise("graph_neighbors requires k")
            if self.depth is None:
                self.depth = 1
            if self.predicate_scope:
                if not isinstance(self.predicate_scope, list):
                    _raise("predicate_scope must be a list")
                if len(self.predicate_scope) > 20:
                    _raise("predicate_scope too large (max 20)")
            return self

        _raise(f"Unknown action: {a}")


class ControllerDecision(BaseModel):
    """
    Output schema produced by the controller.

    - actions: ordered list of RetrievalAction
    - done: if true, controller terminates immediately
    """

    actions: List[RetrievalAction] = Field(default_factory=list)
    done: bool = False

    @classmethod
    def from_json(cls, raw: str) -> "ControllerDecision":
        """
        Parse controller output robustly.

        Controller MUST emit JSON. We attempt recovery but do not
        accept structurally invalid decisions.
        """
        try:
            return cls.model_validate_json(raw)
        except ValidationError:
            cleaned = (raw or "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(cleaned[start : end + 1])
                return cls.model_validate(parsed)
            logger.exception("ControllerDecision.from_json failed to parse decision.")
            raise


def deterministic_decision(
    pack: Any,
    coverage: Any,
    *,
    cfg: Dict[str, Any],
) -> Optional[ControllerDecision]:
    """
    Deterministic decision policy for RLM retrieval.

    This function contains no I/O. It translates the current state
    (ContextPack + CoverageReport) into a bounded list of RetrievalAction(s).
    """
    actions: List[RetrievalAction] = []

    max_items_per_type = int(cfg.get("max_items_per_type", 30))
    cluster_k = int(cfg.get("cluster_k", 3))
    salience_threshold = float(cfg.get("salience_threshold", 0.6))
    graph_predicate_limit = int(cfg.get("graph_predicate_limit", 2))

    chunk_fallback_enabled = bool(cfg.get("chunk_fallback_enabled", True))
    chunk_fallback_k_multiplier = max(1, int(cfg.get("chunk_fallback_k_multiplier", 2)))

    # --- Semantic ---
    if getattr(coverage, "needs_semantic", False):
        facts = getattr(pack, "facts", []) or []
        if facts:
            next_scope = cfg.get("next_predicate_scope")
            preds = next_scope(pack, graph_predicate_limit) if callable(next_scope) else []
            if preds:
                predicate = preds[0]
                offset = getattr(pack, "get_predicate_offset")(predicate)
                actions.append(
                    RetrievalAction(
                        action="fetch_more_facts",
                        predicate=predicate,
                        k=max_items_per_type,
                        filters={"offset": offset},
                        owner_type=getattr(pack, "owner_type", None),
                    )
                )
                getattr(pack, "bump_predicate_offset")(predicate, max_items_per_type)
        else:
            actions.append(
                RetrievalAction(
                    action="search_semantic",
                    k=max_items_per_type,
                    owner_type=getattr(pack, "owner_type", None),
                )
            )

    # --- Chunk fallback (at most once) ---
    if chunk_fallback_enabled and not bool(getattr(pack, "chunk_fallback_used", False)):
        chunks = getattr(pack, "chunks", []) or []
        if len(chunks) == 0:
            actions.append(
                RetrievalAction(
                    action="search_chunks",
                    k=min(max_items_per_type * chunk_fallback_k_multiplier, 120),
                    owner_type=getattr(pack, "owner_type", None),
                )
            )
            try:
                setattr(pack, "chunk_fallback_used", True)
            except Exception:
                logger.exception("deterministic_decision: failed to mark chunk_fallback_used")
                raise

    # --- Episodic / clusters ---
    if getattr(coverage, "needs_clusters", False):
        episodes = getattr(pack, "episodes", []) or []
        has_cluster = any(isinstance(ep, dict) and "episode_ids" in ep for ep in episodes)
        actions.append(
            RetrievalAction(
                action="fetch_episode_clusters",
                k=cluster_k,
                time_range=None,
                min_salience=salience_threshold,
                owner_type=getattr(pack, "owner_type", None),
            )
        )
        if not has_cluster and len(getattr(pack, "steps", []) or []) >= 2:
            actions.append(
                RetrievalAction(
                    action="search_episodic",
                    k=max_items_per_type,
                    owner_type=getattr(pack, "owner_type", None),
                )
            )

    # --- Graph last-mile ---
    graph = getattr(pack, "graph", []) or []
    facts = getattr(pack, "facts", []) or []
    if not graph and facts and not getattr(coverage, "needs_semantic", False) and not getattr(coverage, "needs_clusters", False):
        next_scope = cfg.get("next_predicate_scope")
        predicate_scope = next_scope(pack, graph_predicate_limit) if callable(next_scope) else []
        if predicate_scope and getattr(pack, "owner_type", None) == "user":
            actions.append(
                RetrievalAction(
                    action="expand_graph",
                    subject=getattr(pack, "user_id", None),
                    predicate=predicate_scope[0],
                    hops=1,
                    direction="outbound",
                    k=min(max_items_per_type, 20),
                    owner_type=getattr(pack, "owner_type", None),
                )
            )

    return ControllerDecision(actions=actions) if actions else None


def next_predicate_scope(
    *,
    facts: List[Any],
    predicate_weights: Optional[Dict[str, float]],
    graph_predicate_limit: int,
) -> List[str]:
    """
    Determine a bounded predicate exploration scope based on:
    - configured predicate weights (highest first)
    - observed predicate frequency in current facts
    """
    ordered: List[str] = []

    if predicate_weights:
        for p, _ in sorted(predicate_weights.items(), key=lambda kv: (-float(kv[1]), str(kv[0]))):
            ordered.append(str(p).upper())

    counts: Dict[str, int] = {}
    for f in facts or []:
        pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
        if pred:
            key = str(pred).upper()
            counts[key] = counts.get(key, 0) + 1

    for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if p not in ordered:
            ordered.append(p)

    if not ordered:
        ordered = ["RELATED_TO"]

    try:
        limit = max(1, int(graph_predicate_limit))
    except Exception:
        limit = 2
    return ordered[:limit]


async def execute_action(
    *,
    env: Any,
    user_subject: str,
    action: RetrievalAction,
    query_embedding: List[float],
    query_text: Optional[str],
    owner_type: str,
    owner_id: Optional[str],
    default_k: int,
    trace_id: Optional[str] = None,
) -> List[Any]:
    """
    Execute a RetrievalAction by delegating to domain cores and the RLM environment.

    This keeps controller.py focused on orchestration, budgets, pack mutation, and logging.
    """
    k = int(action.k) if action.k else int(default_k)
    lane_owner_type = action.owner_type or owner_type
    lane_owner_id: Optional[str] = owner_id
    if lane_owner_type == "agent" and lane_owner_id is None:
        lane_owner_id = getattr(env, "_agent_id", None)

    if action.action == "fetch_more_facts":
        offset = int(action.filters.get("offset", 0)) if action.filters else 0
        if trace_id is not None:
            logger.info(
                "RLM_FETCH_MORE_FACTS trace_id=%s predicate=%s offset=%s k=%s",
                trace_id,
                action.predicate,
                offset,
                k,
            )
        return await env.fetch_more_facts(
            user_id=user_subject,
            predicate=action.predicate,
            k=k,
            offset=offset,
            owner_type=lane_owner_type,
            owner_id=lane_owner_id,
        )

    if action.action == "fetch_facts":
        return await env.fetch_facts_by_ids(
            user_id=user_subject,
            ids=action.ids or [],
            owner_type=lane_owner_type,
            owner_id=lane_owner_id,
        )

    if action.action == "fetch_chunks":
        return await env.fetch_chunks(
            user_id=user_subject,
            ids=action.ids or [],
            owner_type=lane_owner_type,
            owner_id=lane_owner_id,
        )

    if action.action == "search_chunks":
        chunk_core = getattr(getattr(env, "_memory", None), "chunk_core", None)
        if chunk_core is None:
            chunk_core = getattr(env, "_chunk_core", None)
        if chunk_core is None:
            return []

        search_fn = getattr(chunk_core, "search_chunks_for_rlm", None) or getattr(chunk_core, "search_chunks", None)
        if search_fn is None:
            return []

        kwargs = {
            "query_embedding": list(query_embedding),
            "owner_type": lane_owner_type,
            "owner_id": lane_owner_id,
            "k": k,
            "query_text": query_text,
        }
        try:
            return await search_fn(**kwargs)
        except TypeError:
            # Back-compat for adapters that don't accept `query_text`.
            kwargs.pop("query_text", None)
            return await search_fn(**kwargs)

    if action.action == "episodic_clusters":
        return await env.episodic_cluster_summaries(
            user_id=user_subject,
            k=k,
            max_episodes=int(default_k),
            owner_type=lane_owner_type,
            owner_id=lane_owner_id,
        )

    if action.action == "fetch_episode_clusters":
        return await env.fetch_episode_clusters(
            user_id=user_subject,
            k=k,
            max_episodes=int(default_k),
            time_range=action.time_range,
            min_salience=action.min_salience,
            owner_type=lane_owner_type,
            owner_id=lane_owner_id,
        )

    if action.action == "graph_neighbors":
        return await env.graph_neighbors(
            user_id=user_subject,
            node_id=action.node_id,
            predicate_scope=action.predicate_scope,
            depth=int(action.depth or 1),
            k=k,
            owner_type=lane_owner_type,
            owner_id=lane_owner_id,
        )

    if action.action == "expand_graph":
        return await env.expand_graph(
            user_id=user_subject,
            subject=action.subject,
            predicate=action.predicate,
            hops=int(action.hops or 1),
            direction=action.direction,
            k=k,
            owner_type=lane_owner_type,
            owner_id=lane_owner_id,
        )

    return []
