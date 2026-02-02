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
from typing import Any, Dict, List, Literal, Optional

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
    owner_scope: Optional[Literal["user", "project", "agent"]] = None

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
                raise ValueError("stop action must not include any parameters")
            return self

        # --- SEARCH SEMANTIC ---
        if a == "search_semantic":
            if self.k is None:
                raise ValueError("search_semantic requires k")
            if self.ids or self.node_id:
                raise ValueError("search_semantic does not accept ids or node_id")
            return self

        # --- SEARCH EPISODIC ---
        if a == "search_episodic":
            if self.k is None:
                raise ValueError("search_episodic requires k")
            if self.ids or self.node_id:
                raise ValueError("search_episodic does not accept ids or node_id")
            return self

        # --- EPISODIC CLUSTERS ---
        if a == "episodic_clusters":
            if self.k is None:
                raise ValueError("episodic_clusters requires k")
            if self.filters or self.ids or self.node_id:
                raise ValueError("episodic_clusters does not accept filters, ids, or node_id")
            return self

        if a == "fetch_episode_clusters":
            if self.k is None:
                raise ValueError("fetch_episode_clusters requires k")
            if self.filters or self.ids or self.node_id:
                raise ValueError("fetch_episode_clusters does not accept filters, ids, or node_id")
            if self.min_salience is not None and not (0 <= self.min_salience <= 1):
                raise ValueError("min_salience must be between 0 and 1")
            return self

        # --- SEARCH PROCEDURAL ---
        if a == "search_procedural":
            if self.k is None:
                raise ValueError("search_procedural requires k")
            if self.ids or self.node_id:
                raise ValueError("search_procedural does not accept ids or node_id")
            return self

        # --- FETCH FACTS ---
        if a == "fetch_facts":
            if not self.ids:
                raise ValueError("fetch_facts requires ids")
            if self.k or self.filters or self.node_id:
                raise ValueError("fetch_facts does not accept k, filters, or node_id")
            return self
        
        # --- FETCH MORE FACTS (predicate-scoped semantic expansion) ---
        if a == "fetch_more_facts":
            if self.k is None:
                raise ValueError("fetch_more_facts requires k")
            if not self.predicate or not isinstance(self.predicate, str):
                raise ValueError("fetch_more_facts requires a non-empty predicate")
            if self.ids or self.node_id or self.time_range:
                raise ValueError("fetch_more_facts does not accept ids, node_id, or time_range")
            if self.owner_scope and self.owner_scope not in {"user", "project", "agent"}:
                raise ValueError("owner_scope must be one of: user, project, agent")
            return self
        
        if a == "expand_graph":
            if not self.subject or not isinstance(self.subject, str):
                raise ValueError("expand_graph requires a non-empty subject")
            if self.k is None:
                raise ValueError("expand_graph requires k")
            if self.filters or self.ids or self.time_range:
                raise ValueError("expand_graph does not accept filters, ids, or time_range")
            if self.node_id:
                raise ValueError("expand_graph does not accept node_id")
            if self.direction:
                dir_val = str(self.direction).lower()
                if dir_val not in {"inbound", "outbound", "both"}:
                    raise ValueError("direction must be one of: inbound, outbound, both")
                self.direction = dir_val
            if self.hops is None:
                self.hops = 1
            return self


        # --- GRAPH NEIGHBORS ---
        if a == "graph_neighbors":
            if not self.node_id:
                raise ValueError("graph_neighbors requires node_id")
            if self.k is None:
                raise ValueError("graph_neighbors requires k")
            if self.depth is None:
                self.depth = 1
            if self.predicate_scope:
                if not isinstance(self.predicate_scope, list):
                    raise ValueError("predicate_scope must be a list")
                if len(self.predicate_scope) > 20:
                    raise ValueError("predicate_scope too large (max 20)")
            return self

        raise ValueError(f"Unknown action: {a}")


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
