from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

ActionType = Literal[
    "retrieve",
    "expand_graph",
    "search_semantic",
    "search_episodic",
    "episodic_clusters",
    "fetch_facts",
    "fetch_episode_summaries",
    "fetch_episode_transcripts",
    "graph_neighbors",
    "stop",
]

MemoryType = Literal["episodic", "semantic", "procedural", "graph"]


class RetrievalAction(BaseModel):
    action: ActionType
    memory_type: Optional[MemoryType] = None
    k: Optional[int] = Field(default=10, ge=1)
    ids: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    time_range: Optional[Dict[str, Any]] = None
    node_id: Optional[str] = None
    predicate_scope: Optional[List[str]] = None
    reason: str = ""

    @model_validator(mode="after")
    def _validate_action(self) -> "RetrievalAction":
        if self.action in {"retrieve", "expand_graph"}:
            if self.memory_type not in {"episodic", "semantic", "procedural", "graph"}:
                raise ValueError("memory_type is required for retrieve/expand_graph")
        if self.action == "episodic_clusters" and self.k is None:
            raise ValueError("k is required for episodic_clusters")
        if self.action in {"fetch_facts", "fetch_episode_summaries", "fetch_episode_transcripts"}:
            if not self.ids:
                raise ValueError("ids are required for fetch actions")
        if self.action == "graph_neighbors" and not self.node_id:
            raise ValueError("node_id is required for graph_neighbors")
        if self.action == "stop" and self.memory_type is not None:
            raise ValueError("memory_type must be null for stop")
        return self


class ControllerDecision(BaseModel):
    actions: List[RetrievalAction] = Field(default_factory=list)
    done: bool = False

    @classmethod
    def from_json(cls, raw: str) -> "ControllerDecision":
        try:
            return cls.model_validate_json(raw)
        except ValidationError as exc:
            raise ValueError(f"Invalid controller JSON: {exc}")
