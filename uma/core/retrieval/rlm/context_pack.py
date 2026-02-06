# uma/core/retrieval/rlm/context_pack.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ContextPack:
    """
    Developer-facing context bundle.

    Note: this dataclass is intentionally mutable so the controller can
    accumulate results across recursive retrieval steps.

    This is NOT a prompt.
    This is structured memory data that downstream agents
    can inject into prompts however they choose.

    Evidence accounting (RLM v2)
    ---------------------------
    - Track seen IDs per store
    - Track per-step novelty
    - Track predicate offsets for semantic expansion
    """
    user_id: str
    query_text: str
    owner_type: Optional[str] = None
    owner_id: Optional[str] = None
    agent_id: Optional[str] = None
    

    # Memory layers
    working_memory: List[Any] = field(default_factory=list)
    episodes: List[Any] = field(default_factory=list)
    facts: List[Any] = field(default_factory=list)
    chunks: List[Any] = field(default_factory=list)
    skills: List[Any] = field(default_factory=list)
    graph: List[Any] = field(default_factory=list)

    # Controller trace
    steps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    coverage: Optional[Any] = None

    # --- Evidence accounting ---
    seen_fact_ids: Set[str] = field(default_factory=set)
    seen_chunk_ids: Set[str] = field(default_factory=set)
    seen_episode_ids: Set[str] = field(default_factory=set)
    seen_skill_ids: Set[str] = field(default_factory=set)
    seen_graph_ids: Set[str] = field(default_factory=set)

    novelty_history: List[Dict[str, int]] = field(default_factory=list)
    predicate_offsets: Dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        """Safe summary for logs / telemetry."""
        return {
            "user_id": self.user_id,
            "query": self.query_text,
            "counts": {
                "wm": len(self.working_memory),
                "episodes": len(self.episodes),
                "facts": len(self.facts),
                "chunks": len(self.chunks),
                "skills": len(self.skills),
                "graph": len(self.graph),
            },
            "seen": {
                "facts": len(self.seen_fact_ids),
                "chunks": len(self.seen_chunk_ids),
                "episodes": len(self.seen_episode_ids),
                "skills": len(self.seen_skill_ids),
                "graph": len(self.seen_graph_ids),
            },
            "novelty_steps": len(self.novelty_history),
            "steps": len(self.steps),
            "warnings": self.warnings,
        }

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    def record_seen(self) -> None:
        try:
            self.seen_fact_ids.update(_collect_ids(self.facts))
            self.seen_chunk_ids.update(_collect_ids(self.chunks))
            self.seen_episode_ids.update(_collect_ids(self.episodes))
            self.seen_skill_ids.update(_collect_ids(self.skills))
            self.seen_graph_ids.update(_collect_ids(self.graph))
            logger.debug(
                "ContextPack.record_seen: facts=%d chunks=%d episodes=%d skills=%d graph=%d",
                len(self.seen_fact_ids),
                len(self.seen_chunk_ids),
                len(self.seen_episode_ids),
                len(self.seen_skill_ids),
                len(self.seen_graph_ids),
            )
        except Exception:
            logger.exception("ContextPack.record_seen failed")

    def compute_novelty(self, items: List[Any], store: str) -> int:
        if not items:
            return 0
        store = self._normalize_store(store)
        new_ids = _collect_ids(items)
        if not new_ids:
            return 0
        seen = self._seen_set(store)
        return sum(1 for i in new_ids if i not in seen)

    def apply_novelty(self, items: List[Any], store: str) -> Dict[str, int]:
        store = self._normalize_store(store)
        novelty = self.compute_novelty(items, store)
        ids = _collect_ids(items)
        self._seen_set(store).update(ids)
        payload = {"facts": 0, "chunks": 0, "episodes": 0, "skills": 0, "graph": 0}
        payload[store] = novelty
        self.novelty_history.append(payload)
        logger.debug("ContextPack.apply_novelty: store=%s novelty=%d", store, novelty)
        return payload

    def get_predicate_offset(self, predicate: str) -> int:
        return int(self.predicate_offsets.get(predicate.upper(), 0))

    def bump_predicate_offset(self, predicate: str, delta: int) -> int:
        key = predicate.upper()
        self.predicate_offsets[key] = self.get_predicate_offset(key) + int(delta)
        return self.predicate_offsets[key]

    def _seen_set(self, store: str) -> Set[str]:
        mapping = {
            "facts": self.seen_fact_ids,
            "chunks": self.seen_chunk_ids,
            "episodes": self.seen_episode_ids,
            "skills": self.seen_skill_ids,
            "graph": self.seen_graph_ids,
        }
        if store not in mapping:
            raise KeyError(f"Unknown store: {store}")
        return mapping[store]

    @staticmethod
    def _normalize_store(store: str) -> str:
        store = (store or "").strip().lower()
        if store not in {"facts", "chunks", "episodes", "skills", "graph"}:
            raise KeyError(f"Unknown store: {store}")
        return store


def _collect_ids(items: List[Any]) -> Set[str]:
    out: Set[str] = set()
    for it in items or []:
        try:
            if isinstance(it, dict) and it.get("id"):
                out.add(str(it["id"]))
            elif hasattr(it, "id"):
                out.add(str(it.id))
        except Exception:
            logger.exception("_collect_ids failed")
    return out
