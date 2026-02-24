"""
uma.core.retrieval.selector
============================

MemorySelector — deterministic ranking + truncation for UMA retrieval results.

Responsibilities
----------------
- Deduplicate results by .id when available
- Score + rank per memory type
- Truncate to configured top-k
- Pure function behavior (no I/O, no DB calls)

Output schema
-------------
Always returns:
{
  "working_memory": [...],
  "episodes": [...],
  "facts": [...],
  "skills": [...],
  "graph": [...],
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..utils.dedupe import dedupe_by_id
from .ranking import Ranker

logger = logging.getLogger(__name__)


class MemorySelector:
    """Ranking + truncation logic (pure, safe)."""

    def __init__(
        self,
        max_episodes: int,
        max_facts: int,
        max_skills: int,
        max_graph_items: int,
        max_chunks: int,
        *,
        debug_scores: bool = False,
    ) -> None:
        self.max_episodes = max(1, int(max_episodes))
        self.max_facts = max(1, int(max_facts))
        self.max_chunks = max(1, int(max_chunks))
        self.max_skills = max(1, int(max_skills))
        self.max_graph_items = max(1, int(max_graph_items))
        self.ranker = Ranker(debug_scores=bool(debug_scores))

        logger.info(
            "MemorySelector initialized: episodes=%d facts=%d chunks=%d skills=%d graph=%d",
            self.max_episodes,
            self.max_facts,
            self.max_chunks,
            self.max_skills,
            self.max_graph_items,
        )

    def select(self, raw: Dict[str, List[Any]], *, policy: Optional[Any] = None) -> Dict[str, List[Any]]:
        """Select/rank/truncate each memory type."""
        return {
            "working_memory": raw.get("working_memory", []) or [],
            "episodes": self._select_episodes(raw.get("episodes", []) or [], policy=policy),
            "facts": self._select_facts(raw.get("facts", []) or [], policy=policy),
            "chunks": self._select_chunks(raw.get("chunks", []) or [], policy=policy),
            "skills": self._select_skills(raw.get("skills", []) or [], policy=policy),
            "graph": self._select_graph(raw.get("graph", []) or []),
        }

    # -------------------- Episodic --------------------

    def _select_episodes(self, items: List[Any], *, policy: Optional[Any] = None) -> List[Any]:
        items = self._dedupe(items)
        query_text = str(getattr(policy, "query_text", "") or "") if policy is not None else ""
        ranked = self.ranker.rank_episodes(items, query_text=query_text)
        return self.ranker.truncate(ranked, self.max_episodes)

    # -------------------- Facts --------------------

    # -------------------- Facts --------------------

    def _select_facts(self, items: List[Any], *, policy: Optional[Any] = None) -> List[Any]:
        """
        Rank semantic facts deterministically.

        Ranking logic (v1):
        -------------------
        - Base score = average(salience, confidence)
        - Apply policy-based scope weighting (recall intent, etc.)
        - Explicitly boost agent-scoped facts (promotion payoff)

        NOTE:
        This makes promotion *matter* without overwhelming user context.
        """
        items = self._dedupe(items)
        query_text = str(getattr(policy, "query_text", "") or "") if policy is not None else ""
        ranked = self.ranker.rank_facts(items, query_text=query_text)
        return self.ranker.truncate(ranked, self.max_facts)

    # -------------------- Chunks --------------------

    def _select_chunks(self, items: List[Any], *, policy: Optional[Any] = None) -> List[Any]:
        items = self._dedupe(items)
        if any(isinstance(x, dict) for x in (items or [])):
            raise TypeError(
                "MemorySelector expected Chunk objects for chunk selection; got dict. "
                "Fix chunk store/core to return Chunk only."
            )
        query_text = str(getattr(policy, "query_text", "") or "") if policy is not None else ""
        # Route precedence is a hard constraint; within each route, rerank improves precision.
        ranked = self.ranker.rank_chunks(items, query_text=query_text)
        return self.ranker.truncate(ranked, self.max_chunks)

    # -------------------- Skills --------------------

    def _select_skills(self, items: List[Any], *, policy: Optional[Any] = None) -> List[Any]:
        items = self._dedupe(items)
        query_text = str(getattr(policy, "query_text", "") or "") if policy is not None else ""
        ranked = self.ranker.rank_skills(items, query_text=query_text)
        return self.ranker.truncate(ranked, self.max_skills)

    # -------------------- Graph --------------------

    def _select_graph(self, items: List[Any]) -> List[Any]:
        if not items:
            return []

        def score(node: Any) -> float:
            # Prefer richer nodes (more labels, updated_at presence).
            try:
                labels = (node.get("labels") or []) if isinstance(node, dict) else []
                props = (node.get("properties") or {}) if isinstance(node, dict) else {}
                label_weight = float(len(labels))
                if "Entity" in labels:
                    label_weight += 2.0
                recency = 1.0 if props.get("updated_at") else 0.0
                return label_weight + recency
            except Exception:
                return 0.0

        ranked = sorted(items, key=score, reverse=True)
        return ranked[: self.max_graph_items]

    # -------------------- Dedupe --------------------

    def _dedupe(self, items: List[Any]) -> List[Any]:
        """Deduplicate by `.id` if present, else by Python object id."""
        return dedupe_by_id(items)
