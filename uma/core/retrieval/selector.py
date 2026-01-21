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
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class MemorySelector:
    """Ranking + truncation logic (pure, safe)."""

    def __init__(
        self,
        max_episodes: int,
        max_facts: int,
        max_skills: int,
        max_graph_items: int,
    ) -> None:
        self.max_episodes = max(1, int(max_episodes))
        self.max_facts = max(1, int(max_facts))
        self.max_skills = max(1, int(max_skills))
        self.max_graph_items = max(1, int(max_graph_items))

        logger.info(
            "MemorySelector initialized: episodes=%d facts=%d skills=%d graph=%d",
            self.max_episodes,
            self.max_facts,
            self.max_skills,
            self.max_graph_items,
        )

    def select(self, raw: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Select/rank/truncate each memory type."""
        return {
            "working_memory": raw.get("working_memory", []) or [],
            "episodes": self._select_episodes(raw.get("episodes", []) or []),
            "facts": self._select_facts(raw.get("facts", []) or []),
            "skills": self._select_skills(raw.get("skills", []) or []),
            "graph": self._select_graph(raw.get("graph", []) or []),
        }

    # -------------------- Episodic --------------------

    def _select_episodes(self, items: List[Any]) -> List[Any]:
        items = self._dedupe(items)
        now = datetime.now(timezone.utc)

        def score(ep: Any) -> float:
            # Prefer recent episodes. If missing timestamp, lowest score.
            try:
                ts = getattr(ep, "timestamp", None)
                if ts is None:
                    return 0.0
                ts = ts.replace(tzinfo=timezone.utc)
                age_days = (now - ts).total_seconds() / 86400.0
                return max(0.0, 1.0 - age_days / 30.0)  # linear decay over 30d
            except Exception:
                return 0.0

        ranked = sorted(items, key=score, reverse=True)
        return ranked[: self.max_episodes]

    # -------------------- Facts --------------------

    def _select_facts(self, items: List[Any]) -> List[Any]:
        items = self._dedupe(items)

        def score(f: Any) -> float:
            # Prefer high salience + confidence.
            try:
                meta = getattr(f, "meta", {}) or {}
                sal = float(meta.get("salience", 0.0))
            except Exception:
                sal = 0.0

            try:
                conf = float(getattr(f, "confidence", 0.5) or 0.5)
            except Exception:
                conf = 0.5

            return (sal + conf) / 2.0

        ranked = sorted(items, key=score, reverse=True)
        return ranked[: self.max_facts]

    # -------------------- Skills --------------------

    def _select_skills(self, items: List[Any]) -> List[Any]:
        items = self._dedupe(items)

        def diversity(skill: Any) -> int:
            try:
                phrases = set(getattr(skill, "trigger_phrases", []) or [])
                patterns = set(getattr(skill, "trigger_patterns", []) or [])
                return len(phrases) + len(patterns)
            except Exception:
                return 0

        ranked = sorted(items, key=diversity, reverse=True)
        return ranked[: self.max_skills]

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
        seen: Set[Any] = set()
        out: List[Any] = []
        for it in items:
            key = getattr(it, "id", None)
            if key is None:
                key = id(it)
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out