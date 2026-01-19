"""
uma3.core.selector
===================

MemorySelector — Advanced UMA-3 result selector.

This component:
    • Dedupes
    • Applies per-type top-k
    • Scores episodes by recency
    • Scores facts by salience + confidence
    • Scores procedural skills by diversity (trigger variety)
    • Scores graph nodes by semantic richness / recency
    • Ensures stability and safety (no crashes)

Coding Agent Instructions
-------------------------
- Keep this selector pure (no DB or I/O).
- Future extensions may add:
    • Borda rank fusion
    • Cross-memory weighting
    • Hybrid graph-aware ranking
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class MemorySelector:
    """
    UMA-3 advanced selector.

    Performs:
        - Deduplication
        - Recency scoring (episodic)
        - Salience/Confidence scoring (facts)
        - Diversity scoring (skills)
        - Graph-aware ranking (semantic richness + recency)
        - Top-k truncation
    """

    def __init__(
        self,
        max_episodes: int = 3,
        max_facts: int = 10,
        max_skills: int = 2,
        max_graph_items: int = 5,
    ) -> None:
        self.max_episodes = max_episodes
        self.max_facts = max_facts
        self.max_skills = max_skills
        self.max_graph_items = max_graph_items

        logger.info(
            "MemorySelector initialized (episodes=%d, facts=%d, skills=%d, graph=%d)",
            max_episodes,
            max_facts,
            max_skills,
            max_graph_items,
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def _select_graph(self, items: List[Any]) -> List[Any]:
        """
        Graph-aware ranking:

        - Prefer nodes with more labels (semantic richness)
        - Prefer nodes labeled :Entity (facts/object entities)
        - Prefer nodes with 'updated_at' set (more recent semantic updates)
        - Fallback to plain truncation if anything goes wrong

        Each item is expected to be a dict row returned from GraphAdapter, e.g.:

            {
                "n": <backend-specific node>,
                "labels": ["Entity", "Person"],
                "properties": {"id": "user:123", "updated_at": "...", ...}
            }
        """
        if not items:
            return []

        def score(node: Dict[str, Any]) -> float:
            try:
                labels = node.get("labels", []) or []
                props = node.get("properties", {}) or {}

                # Base weight: more labels = more semantic meaning
                label_weight = len(labels)

                # Extra weight if this is a semantic entity
                if "Entity" in labels:
                    label_weight += 2

                # Recency: we don't parse dates here (keep it pure),
                # but presence of updated_at is a hint that the node
                # has been recently touched by UMA-3.
                updated = props.get("updated_at")
                recency = 1.0 if updated else 0.0

                return label_weight + recency
            except Exception:
                # Fail-safe: do not break ranking if one row is malformed
                return 0.0

        ranked = sorted(items, key=score, reverse=True)
        return ranked[: self.max_graph_items]

    def select(self, raw: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Entry point: apply per-type selection logic to raw results.

        Parameters
        ----------
        raw : Dict[str, List[Any]]
            Raw retrieval output from MultiStoreRetriever.

        Returns
        -------
        Dict[str, List[Any]]
            Normalized, ranked, truncated results per memory type.
        """
        return {
            "working_memory": raw.get("working_memory", []),
            "episodes": self._select_episodes(raw.get("episodes", [])),
            "facts": self._select_facts(raw.get("facts", [])),
            "skills": self._select_skills(raw.get("skills", [])),
            "graph": self._select_graph(raw.get("graph", [])),
        }

    # ------------------------------------------------------------------
    # EPISODIC — RECENCY WEIGHTING
    # ------------------------------------------------------------------

    def _select_episodes(self, items: List[Any]) -> List[Any]:
        items = self._dedupe(items)
        now = datetime.now(timezone.utc)

        def score(ep: Any) -> float:
            try:
                ts = ep.timestamp.replace(tzinfo=timezone.utc)
                age_days = (now - ts).total_seconds() / 86400
                # Linear decay over 30 days
                return max(0.0, 1.0 - age_days / 30.0)
            except Exception:
                return 0.0

        ranked = sorted(items, key=score, reverse=True)
        return ranked[: self.max_episodes]

    # ------------------------------------------------------------------
    # FACTS — SALIENCE + CONFIDENCE
    # ------------------------------------------------------------------

    def _select_facts(self, items: List[Any]) -> List[Any]:
        items = self._dedupe(items)

        def score(f: Any) -> float:
            try:
                sal = f.meta.get("salience", 0.0) if hasattr(f, "meta") else 0.0
            except Exception:
                sal = 0.0
            try:
                conf = getattr(f, "confidence", 0.5) or 0.5
            except Exception:
                conf = 0.5
            return (sal + conf) / 2.0

        ranked = sorted(items, key=score, reverse=True)
        return ranked[: self.max_facts]

    # ------------------------------------------------------------------
    # SKILLS — DIVERSITY BY TRIGGER VARIETY
    # ------------------------------------------------------------------

    def _select_skills(self, items: List[Any]) -> List[Any]:
        items = self._dedupe(items)

        def diversity(skill: Any) -> int:
            try:
                phrases = set(skill.trigger_phrases or [])
                patterns = set(skill.trigger_patterns or [])
                return len(phrases) + len(patterns)
            except Exception:
                return 0

        ranked = sorted(items, key=diversity, reverse=True)
        return ranked[: self.max_skills]

    # ------------------------------------------------------------------
    # DEDUPLICATION
    # ------------------------------------------------------------------

    def _dedupe(self, items: List[Any]) -> List[Any]:
        """
        Deduplicate items based on their `.id` attribute if present,
        otherwise using the Python object id.

        This ensures stability across calls and avoids repeated memories.
        """
        seen: Set[Any] = set()
        result: List[Any] = []

        for it in items:
            key = getattr(it, "id", id(it))
            if key not in seen:
                seen.add(key)
                result.append(it)

        return result