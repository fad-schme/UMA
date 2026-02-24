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
from typing import Any, Dict, List, Optional, Set

from ..utils.dedupe import dedupe_by_id

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
    ) -> None:
        self.max_episodes = max(1, int(max_episodes))
        self.max_facts = max(1, int(max_facts))
        self.max_chunks = max(1, int(max_chunks))
        self.max_skills = max(1, int(max_skills))
        self.max_graph_items = max(1, int(max_graph_items))

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
            "episodes": self._select_episodes(raw.get("episodes", []) or []),
            "facts": self._select_facts(raw.get("facts", []) or [], policy=policy),
            "chunks": self._select_chunks(raw.get("chunks", []) or [], policy=policy),
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

        def score(f: Any) -> float:
            # --- Base score: salience + confidence ---
            try:
                if isinstance(f, dict):
                    sal = float(f.get("salience", 0.0) or 0.0)
                else:
                    sal = float(getattr(f, "salience", 0.0) or 0.0)
            except Exception:
                sal = 0.0

            try:
                if isinstance(f, dict):
                    conf = float(f.get("confidence", 0.5) or 0.5)
                else:
                    conf = float(getattr(f, "confidence", 0.5) or 0.5)
            except Exception:
                conf = 0.5

            base = (sal + conf) / 2.0

            # --- Scope weighting via policy (if present) ---
            return base

        ranked = sorted(items, key=score, reverse=True)
        return ranked[: self.max_facts]

    # -------------------- Chunks --------------------

    def _select_chunks(self, items: List[Any], *, policy: Optional[Any] = None) -> List[Any]:
        items = self._dedupe(items)

        def _meta(ch: Any) -> Dict[str, Any]:
            if isinstance(ch, dict):
                raise TypeError(
                    "MemorySelector expected Chunk objects for chunk selection; got dict. "
                    "Fix chunk store/core to return Chunk only."
                )
            m = getattr(ch, "meta", None) or {}
            return m if isinstance(m, dict) else {}

        def _route_weight(route: str) -> float:
            # Strong, deterministic precedence:
            # evidence > query hits > neighbors
            route = (route or "").strip().lower()
            if route == "evidence":
                return 10.0
            if route == "neighbor":
                return 0.0
            # default: query hit (vector/lexical)
            return 5.0

        def _method_weight(method: str) -> float:
            # Lexical confirmation gets a small deterministic bump.
            method = (method or "").strip().lower()
            if method == "lexical":
                return 0.5
            if method == "vector":
                return 0.2
            return 0.0

        def _pos(ch: Any) -> int:
            try:
                return int(getattr(ch, "position", 0) or 0)
            except Exception:
                return 0

        def score(ch: Any) -> float:
            m = _meta(ch)
            route = m.get("retrieval_route")
            method = m.get("retrieval_method")
            lexical_score = 0.0
            try:
                lexical_score = float(m.get("lexical_score", 0.0) or 0.0)
            except Exception:
                lexical_score = 0.0

            # Note: we don't currently have vector similarity scores plumbed through
            # from the vector store, so we rely on route/method + lexical score.
            return _route_weight(str(route or "")) + _method_weight(str(method or "")) + lexical_score

        # Rank by evidence first, then lexical confirmations, with position only as a tie-breaker
        # within the same doc (so we don't globally bias toward intros).
        def sort_key(ch: Any) -> tuple:
            m = _meta(ch)
            route = str(m.get("retrieval_route") or "")
            method = str(m.get("retrieval_method") or "")
            try:
                lex = float(m.get("lexical_score", 0.0) or 0.0)
            except Exception:
                lex = 0.0
            doc = str(getattr(ch, "doc_id", "") or "")
            pos = _pos(ch)
            return (
                -score(ch),  # primary: evidence-driven score
                doc,         # stable grouping
                pos,         # tie-break within doc only
            )

        ranked = sorted(items, key=sort_key)
        return ranked[: self.max_chunks]

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
        return dedupe_by_id(items)


def _get_owner_type(item: Any) -> str:
    if isinstance(item, dict):
        return (item.get("owner_type") or "").lower()
    return (getattr(item, "owner_type", None) or "").lower()
