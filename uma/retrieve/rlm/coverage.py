# uma/retrieve/rlm/coverage.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from uma.common.accessors import get_attr_or_key
from uma.common.text import get_stopwords

logger = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    semantic_total: int
    semantic_high_salience: int
    semantic_enough: bool
    cluster_summaries: int
    episode_summaries: int
    graph_total: int
    graph_entity_support: int
    graph_entity_total: int
    graph_predicate_support: int
    graph_predicate_total: int
    contradiction_count: int
    contradictions: list[str]
    has_contradictions: bool
    require_semantic: bool
    enough: bool
    needs_semantic: bool
    needs_clusters: bool
    needs_episode_summaries: bool
    novelty_last_step: int
    novelty_recent_sum: int
    diminishing_returns: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise the coverage report to a plain dict for logging and debugging."""
        return self.__dict__


def assess_coverage(
    *,
    facts: list[Any],
    episodes: list[Any],
    graph: list[Any],
    salience_threshold: float,
    min_semantic_facts: int,
    min_high_salience_facts: int,
    min_cluster_summaries: int,
    require_semantic: bool,
    prefer_clusters: bool,
    novelty_history: Optional[list[dict[str, int]]] = None,
    novelty_window: int = 2,
    min_recent_novelty: int = 1,
) -> CoverageReport:
    """
    Evaluate how well the current ``ContextPack`` covers the query.

    Returns a ``CoverageReport`` with ``enough=True`` when semantic fact counts
    and salience thresholds are satisfied. Also tracks novelty across steps to
    detect diminishing returns and signal the controller to stop iterating.
    """
    logger.debug(
        "assess_coverage: facts=%d episodes=%d graph=%d",
        len(facts),
        len(episodes),
        len(graph),
    )
    semantic_total = len(facts)
    semantic_high_salience = sum(
        1 for f in facts if _fact_salience(f) >= salience_threshold
    )
    semantic_enough = (
        semantic_total >= min_semantic_facts
        or semantic_high_salience >= min_high_salience_facts
    )

    cluster_summaries = sum(1 for ep in episodes if _is_cluster_summary(ep))
    episode_summaries = sum(1 for ep in episodes if _is_episode_summary(ep))
    if prefer_clusters and cluster_summaries == 0:
        episode_summaries = 0

    graph_total = len(graph)
    graph_entity_support, graph_entity_total = _graph_entity_support(facts, graph)
    graph_predicate_support, graph_predicate_total = _graph_predicate_support(facts, graph)
    contradictions = detect_contradictions(episodes)

    novelty_last, novelty_sum, diminishing = compute_novelty_signals(
        novelty_history or [], novelty_window, min_recent_novelty
    )

    enough = semantic_enough if require_semantic else semantic_total >= 6

    return CoverageReport(
        semantic_total,
        semantic_high_salience,
        semantic_enough,
        cluster_summaries,
        episode_summaries,
        graph_total,
        graph_entity_support,
        graph_entity_total,
        graph_predicate_support,
        graph_predicate_total,
        len(contradictions),
        contradictions,
        bool(contradictions),
        require_semantic,
        enough,
        not semantic_enough,
        prefer_clusters and semantic_enough and cluster_summaries < min_cluster_summaries,
        False,
        novelty_last,
        novelty_sum,
        diminishing,
    )


def compute_novelty_signals(
    novelty_history: list[dict[str, int]],
    window: int,
    min_recent_sum: int,
) -> tuple[int, int, bool]:
    """Compute last-step and windowed novelty totals, and a diminishing-returns flag."""
    if not novelty_history:
        return 0, 0, False

    recent = novelty_history[-window:]

    def step_total(d):
        """Sum all novelty counts for a single step dict."""
        return sum(d.values())

    last = step_total(novelty_history[-1])
    total = sum(step_total(d) for d in recent)
    return last, total, total < min_recent_sum


def compute_confidence(coverage: CoverageReport) -> dict[str, float]:
    """
    Compute a simple confidence score from coverage signals.
    """
    score = 0.0
    if coverage.semantic_enough:
        score += 0.4
    if coverage.cluster_summaries > 0 or coverage.episode_summaries > 0:
        score += 0.2
    if coverage.graph_total > 0:
        score += 0.1
    if coverage.graph_entity_support > 0 or coverage.graph_predicate_support > 0:
        score += 0.1
    if coverage.novelty_recent_sum > 0:
        score += 0.2
    if coverage.has_contradictions:
        score -= 0.2
    if coverage.diminishing_returns:
        score -= 0.1
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "semantic_enough": 1.0 if coverage.semantic_enough else 0.0,
        "clusters_present": 1.0 if coverage.cluster_summaries > 0 else 0.0,
        "graph_present": 1.0 if coverage.graph_total > 0 else 0.0,
        "graph_entity_support": float(coverage.graph_entity_support),
        "graph_predicate_support": float(coverage.graph_predicate_support),
        "novelty_recent": float(coverage.novelty_recent_sum),
        "contradictions": 1.0 if coverage.has_contradictions else 0.0,
    }


# --- helpers (unchanged logic) ---

def _fact_salience(fact: Any) -> float:
    try:
        # Direct field takes precedence (Fact dataclass stores salience as a field, not in meta)
        val = get_attr_or_key(fact, "salience", None)
        if val is not None:
            return float(val)
        meta = get_attr_or_key(fact, "meta") or {}
        return float(meta.get("salience", 0.0)) if isinstance(meta, dict) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fact_entities(facts: list[Any], limit: int = 12) -> list[str]:
    entities: list[str] = []
    seen = set()
    for f in facts or []:
        if len(entities) >= limit:
            break
        try:
            if isinstance(f, dict):
                subj = f.get("subject")
                obj = f.get("object")
            else:
                subj = getattr(f, "subject", None)
                obj = getattr(f, "object", None)
            for val in (subj, obj):
                if val is None:
                    continue
                s = str(val).strip().lower()
                if s.startswith("user:"):
                    s = s[5:]
                if not s or s in seen:
                    continue
                seen.add(s)
                entities.append(s)
                if len(entities) >= limit:
                    break
        except Exception:
            logger.debug("coverage: skipped malformed item", exc_info=True)
            continue
    return entities


def _graph_entity_support(facts: list[Any], graph: list[Any]) -> tuple[int, int]:
    entities = _fact_entities(facts)
    if not entities:
        return 0, 0
    graph_ids: set = set()
    for node in graph or []:
        try:
            if isinstance(node, dict):
                props = node.get("properties") or {}
                for key in ("id", "name", "value", "text"):
                    val = props.get(key)
                    if val:
                        graph_ids.add(str(val).strip().lower())
                node_id = node.get("id")
                if node_id:
                    graph_ids.add(str(node_id).strip().lower())
        except Exception:
            logger.debug("coverage: skipped malformed item", exc_info=True)
            continue
    support = sum(1 for e in entities if e in graph_ids)
    return support, len(entities)


def _graph_predicate_support(facts: list[Any], graph: list[Any]) -> tuple[int, int]:
    predicates: list[str] = []
    seen = set()
    for f in facts or []:
        try:
            pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
            if pred:
                p = str(pred).strip().upper()
                if p and p not in seen:
                    seen.add(p)
                    predicates.append(p)
        except Exception:
            logger.debug("coverage: skipped malformed item", exc_info=True)
            continue
    if not predicates:
        return 0, 0
    graph_preds: set = set()
    for node in graph or []:
        try:
            if isinstance(node, dict):
                for key in ("predicate", "rel_type", "type"):
                    val = node.get(key)
                    if val:
                        graph_preds.add(str(val).strip().upper())
        except Exception:
            logger.debug("coverage: skipped malformed item", exc_info=True)
            continue
    support = sum(1 for p in predicates if p in graph_preds)
    return support, len(predicates)


def _is_cluster_summary(item: Any) -> bool:
    return isinstance(item, dict) and "episode_ids" in item


def _is_episode_summary(item: Any) -> bool:
    return isinstance(item, dict) and "summary" in item

def detect_contradictions(episodes: list[Any]) -> list[str]:
    """
    Lightweight contradiction detection from episode summaries.

    This is NOT a semantic truth checker.
    It is a heuristic used only as a coverage signal for the RLM controller.
    """
    summaries = [_episode_summary_text(ep) for ep in episodes]
    summaries = [s for s in summaries if s]
    if not summaries:
        return []

    groups = _contradiction_groups()
    positive: dict[str, set] = {name: set() for name, _ in groups}
    negative: dict[str, set] = {name: set() for name, _ in groups}

    for summary in summaries:
        for name, (pos_verbs, neg_verbs) in groups:
            positive[name].update(_extract_objects(summary, pos_verbs))
            negative[name].update(_extract_objects(summary, neg_verbs))

    contradictions: list[str] = []
    for name in positive.keys():
        overlap = positive[name] & negative[name]
        for obj in sorted(overlap):
            contradictions.append(f"{name}:{obj}")

    return contradictions


def _episode_summary_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("summary", "")).lower()
    return str(getattr(item, "summary", "")).lower()


def _contradiction_groups() -> list[tuple[str, tuple[list[str], list[str]]]]:
    """
    Define simple verb polarity groups.

    These are intentionally coarse and deterministic.
    """
    return [
        (
            "preference",
            (
                ["like", "likes", "love", "loves", "enjoy", "enjoys", "prefer", "prefers"],
                ["dislike", "dislikes", "hate", "hates", "avoid", "avoids"],
            ),
        ),
        (
            "permission",
            (
                ["allow", "allows", "enable", "enabled", "approve", "approves"],
                ["deny", "denies", "disable", "disabled", "reject", "rejects"],
            ),
        ),
        (
            "inclusion",
            (
                ["include", "includes", "add", "adds", "keep", "keeps"],
                ["exclude", "excludes", "remove", "removes", "drop", "drops"],
            ),
        ),
    ]


def _extract_objects(text: str, verbs: list[str]) -> list[str]:
    """
    Extract object phrases following given verbs.
    """
    out: list[str] = []
    if not text:
        return out

    for verb in verbs:
        pattern = r"\b" + re.escape(verb) + r"\b\s+([a-z0-9_\- ]{1,80})"
        for match in re.findall(pattern, text):
            obj = _normalize_object_phrase(match)
            if obj:
                out.append(obj)

    return out


def _normalize_object_phrase(phrase: str) -> str:
    tokens = re.findall(r"[a-z0-9_\-]+", phrase.lower())
    if not tokens:
        return ""

    stop = get_stopwords()
    while tokens and tokens[0] in stop:
        tokens.pop(0)

    return " ".join(tokens[:6]) if tokens else ""
