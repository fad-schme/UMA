# uma/core/retrieval/rlm/policy.py

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Tuple


def good_enough(counts: Dict[str, int]) -> bool:
    """
    Deterministic stopping heuristic.

    Keep SIMPLE and explainable.
    """
    if counts.get("facts", 0) >= 6:
        return True
    if counts.get("episodes", 0) >= 4 and counts.get("facts", 0) >= 2:
        return True
    if counts.get("skills", 0) >= 3 and counts.get("facts", 0) >= 2:
        return True
    if counts.get("graph", 0) >= 8:
        return True
    return False


@dataclass
class CoverageReport:
    semantic_total: int
    semantic_high_salience: int
    semantic_enough: bool
    cluster_summaries: int
    episode_summaries: int
    graph_total: int
    contradiction_count: int
    contradictions: List[str]
    has_contradictions: bool
    require_semantic: bool
    enough: bool
    needs_semantic: bool
    needs_clusters: bool
    needs_episode_summaries: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_total": self.semantic_total,
            "semantic_high_salience": self.semantic_high_salience,
            "semantic_enough": self.semantic_enough,
            "cluster_summaries": self.cluster_summaries,
            "episode_summaries": self.episode_summaries,
            "graph_total": self.graph_total,
            "contradiction_count": self.contradiction_count,
            "contradictions": self.contradictions,
            "has_contradictions": self.has_contradictions,
            "require_semantic": self.require_semantic,
            "enough": self.enough,
            "needs_semantic": self.needs_semantic,
            "needs_clusters": self.needs_clusters,
            "needs_episode_summaries": self.needs_episode_summaries,
        }


def assess_coverage(
    *,
    facts: List[Any],
    episodes: List[Any],
    graph: List[Any],
    salience_threshold: float,
    min_semantic_facts: int,
    min_high_salience_facts: int,
    min_cluster_summaries: int,
    require_semantic: bool,
    prefer_clusters: bool,
) -> CoverageReport:
    semantic_total = len(facts)
    semantic_high_salience = sum(
        1 for fact in facts if _fact_salience(fact) >= float(salience_threshold)
    )
    semantic_enough = (
        semantic_total >= int(min_semantic_facts)
        or semantic_high_salience >= int(min_high_salience_facts)
    )

    cluster_summaries = sum(1 for ep in episodes if _is_cluster_summary(ep))
    episode_summaries = sum(1 for ep in episodes if _is_episode_summary(ep))
    if prefer_clusters and cluster_summaries == 0:
        episode_summaries = 0

    graph_total = len(graph)

    contradictions = detect_contradictions(episodes)
    contradiction_count = len(contradictions)
    has_contradictions = contradiction_count > 0

    needs_semantic = not semantic_enough
    needs_clusters = (
        prefer_clusters
        and needs_semantic
        and cluster_summaries < int(min_cluster_summaries)
    )
    needs_episode_summaries = (
        prefer_clusters
        and needs_semantic
        and cluster_summaries >= int(min_cluster_summaries)
        and episode_summaries == 0
    )

    counts = {
        "facts": semantic_total,
        "episodes": len(episodes),
        "skills": 0,
        "graph": graph_total,
    }
    enough = semantic_enough if require_semantic else good_enough(counts)

    return CoverageReport(
        semantic_total=semantic_total,
        semantic_high_salience=semantic_high_salience,
        semantic_enough=semantic_enough,
        cluster_summaries=cluster_summaries,
        episode_summaries=episode_summaries,
        graph_total=graph_total,
        contradiction_count=contradiction_count,
        contradictions=contradictions,
        has_contradictions=has_contradictions,
        require_semantic=require_semantic,
        enough=enough,
        needs_semantic=needs_semantic,
        needs_clusters=needs_clusters,
        needs_episode_summaries=needs_episode_summaries,
    )


def _fact_salience(fact: Any) -> float:
    try:
        if isinstance(fact, dict):
            if "salience" in fact and fact["salience"] is not None:
                return float(fact["salience"])
            meta = fact.get("meta") or {}
            return float(meta.get("salience") or 0.0)
        meta = getattr(fact, "meta", {}) or {}
        return float(meta.get("salience") or 0.0)
    except Exception:
        return 0.0


def _is_cluster_summary(item: Any) -> bool:
    return isinstance(item, dict) and "episode_ids" in item and "latest_timestamp" in item


def _is_episode_summary(item: Any) -> bool:
    if _is_cluster_summary(item):
        return False
    if isinstance(item, dict):
        return "summary" in item and "timestamp" in item
    return hasattr(item, "summary") and hasattr(item, "timestamp")


def merge_unique(existing: List[Any], new: List[Any], max_items: int) -> List[Any]:
    """
    Merge + dedupe by `.id` or dict["id"].
    """
    seen = set()
    out = []

    def key(x):
        if hasattr(x, "id"):
            return x.id
        if isinstance(x, dict):
            return x.get("id")
        return id(x)

    for it in existing + new:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
        if len(out) >= max_items:
            break

    return out


def detect_contradictions(episodes: List[Any]) -> List[str]:
    summaries = [_episode_summary_text(ep) for ep in episodes]
    summaries = [s for s in summaries if s]
    if not summaries:
        return []

    groups = _contradiction_groups()
    positive: Dict[str, set] = {name: set() for name, _ in groups}
    negative: Dict[str, set] = {name: set() for name, _ in groups}

    for summary in summaries:
        for name, (pos_verbs, neg_verbs) in groups:
            positive[name].update(_extract_objects(summary, pos_verbs))
            negative[name].update(_extract_objects(summary, neg_verbs))

    contradictions: List[str] = []
    for name in positive.keys():
        overlap = positive[name] & negative[name]
        for obj in sorted(overlap):
            contradictions.append(f"{name}:{obj}")
    return contradictions


def _episode_summary_text(item: Any) -> str:
    text = ""
    if isinstance(item, dict):
        text = item.get("summary") or ""
    else:
        text = getattr(item, "summary", "") or ""
    return str(text).lower().strip()


def _contradiction_groups() -> List[Tuple[str, Tuple[List[str], List[str]]]]:
    return [
        (
            "preference",
            (
                ["like", "likes", "love", "loves", "enjoy", "enjoys", "prefer", "prefers", "want", "wants"],
                ["dislike", "dislikes", "hate", "hates", "avoid", "avoids", "reject", "rejects"],
            ),
        ),
        (
            "permission",
            (
                ["allow", "allows", "enable", "enabled", "approve", "approves", "accept", "accepts"],
                ["deny", "denies", "disable", "disabled", "reject", "rejects", "block", "blocks", "ban", "bans"],
            ),
        ),
        (
            "inclusion",
            (
                ["include", "includes", "add", "adds", "keep", "keeps"],
                ["exclude", "excludes", "remove", "removes", "drop", "drops"],
            ),
        ),
        (
            "state",
            (
                ["turn on", "turned on", "switch on", "switched on"],
                ["turn off", "turned off", "switch off", "switched off"],
            ),
        ),
    ]


def _extract_objects(text: str, verbs: List[str]) -> List[str]:
    if not text:
        return []
    out: List[str] = []
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
    stop = {"the", "a", "an", "to", "of", "for", "on", "in", "my", "our", "their", "his", "her", "your"}
    while tokens and tokens[0] in stop:
        tokens.pop(0)
    if not tokens:
        return ""
    return " ".join(tokens[:6])
