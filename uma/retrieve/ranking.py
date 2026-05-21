"""
uma.retrieve.ranking
=========================

Canonical deterministic ranking for UMA retrieval candidates.

Design constraints
------------------
- Pure functions only (no I/O, no DB, no network).
- Candidate features must be carried on the existing object model via `.meta`.
- Ranking math lives here (single owner) to avoid distributed reranking.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from uma.common.accessors import get_attr_or_key
from uma.retrieve.user_query_helper import extract_keywords_and_phrases, build_fact_embedding_text

logger = logging.getLogger(__name__)


def _safe_float(val: Any, *, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


def _meta(obj: Any) -> Dict[str, Any]:
    m = get_attr_or_key(obj, "meta") or {}
    return m if isinstance(m, dict) else {}


def _id(obj: Any) -> str:
    sid = get_attr_or_key(obj, "id", "") or ""
    return str(sid)


def _doc_id(obj: Any) -> str:
    sid = get_attr_or_key(obj, "doc_id", "") or ""
    return str(sid)

def _owner_type(obj: Any) -> str:
    v = get_attr_or_key(obj, "owner_type", "") or ""
    return str(v)


def _owner_id(obj: Any) -> str:
    v = get_attr_or_key(obj, "owner_id", "") or ""
    return str(v)


def _position(obj: Any) -> int:
    return int(_safe_float(get_attr_or_key(obj, "position", 0) or 0, default=0.0))


def extract_terms(query_text: str) -> List[str]:
    extracted = extract_keywords_and_phrases(query_text or "")
    terms = (extracted.get("keywords") or []) + (extracted.get("keyphrases") or [])
    out = []
    for t in terms:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if len(t) <= 2:
            continue
        out.append(t)
    return out


def _update_meta(obj: Any, updates: Dict[str, Any]) -> None:
    if not updates:
        return
    try:
        meta = _meta(obj)
        merged = dict(meta)
        merged.update(updates)
        if isinstance(obj, dict):
            obj["meta"] = merged
        else:
            obj.meta = merged  # type: ignore[attr-defined]
    except Exception:
        # Scoring metadata is optional; never fail ranking because of meta assignment.
        pass


def _candidate_text_for_rerank(obj: Any) -> str:
    t = get_attr_or_key(obj, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    try:
        return str(build_fact_embedding_text(obj) or "")
    except Exception:
        subj = get_attr_or_key(obj, "subject", "") or ""
        pred = get_attr_or_key(obj, "predicate", "") or ""
        objv = get_attr_or_key(obj, "object", "") or ""
        return f"{subj} {pred} {objv}".strip()


def _tokenize_text(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 3]


def _fact_specificity_adjustment(candidate: Any, terms: Sequence[str], *, exact_query_hit: int) -> float:
    predicate_text = str(get_attr_or_key(candidate, "predicate", "") or "").strip().lower()
    object_raw = get_attr_or_key(candidate, "object", "") or ""
    if isinstance(object_raw, dict):
        object_text = str(object_raw.get("text") or object_raw.get("content") or object_raw).strip().lower()
    else:
        object_text = str(object_raw).strip().lower()
    if not predicate_text and not object_text:
        return 0.0

    predicate_tokens = set(_tokenize_text(predicate_text))
    object_tokens = set(_tokenize_text(object_text))
    query_tokens = set(_tokenize_text(" ".join(terms or [])))
    if not query_tokens:
        return 0.0

    predicate_hits = sum(1 for term in query_tokens if term in predicate_text)
    object_hits = sum(1 for term in query_tokens if term in object_text)
    if predicate_hits <= 0 and object_hits <= 0 and not exact_query_hit:
        return 0.0

    object_hit_ratio = float(object_hits) / max(1, len(query_tokens))
    predicate_hit_ratio = float(predicate_hits) / max(1, len(query_tokens))
    novel_object_tokens = [tok for tok in object_tokens if tok not in predicate_tokens and tok not in query_tokens]
    specificity_bonus = min(1.0, float(len(novel_object_tokens)) / 2.0)

    generic_penalty = 0.0
    if object_tokens and len(object_tokens) <= 1 and object_tokens.issubset(predicate_tokens | query_tokens):
        generic_penalty = 0.75

    return (1.25 * object_hit_ratio) + (0.75 * predicate_hit_ratio) + (1.0 * specificity_bonus) - generic_penalty


def compute_rerank_score(*, query_text: str, candidate: Any, terms: Optional[Sequence[str]] = None) -> float:
    """
    Compute a deterministic rerank score using upstream signals and query overlap.

    This is post-retrieval only: it may reorder candidates but must never expand the pool.
    """
    m = _meta(candidate)
    v_raw = _safe_float(m.get("vector_score", 0.0) or 0.0, default=0.0)
    v = math.log1p(max(0.0, v_raw))
    lex = _safe_float(m.get("lexical_score", m.get("lexical_confidence", 0.0)) or 0.0, default=0.0)
    lex = max(0.0, lex)

    q = str(query_text or "").strip().lower()
    if terms is None:
        terms = extract_terms(q)
    text = _candidate_text_for_rerank(candidate).lower()

    term_hits = 0
    phrase_hits = 0
    for t in terms or []:
        if t and t in text:
            term_hits += 1
            if " " in t:
                phrase_hits += 1

    exact_query_hit = 1 if (len(q) >= 4 and q in text) else 0

    # Normalize term/phrase hits as coverage ratios so multi-term queries
    # don't balloon the score relative to the vector/lexical components.
    n_terms = max(1, len(terms or []))
    term_ratio = float(term_hits) / n_terms
    phrase_ratio = float(phrase_hits) / n_terms
    specificity = _fact_specificity_adjustment(candidate, terms or [], exact_query_hit=exact_query_hit)

    return (
        v
        + min(lex, 1.0)           # clamp lexical to [0, 1] — same scale as log1p(vector)
        + (2.0 * term_ratio)      # 0–2.0
        + (3.0 * phrase_ratio)    # 0–3.0 (sub-score of term coverage)
        + float(exact_query_hit)  # 0–1
        + float(specificity)      # prefer concrete answer-bearing facts over generic placeholders
    )


def rerank_candidates(query_text: str, candidates: Sequence[Any]) -> List[Any]:
    """
    Canonical rerank API.

    Reorders the already-retrieved candidate pool and attaches:
      - meta["rerank_score"]

    Owner scoping invariant:
    Reranking is applied within each (owner_type, owner_id) group as encountered
    in the input list (no cross-owner normalization). Membership never changes.
    """
    items = list(candidates or [])
    if not items:
        return []

    terms = extract_terms(query_text or "")

    groups: List[Tuple[str, str]] = []
    bucket: Dict[Tuple[str, str], List[Tuple[float, str, int, Any]]] = {}

    for idx, it in enumerate(items):
        key = (_owner_type(it), _owner_id(it))
        if key not in bucket:
            bucket[key] = []
            groups.append(key)
        score = compute_rerank_score(query_text=query_text or "", candidate=it, terms=terms)
        _update_meta(it, {"rerank_score": float(score)})
        bucket[key].append((float(score), _id(it), idx, it))

    out: List[Any] = []
    for key in groups:
        scored = bucket.get(key) or []
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        out.extend([it for _s, _sid, _idx, it in scored])
    return out


def fuse_candidates(
    *,
    dense: Sequence[Any],
    sparse: Sequence[Any],
    strategy: str = "rrf",
    rrf_k: int = 60,
) -> List[Any]:
    """
    Deterministically fuse dense + lexical candidate pools into a single union list.

    Strategies
    ----------
    - `rrf`: reciprocal-rank fusion using ranks only (provider-agnostic).
    - `overlap_boost`: prioritize overlap, then ranks.

    Feature merge
    -------------
    For ids present in both lists, preserves the dense object when possible (to keep
    `vector_score`), while copying missing sparse lexical features onto `.meta`.
    """
    strat = (strategy or "rrf").strip().lower()
    if strat not in ("rrf", "overlap_boost"):
        raise ValueError("strategy must be one of: rrf, overlap_boost")
    try:
        k0 = max(1, int(rrf_k))
    except Exception:
        k0 = 60

    dense_list = list(dense or [])
    sparse_list = list(sparse or [])

    dense_rank: Dict[str, int] = {}
    sparse_rank: Dict[str, int] = {}
    dense_item: Dict[str, Any] = {}
    sparse_item: Dict[str, Any] = {}

    for i, it in enumerate(dense_list, start=1):
        sid = _id(it)
        if sid and sid not in dense_rank:
            dense_rank[sid] = i
            dense_item[sid] = it

    for i, it in enumerate(sparse_list, start=1):
        sid = _id(it)
        if sid and sid not in sparse_rank:
            sparse_rank[sid] = i
            sparse_item[sid] = it

    ids = sorted(set(dense_rank) | set(sparse_rank))

    def _rrf_score(sid: str) -> float:
        s = 0.0
        dr = dense_rank.get(sid)
        sr = sparse_rank.get(sid)
        if dr is not None:
            s += 1.0 / float(k0 + dr)
        if sr is not None:
            s += 1.0 / float(k0 + sr)
        return s

    def _overlap_score(sid: str) -> float:
        dr = dense_rank.get(sid)
        sr = sparse_rank.get(sid)
        overlap = 1.0 if (dr is not None and sr is not None) else 0.0
        rank_bonus = 0.0
        if dr is not None:
            rank_bonus += 1.0 / float(k0 + dr)
        if sr is not None:
            rank_bonus += 1.0 / float(k0 + sr)
        return (10.0 * overlap) + rank_bonus

    score_fn = _rrf_score if strat == "rrf" else _overlap_score

    scored: List[Tuple[float, str]] = [(score_fn(sid), sid) for sid in ids]

    def _tie_key(pair: Tuple[float, str]) -> Tuple[float, int, int, int, int, str]:
        score, sid = pair
        dr = dense_rank.get(sid)
        sr = sparse_rank.get(sid)
        overlap = 1 if (dr is not None and sr is not None) else 0
        has_sparse = 1 if sr is not None else 0
        has_dense = 1 if dr is not None else 0
        # Prefer overlap, then sparse presence (lexical), then better ranks, then id.
        sr_val = int(sr) if sr is not None else 10**9
        dr_val = int(dr) if dr is not None else 10**9
        return (-float(score), -overlap, -has_sparse, -has_dense, sr_val + dr_val, str(sid))

    scored.sort(key=_tie_key)

    out: List[Any] = []
    for score, sid in scored:
        d_it = dense_item.get(sid)
        s_it = sparse_item.get(sid)
        it = d_it if d_it is not None else s_it
        if it is None:
            continue
        # Best-effort merge lexical features from sparse side onto dense object.
        if d_it is not None and s_it is not None:
            try:
                d_meta = _meta(d_it)
                s_meta = _meta(s_it)
                merged = dict(d_meta)
                for k in ("lexical_confidence", "lexical_score"):
                    if k not in merged and k in s_meta:
                        merged[k] = s_meta.get(k)
                merged["hybrid_overlap"] = True
                merged["hybrid_dense_rank"] = int(dense_rank.get(sid) or 0)
                merged["hybrid_sparse_rank"] = int(sparse_rank.get(sid) or 0)
                merged["hybrid_fusion_score"] = float(score)
                if isinstance(d_it, dict):
                    d_it["meta"] = merged
                else:
                    d_it.meta = merged  # type: ignore[attr-defined]
            except Exception:
                logger.exception("fuse_candidates: failed merging meta for id=%s", sid)
        else:
            try:
                meta = _meta(it)
                merged = dict(meta)
                merged["hybrid_overlap"] = bool(d_it is not None and s_it is not None)
                merged["hybrid_dense_rank"] = int(dense_rank.get(sid) or 0)
                merged["hybrid_sparse_rank"] = int(sparse_rank.get(sid) or 0)
                merged["hybrid_fusion_score"] = float(score)
                if isinstance(it, dict):
                    it["meta"] = merged
                else:
                    it.meta = merged  # type: ignore[attr-defined]
            except Exception:
                # Optional metadata. Never fail fusion because an object doesn't support meta assignment.
                pass
        out.append(it)
    return out


@dataclass(frozen=True)
class ScoreCard:
    id: str
    vector_score: float
    lexical_score: float
    rerank_score: float
    route: str
    method: str
    final_score: float


class Ranker:
    """
    Canonical ranking behavior for UMA retrieval.

    Ranking attaches derived features onto `.meta` (e.g., `rerank_score`, `final_score`)
    so downstream debugging is explainable.
    """

    def __init__(self, *, debug_scores: bool = False) -> None:
        self._debug = bool(debug_scores)

    # ----------------------------- Public -----------------------------

    def truncate(self, items: Sequence[Any], k: int) -> List[Any]:
        try:
            k_i = max(0, int(k))
        except Exception:
            k_i = 0
        return list(items or [])[:k_i] if k_i else []

    def rank_facts(self, items: Sequence[Any], *, query_text: str = "") -> List[Any]:
        items = list(items or [])
        if not items:
            return []

        terms = extract_terms(query_text or "")

        scored: List[Tuple[float, str, Any]] = []
        for f in items:
            rerank = compute_rerank_score(query_text=query_text or "", candidate=f, terms=terms)
            sal = _safe_float(get_attr_or_key(f, "salience", 0.0) or 0.0, default=0.0)
            conf = _safe_float(get_attr_or_key(f, "confidence", 0.5) or 0.5, default=0.5)
            quality = (max(0.0, sal) + max(0.0, conf)) / 2.0
            final = float(rerank) + (0.1 * float(quality))

            _update_meta(f, {"rerank_score": float(rerank), "final_score": float(final)})
            scored.append((float(final), _id(f), f))

        scored.sort(key=lambda x: (-x[0], x[1]))
        self._emit_scorecards("facts", scored)
        return [f for _s, _sid, f in scored]

    def rank_chunks(self, items: Sequence[Any], *, query_text: str = "") -> List[Any]:
        items = list(items or [])
        if not items:
            return []

        terms = extract_terms(query_text or "")

        scored: List[Tuple[int, float, str, int, str, Any]] = []
        for ch in items:
            m = _meta(ch)
            route = str(m.get("retrieval_route") or "")
            method = str(m.get("retrieval_method") or "")
            rerank = compute_rerank_score(query_text=query_text or "", candidate=ch, terms=terms)
            final = self._route_weight(route) + self._method_weight(method) + float(rerank)
            _update_meta(ch, {"rerank_score": float(rerank), "final_score": float(final)})

            route_pri = self._route_priority(route)
            scored.append((route_pri, float(final), _doc_id(ch), _position(ch), _id(ch), ch))

        scored.sort(key=lambda x: (x[0], -x[1], x[2], x[3], x[4]))
        self._emit_scorecards("chunks", [(s, sid, it) for (_rp, s, _d, _p, sid, it) in scored])
        return [ch for _rp, _s, _d, _p, _sid, ch in scored]

    def rank_episodes(self, items: Sequence[Any], *, query_text: str = "") -> List[Any]:
        items = list(items or [])
        if not items:
            return []
        now = datetime.now(timezone.utc)
        terms = extract_terms(query_text or "")
        scored: List[Tuple[float, str, Any]] = []
        for ep in items:
            sid = _id(ep)
            rerank = compute_rerank_score(query_text=query_text or "", candidate=ep, terms=terms)

            recency = 0.0
            try:
                ts = get_attr_or_key(ep, "timestamp", None)
                if ts is not None:
                    if getattr(ts, "tzinfo", None) is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_days = (now - ts).total_seconds() / 86400.0
                    recency = max(0.0, 1.0 - age_days / 90.0)
            except Exception:
                recency = 0.0

            final = float(rerank) + (0.3 * float(recency))
            _update_meta(ep, {"rerank_score": float(rerank), "final_score": float(final)})
            scored.append((final, sid, ep))
        scored.sort(key=lambda x: (-x[0], x[1]))
        self._emit_scorecards("episodes", scored)
        return [ep for _s, _sid, ep in scored]

    def rank_skills(self, items: Sequence[Any], *, query_text: str = "") -> List[Any]:
        items = list(items or [])
        if not items:
            return []
        terms = extract_terms(query_text or "")
        scored: List[Tuple[float, str, Any]] = []
        for sk in items:
            sid = _id(sk)
            rerank = compute_rerank_score(query_text=query_text or "", candidate=sk, terms=terms)
            phrases = get_attr_or_key(sk, "trigger_phrases", []) or []
            patterns = get_attr_or_key(sk, "trigger_patterns", []) or []
            diversity = len(set(phrases)) + len(set(patterns))
            final = float(rerank) + (0.05 * float(diversity))
            _update_meta(sk, {"rerank_score": float(rerank), "final_score": float(final)})
            scored.append((final, sid, sk))
        scored.sort(key=lambda x: (-x[0], x[1]))
        self._emit_scorecards("skills", scored)
        return [sk for _s, _sid, sk in scored]

    # ----------------------------- Internals -----------------------------

    def _route_weight(self, route: str) -> float:
        route = (route or "").strip().lower()
        if route == "evidence":
            return 10.0
        if route == "neighbor":
            return 0.0
        return 5.0

    def _route_priority(self, route: str) -> int:
        """
        Strong precedence:
          evidence (0) > query hits (1) > neighbors (2)
        """
        r = (route or "").strip().lower()
        if r == "evidence":
            return 0
        if r == "neighbor":
            return 2
        return 1

    def _method_weight(self, method: str) -> float:
        method = (method or "").strip().lower()
        if method == "lexical":
            return 0.5
        if method == "vector":
            return 0.2
        return 0.0

    def _emit_scorecards(self, lane: str, scored: Iterable[Tuple[float, str, Any]]) -> None:
        if not self._debug:
            return
        cards: List[ScoreCard] = []
        for final, sid, obj in scored:
            m = _meta(obj)
            card = ScoreCard(
                id=sid,
                vector_score=_safe_float(m.get("vector_score", 0.0) or 0.0),
                lexical_score=_safe_float(m.get("lexical_score", m.get("lexical_confidence", 0.0)) or 0.0),
                rerank_score=_safe_float(m.get("rerank_score", 0.0) or 0.0),
                route=str(m.get("retrieval_route") or ""),
                method=str(m.get("retrieval_method") or ""),
                final_score=float(final),
            )
            cards.append(card)
            _update_meta(
                obj,
                {
                    "score_card": {
                        "id": card.id,
                        "vector_score": float(card.vector_score),
                        "lexical_score": float(card.lexical_score),
                        "route": str(card.route),
                        "rerank_score": float(card.rerank_score),
                        "final_score": float(card.final_score),
                    }
                },
            )

        if not logger.isEnabledFor(logging.DEBUG):
            return
        # Log only the top few to avoid noise.
        for c in cards[:10]:
            logger.debug(
                "Ranker scorecard lane=%s id=%s final=%.4f rerank=%.4f vector=%.4f lexical=%.4f route=%s method=%s",
                lane,
                c.id,
                c.final_score,
                c.rerank_score,
                c.vector_score,
                c.lexical_score,
                c.route,
                c.method,
            )
