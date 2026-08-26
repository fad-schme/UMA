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
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from uma.common.accessors import get_attr_or_key
from uma.common.text import extract_keywords_and_phrases, build_fact_embedding_text, build_query_term_set

logger = logging.getLogger(__name__)

# Weight applied to vector_score (already normalized to (0, 1]) in
# compute_rerank_score, on a comparable scale to term_ratio (0-2) + phrase_ratio
# (0-3) + exact_query_hit (0-1) so semantically strong but lexically distant
# candidates can compete with lexically-adjacent but topically generic ones.
VECTOR_SCORE_WEIGHT = 3.0


def _safe_float(val: Any, *, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _meta(obj: Any) -> dict[str, Any]:
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


def extract_terms(query_text: str) -> list[str]:
    """Extract normalised keyword terms from a query string for lexical scoring."""
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


def extract_query_entities(query_text: str) -> list[str]:
    """Extract normalised candidate entities from a query for entity-overlap scoring."""
    return build_query_term_set(query_text or "").entities


def _update_meta(obj: Any, updates: dict[str, Any]) -> None:
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
        logger.debug("_update_meta: failed to attach score metadata to object=%r", obj, exc_info=True)


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


def _tokenize_text(text: str) -> list[str]:
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


def compute_rerank_score(
    *,
    query_text: str,
    candidate: Any,
    terms: Optional[Sequence[str]] = None,
    entities: Optional[Sequence[str]] = None,
) -> float:
    """
    Compute a deterministic rerank score using upstream signals and query overlap.

    This is post-retrieval only: it may reorder candidates but must never expand the pool.
    """
    m = _meta(candidate)
    v_raw = _safe_float(m.get("vector_score", 0.0) or 0.0, default=0.0)
    # vector_score is already normalized to (0, 1] by the vector backends
    # (e.g. exp(-distance) in LanceDBIndex.query). log1p compressed an
    # already-bounded score into (0, log(2)] =~ (0, 0.69], leaving vector
    # similarity unable to compete with the 0-6 point range of term/phrase
    # overlap below — a semantically strong, lexically-distant candidate
    # could never outrank a lexically-adjacent but topically generic one.
    # Clamp (defends against out-of-range values from callers/tests) and
    # weight on a comparable scale to term/phrase overlap instead.
    v = VECTOR_SCORE_WEIGHT * min(max(0.0, v_raw), 1.0)
    lex = _safe_float(m.get("lexical_score", m.get("lexical_confidence", 0.0)) or 0.0, default=0.0)
    lex = max(0.0, lex)

    q = str(query_text or "").strip().lower()
    if terms is None:
        terms = extract_terms(q)
    if entities is None:
        entities = extract_query_entities(q)
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

    # Entity overlap: candidates tagged with a query entity (facts get tagged
    # at ingest time, in meta["entities"]) rank above otherwise-identical
    # candidates with no entity overlap. Untagged candidates (chunks,
    # episodes, skills — not tagged by this pass) simply score 0 here.
    candidate_entities = {str(e).strip().lower() for e in (m.get("entities") or []) if e}
    entity_hits = sum(1 for e in (entities or []) if e in candidate_entities)
    n_entities = max(1, len(entities or []))
    entity_ratio = float(entity_hits) / n_entities if entities else 0.0

    return (
        v
        + min(lex, 1.0)           # clamp lexical to [0, 1]
        + (2.0 * term_ratio)      # 0–2.0
        + (3.0 * phrase_ratio)    # 0–3.0 (sub-score of term coverage)
        + float(exact_query_hit)  # 0–1
        + float(specificity)      # prefer concrete answer-bearing facts over generic placeholders
        + (2.0 * entity_ratio)    # 0–2.0 — same order of magnitude as term_ratio
    )


def rerank_candidates(query_text: str, candidates: Sequence[Any]) -> list[Any]:
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
    entities = extract_query_entities(query_text or "")

    groups: list[tuple[str, str]] = []
    bucket: dict[tuple[str, str], list[tuple[float, str, int, Any]]] = {}

    for idx, it in enumerate(items):
        key = (_owner_type(it), _owner_id(it))
        if key not in bucket:
            bucket[key] = []
            groups.append(key)
        score = compute_rerank_score(query_text=query_text or "", candidate=it, terms=terms, entities=entities)
        _update_meta(it, {"rerank_score": float(score)})
        bucket[key].append((float(score), _id(it), idx, it))

    out: list[Any] = []
    for key in groups:
        scored = bucket.get(key) or []
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        out.extend([it for _s, _sid, _idx, it in scored])
    return out


def mmr_select(
    candidates: Sequence[tuple[str, float, Sequence[float]]],
    *,
    k: int,
    lambda_diversity: float,
) -> list[str]:
    """
    Maximal Marginal Relevance selection (retrieval-ranking-gap ticket 07).

    Iteratively picks the candidate maximizing
    ``lambda_diversity * relevance - (1 - lambda_diversity) * redundancy``,
    where redundancy is the candidate's max cosine similarity to anything
    already selected. Plain top-k-by-score is the ``lambda_diversity=1.0``
    special case.

    Exists to counter a specific failure mode ticket 02 diagnosed and
    ticket 06 confirmed on real data: near-duplicate, topically-adjacent
    candidates can fill an entire top-k pool and crowd out a scattered but
    genuinely distinct answer-bearing candidate, even when the pool's
    aggregate relevance looks fine by any score threshold (a relevance
    floor cannot detect this -- see ticket 05's reverted attempt).

    Vectorized: normalizes every candidate vector once, then scores the
    full remaining set against the whole selected set with one matrix-
    vector product per round (O(n * k) total) instead of a per-pair Python
    loop (O(n * k * k) scalar calls, ~6s/query for n=200, k=30 in ticket
    06's initial prototype vs. ~150ms here for the same input).

    Parameters
    ----------
    candidates:
        ``(id, relevance_score, vector)`` triples. Order does not matter.
        ``relevance_score`` may be on any scale (raw vector similarity,
        a multi-term rerank formula, anything monotonic in relevance) --
        it is min-max normalized to ``[0, 1]`` internally so it trades off
        against redundancy (cosine similarity, naturally bounded ~``[0, 1]``
        for embeddings) on a comparable scale. Skipping this step is a real
        footgun, not a formality: an unnormalized relevance score whose
        typical magnitude is several times larger than 1 makes the
        redundancy term negligible regardless of ``lambda_diversity``,
        silently degrading MMR to plain top-k even at low lambda (caught in
        ticket 07 review -- switching the relevance signal from raw
        ``vector_score`` (~``[0, 1]``) to ``compute_rerank_score`` (~``[0, 9]``)
        without this normalization made lambda=0.5 behave like lambda≈0.95).
    k:
        Selection size. Capped at ``len(candidates)``.
    lambda_diversity:
        1.0 = pure relevance (reproduces top-k-by-score exactly).
        0.0 = pure diversity (ignores relevance entirely).

    Returns
    -------
    Selected ids, in selection order (highest marginal-relevance first).
    """
    if not candidates or k <= 0:
        return []
    ids = [cid for cid, _score, _vec in candidates]
    raw_scores = np.asarray([score for _cid, score, _vec in candidates], dtype=np.float64)
    score_range = raw_scores.max() - raw_scores.min()
    scores = (raw_scores - raw_scores.min()) / score_range if score_range > 0 else np.zeros_like(raw_scores)
    vectors = np.asarray([list(vec) for _cid, _score, vec in candidates], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms == 0, 1e-8, norms)

    n = len(ids)
    k = min(k, n)
    remaining = np.ones(n, dtype=bool)
    max_redundancy = np.zeros(n, dtype=np.float64)
    selected_idxs: list[int] = []

    for _ in range(k):
        mmr_scores = lambda_diversity * scores - (1 - lambda_diversity) * max_redundancy
        mmr_scores = np.where(remaining, mmr_scores, -np.inf)
        best_idx = int(np.argmax(mmr_scores))
        selected_idxs.append(best_idx)
        remaining[best_idx] = False
        similarity_to_new = normalized @ normalized[best_idx]
        max_redundancy = np.maximum(max_redundancy, similarity_to_new)

    return [ids[i] for i in selected_idxs]


def fuse_candidates(
    *,
    dense: Sequence[Any],
    sparse: Sequence[Any],
    strategy: str = "rrf",
    rrf_k: int = 60,
) -> list[Any]:
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
    except (TypeError, ValueError):
        k0 = 60

    dense_list = list(dense or [])
    sparse_list = list(sparse or [])

    dense_rank: dict[str, int] = {}
    sparse_rank: dict[str, int] = {}
    dense_item: dict[str, Any] = {}
    sparse_item: dict[str, Any] = {}

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

    scored: list[tuple[float, str]] = [(score_fn(sid), sid) for sid in ids]

    def _tie_key(pair: tuple[float, str]) -> tuple[float, int, int, int, int, str]:
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

    out: list[Any] = []
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
                logger.debug("fuse_candidates: skipped meta attachment for id=%s", sid, exc_info=True)
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
    trust_score: float = 0.0
    final_score_with_trust: float = 0.0


class Ranker:
    """
    Canonical ranking behavior for UMA retrieval.

    Ranking attaches derived features onto `.meta` (e.g., `rerank_score`, `final_score`)
    so downstream debugging is explainable.
    """

    def __init__(
        self,
        *,
        debug_scores: bool = False,
        trust_weight: float = 0.15,
        min_trust_score: float = 0.5,
        recency_decay_days: int = 90,
    ) -> None:
        self._debug = bool(debug_scores)
        self._trust_weight = max(0.0, min(1.0, float(trust_weight)))
        self._trust_alpha = 1.0 - self._trust_weight
        self._min_trust_score = max(0.0, float(min_trust_score))
        self._recency_decay_days = max(1, int(recency_decay_days))

    # ----------------------------- Public -----------------------------

    def truncate(self, items: Sequence[Any], k: int) -> list[Any]:
        """Truncate a ranked candidate list to ``max_items``, preserving order."""
        try:
            k_i = max(0, int(k))
        except (TypeError, ValueError):
            k_i = 0
        return list(items or [])[:k_i] if k_i else []

    def rank_facts(self, items: Sequence[Any], *, query_text: str = "", debug: bool = False) -> list[Any]:
        """Rank semantic facts by the trust-weighted blend of similarity and trust score."""
        items = list(items or [])
        if not items:
            return []

        terms = extract_terms(query_text or "")
        entities = extract_query_entities(query_text or "")

        scored: list[tuple[float, str, Any]] = []
        for f in items:
            rerank = compute_rerank_score(query_text=query_text or "", candidate=f, terms=terms, entities=entities)
            sal = _safe_float(get_attr_or_key(f, "salience", 0.0) or 0.0, default=0.0)
            conf = _safe_float(get_attr_or_key(f, "confidence", 0.5) or 0.5, default=0.5)
            quality = (max(0.0, sal) + max(0.0, conf)) / 2.0
            final = float(rerank) + (0.1 * float(quality))

            _update_meta(f, {"rerank_score": float(rerank), "final_score": float(final)})
            scored.append((float(final), _id(f), f))

        scored.sort(key=lambda x: (-x[0], x[1]))
        scored = self._apply_trust_weight(scored)
        self._emit_scorecards("facts", scored, debug=debug)
        return self._filter_by_trust([f for _s, _sid, f in scored])

    def rank_chunks(self, items: Sequence[Any], *, query_text: str = "", debug: bool = False) -> list[Any]:
        """Rank document chunks by the trust-weighted blend of similarity and trust score."""
        items = list(items or [])
        if not items:
            return []

        terms = extract_terms(query_text or "")
        entities = extract_query_entities(query_text or "")

        scored: list[tuple[int, float, str, int, str, Any]] = []
        for ch in items:
            m = _meta(ch)
            route = str(m.get("retrieval_route") or "")
            method = str(m.get("retrieval_method") or "")
            rerank = compute_rerank_score(query_text=query_text or "", candidate=ch, terms=terms, entities=entities)
            final = self._route_weight(route) + self._method_weight(method) + float(rerank)
            trust_raw = getattr(ch, "trust_score", None)
            trust = max(0.0, min(1.0, _safe_float(trust_raw if trust_raw is not None else 1.0)))
            final_with_trust = self._trust_alpha * final + self._trust_weight * trust
            _update_meta(ch, {"rerank_score": float(rerank), "final_score": float(final_with_trust)})

            route_pri = self._route_priority(route)
            scored.append((route_pri, float(final_with_trust), _doc_id(ch), _position(ch), _id(ch), ch))

        scored.sort(key=lambda x: (x[0], -x[1], x[2], x[3], x[4]))
        self._emit_scorecards("chunks", [(s, sid, it) for (_rp, s, _d, _p, sid, it) in scored], debug=debug)
        return self._filter_by_trust([ch for _rp, _s, _d, _p, _sid, ch in scored])

    def rank_episodes(self, items: Sequence[Any], *, query_text: str = "", debug: bool = False) -> list[Any]:
        """Rank episodic memories by the trust-weighted blend of similarity and trust score."""
        items = list(items or [])
        if not items:
            return []
        now = datetime.now(timezone.utc)
        terms = extract_terms(query_text or "")
        entities = extract_query_entities(query_text or "")
        scored: list[tuple[float, str, Any]] = []
        for ep in items:
            sid = _id(ep)
            rerank = compute_rerank_score(query_text=query_text or "", candidate=ep, terms=terms, entities=entities)

            recency = 0.0
            try:
                ts = get_attr_or_key(ep, "timestamp", None)
                if ts is not None:
                    if getattr(ts, "tzinfo", None) is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_days = (now - ts).total_seconds() / 86400.0
                    recency = max(0.0, 1.0 - age_days / self._recency_decay_days)
            except (TypeError, ValueError, AttributeError):
                recency = 0.0

            final = float(rerank) + (0.3 * float(recency))
            _update_meta(ep, {"rerank_score": float(rerank), "final_score": float(final)})
            scored.append((final, sid, ep))
        scored.sort(key=lambda x: (-x[0], x[1]))
        scored = self._apply_trust_weight(scored)
        self._emit_scorecards("episodes", scored, debug=debug)
        return self._filter_by_trust([ep for _s, _sid, ep in scored])

    def rank_skills(self, items: Sequence[Any], *, query_text: str = "", debug: bool = False) -> list[Any]:
        """Rank procedural skills by the trust-weighted blend of similarity and trust score."""
        items = list(items or [])
        if not items:
            return []
        terms = extract_terms(query_text or "")
        entities = extract_query_entities(query_text or "")
        scored: list[tuple[float, str, Any]] = []
        for sk in items:
            sid = _id(sk)
            rerank = compute_rerank_score(query_text=query_text or "", candidate=sk, terms=terms, entities=entities)
            phrases = get_attr_or_key(sk, "trigger_phrases", []) or []
            patterns = get_attr_or_key(sk, "trigger_patterns", []) or []
            diversity = len(set(phrases)) + len(set(patterns))
            final = float(rerank) + (0.05 * float(diversity))
            _update_meta(sk, {"rerank_score": float(rerank), "final_score": float(final)})
            scored.append((final, sid, sk))
        scored.sort(key=lambda x: (-x[0], x[1]))
        scored = self._apply_trust_weight(scored)
        self._emit_scorecards("skills", scored, debug=debug)
        return self._filter_by_trust([sk for _s, _sid, sk in scored])

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

    def _apply_trust_weight(
        self, scored: list[tuple[float, str, Any]]
    ) -> list[tuple[float, str, Any]]:
        """Blend existing final_score with trust_score and return re-sorted list.

        formula: final_with_trust = (1 - trust_weight) * existing + trust_weight * trust_score
        """
        result = []
        for existing_final, sid, obj in scored:
            trust_raw = getattr(obj, "trust_score", None)
            trust = max(0.0, min(1.0, _safe_float(trust_raw if trust_raw is not None else 1.0)))
            adjusted = self._trust_alpha * existing_final + self._trust_weight * trust
            _update_meta(obj, {"final_score": float(adjusted)})
            result.append((adjusted, sid, obj))
        result.sort(key=lambda x: (-x[0], x[1]))
        return result

    def _filter_by_trust(self, items: list[Any]) -> list[Any]:
        """Drop candidates whose trust_score is below the configured threshold (inclusive)."""
        if self._min_trust_score <= 0.0:
            return items
        out = []
        for it in items:
            trust_raw = getattr(it, "trust_score", None)
            trust = _safe_float(trust_raw if trust_raw is not None else 1.0)
            if trust >= self._min_trust_score:
                out.append(it)
        return out

    def _emit_scorecards(
        self, lane: str, scored: Iterable[tuple[float, str, Any]], *, debug: bool = False
    ) -> None:
        """Attach a per-candidate ``score_card`` to each object's meta.

        Emission is opt-in per request. ``debug`` is the request-scoped flag
        (``include_debug`` at the public API); ``self._debug`` is the global
        ``retrieval.debug_scores`` config default. Either enables emission.
        The flag is passed per call rather than held on the Ranker because a
        single Ranker instance is shared across concurrent requests.
        """
        if not (self._debug or debug):
            return
        cards: list[ScoreCard] = []
        for final, sid, obj in scored:
            m = _meta(obj)
            trust_raw = getattr(obj, "trust_score", None)
            trust = max(0.0, min(1.0, _safe_float(trust_raw if trust_raw is not None else 1.0)))
            card = ScoreCard(
                id=sid,
                vector_score=_safe_float(m.get("vector_score", 0.0) or 0.0),
                lexical_score=_safe_float(m.get("lexical_score", m.get("lexical_confidence", 0.0)) or 0.0),
                rerank_score=_safe_float(m.get("rerank_score", 0.0) or 0.0),
                route=str(m.get("retrieval_route") or ""),
                method=str(m.get("retrieval_method") or ""),
                final_score=float(final),
                trust_score=float(trust),
                final_score_with_trust=float(final),  # final is already trust-adjusted at emit time
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
                        "trust_score": float(card.trust_score),
                        "final_score_with_trust": float(card.final_score_with_trust),
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
