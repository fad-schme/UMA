from __future__ import annotations

"""
uma.retrieve.rlm.decisions
================================

This module defines the *only* action space the RLM controller is allowed to use.

Design goals
------------
- Store-native: actions map directly to safe, bounded memory store operations.
- Production-safe: strict validation of parameters, no arbitrary queries.
- No backwards-compat: UMA is v1 in active development.

Action space
------------
The controller may:
- vector-search semantic/episodic/procedural (bounded k)
- fetch by ids (bounded list length)
- graph neighbor expansion (bounded k, bounded depth)
- episodic cluster summaries (bounded k)
- stop
"""

import logging
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .domain import PREFERENCE_PREDICATES, filter_facts_by_domains
from .entity_seed import extract_candidate_entities
from uma.common.text import extract_keywords_and_phrases

logger = logging.getLogger(__name__)


MAX_FACT_IDS = 50


# --------------------------------------------------------------------------- #
# Action space — one subtype per action. Per-branch invariants are enforced
# structurally by pydantic v2, not by a runtime `if/elif` on a string enum.
# Adding an action means adding a class and extending the RetrievalAction
# union — no central validator to edit.
# --------------------------------------------------------------------------- #


class _RetrievalActionBase(BaseModel):
    """
    Fields shared across every action. Not instantiated directly.

    `extra="forbid"` preserves the invariant the old flat validator enforced
    for `stop` (and implicitly for the reject-lists on every other branch):
    unrecognized fields raise instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    reason: str = ""
    owner_type: Optional[Literal["user", "agent"]] = None


class SearchSemanticAction(_RetrievalActionBase):
    action: Literal["search_semantic"] = "search_semantic"
    k: int = Field(ge=1, le=500)
    filters: Optional[dict[str, Any]] = None


class FetchMoreFactsAction(_RetrievalActionBase):
    action: Literal["fetch_more_facts"] = "fetch_more_facts"
    k: int = Field(ge=1, le=500)
    predicate: str = Field(min_length=1)
    filters: Optional[dict[str, Any]] = None


class FetchFactsAction(_RetrievalActionBase):
    action: Literal["fetch_facts"] = "fetch_facts"
    ids: list[str] = Field(min_length=1, max_length=MAX_FACT_IDS)


class SearchEpisodicAction(_RetrievalActionBase):
    action: Literal["search_episodic"] = "search_episodic"
    k: int = Field(ge=1, le=500)
    filters: Optional[dict[str, Any]] = None


class EpisodicClustersAction(_RetrievalActionBase):
    action: Literal["episodic_clusters"] = "episodic_clusters"
    k: int = Field(ge=1, le=500)


class FetchEpisodeClustersAction(_RetrievalActionBase):
    action: Literal["fetch_episode_clusters"] = "fetch_episode_clusters"
    k: int = Field(ge=1, le=500)
    time_range: Optional[dict[str, Any]] = None
    min_salience: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SearchProceduralAction(_RetrievalActionBase):
    action: Literal["search_procedural"] = "search_procedural"
    k: int = Field(ge=1, le=500)
    filters: Optional[dict[str, Any]] = None


class FetchChunksAction(_RetrievalActionBase):
    action: Literal["fetch_chunks"] = "fetch_chunks"
    ids: list[str] = Field(min_length=1)


class SearchChunksAction(_RetrievalActionBase):
    action: Literal["search_chunks"] = "search_chunks"
    k: int = Field(ge=1, le=500)


class ExpandGraphAction(_RetrievalActionBase):
    action: Literal["expand_graph"] = "expand_graph"
    k: int = Field(ge=1, le=500)
    subject: str = Field(min_length=1)
    predicate: Optional[str] = None
    domain_scope: Optional[list[str]] = Field(default=None, max_length=10)
    direction: Optional[Literal["inbound", "outbound", "both"]] = None
    hops: int = Field(default=1, ge=1, le=3)


class GraphNeighborsAction(_RetrievalActionBase):
    action: Literal["graph_neighbors"] = "graph_neighbors"
    k: int = Field(ge=1, le=500)
    node_id: str = Field(min_length=1)
    predicate_scope: Optional[list[str]] = Field(default=None, max_length=20)
    domain_scope: Optional[list[str]] = Field(default=None, max_length=10)
    depth: int = Field(default=1, ge=1, le=3)


class StopAction(_RetrievalActionBase):
    action: Literal["stop"] = "stop"
    # No other fields. `extra="forbid"` on the base ensures the old
    # "stop action must not include any parameters" invariant survives.


# Discriminated union: pydantic dispatches on the `action` string literal at
# parse time. `RetrievalAction` is a type alias, not a class — every call
# site that used to write `RetrievalAction(action="X", ...)` now writes the
# specific subtype directly (e.g. `SearchSemanticAction(k=7)`).
RetrievalAction = Annotated[
    Union[
        SearchSemanticAction,
        FetchMoreFactsAction,
        FetchFactsAction,
        SearchEpisodicAction,
        EpisodicClustersAction,
        FetchEpisodeClustersAction,
        SearchProceduralAction,
        FetchChunksAction,
        SearchChunksAction,
        ExpandGraphAction,
        GraphNeighborsAction,
        StopAction,
    ],
    Field(discriminator="action"),
]


class ControllerDecision(BaseModel):
    """
    Output schema produced by the controller.

    - actions: ordered list of RetrievalAction
    - done: if true, controller terminates immediately
    """

    actions: list[RetrievalAction] = Field(default_factory=list)
    done: bool = False


def deterministic_decision(
    pack: Any,
    coverage: Any,
    *,
    cfg: dict[str, Any],
) -> Optional[ControllerDecision]:
    """
    Deterministic decision policy for RLM retrieval.

    This function contains no I/O. It translates the current state
    (ContextPack + CoverageReport) into a bounded list of RetrievalAction(s).
    """
    decision = _decide_zero_yield_fallback(pack, cfg)
    if decision is not None:
        return decision

    actions: list[RetrievalAction] = []
    actions.extend(_decide_semantic(pack, coverage, cfg))
    actions.extend(_decide_chunk_fallback(pack, cfg))
    actions.extend(_decide_episodic_clusters(pack, coverage, cfg))
    actions.extend(_decide_graph(pack, coverage, cfg))
    return ControllerDecision(actions=actions) if actions else None


def _decide_zero_yield_fallback(pack: Any, cfg: dict[str, Any]) -> Optional[ControllerDecision]:
    max_items_per_type = int(cfg.get("max_items_per_type", 30))
    chunk_fallback_enabled = bool(cfg.get("chunk_fallback_enabled", True))
    chunk_fallback_k_multiplier = max(1, int(cfg.get("chunk_fallback_k_multiplier", 2)))
    trace_id = (cfg.get("trace_id") if isinstance(cfg, dict) else None) or getattr(pack, "trace_id", None)
    intent = str(getattr(pack, "intent", "") or "").strip().lower()
    active_domains = list(getattr(pack, "active_domains", []) or [])

    last = _last_action_result(getattr(pack, "steps", []) or [])
    if not isinstance(last, dict):
        return None
    last_action = str(last.get("action") or "").strip()
    last_store = str(last.get("store") or "").strip()
    try:
        last_novelty = int(last.get("novelty") or 0)
    except Exception:
        last_novelty = 0
    active_domain_set = set(active_domains or [])
    chunks_allowed = "kb_doc" in active_domain_set
    chunks_used = bool(getattr(pack, "chunk_fallback_used", False)) or _did_action(
        getattr(pack, "steps", []) or [], "search_chunks"
    )

    if last_action == "fetch_more_facts" and last_novelty == 0 and last_store == "facts":
        if chunks_allowed and not chunks_used and chunk_fallback_enabled:
            logger.info(
                "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_chunks reason=fetch_more_facts_zero_yield",
                trace_id, (intent or "").upper(), active_domains,
            )
            action = SearchChunksAction(
                k=min(max_items_per_type * max(3, chunk_fallback_k_multiplier), 180),
                owner_type=getattr(pack, "owner_type", None),
            )
            try:
                setattr(pack, "chunk_fallback_used", True)
            except Exception:
                logger.exception("_decide_zero_yield_fallback: failed to mark chunk_fallback_used")
                raise
            return ControllerDecision(actions=[action])
        logger.info(
            "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_semantic reason=fetch_more_facts_zero_yield",
            trace_id, (intent or "").upper(), active_domains,
        )
        return ControllerDecision(actions=[SearchSemanticAction(
            k=max_items_per_type, owner_type=getattr(pack, "owner_type", None),
        )])

    if last_action == "expand_graph" and last_novelty == 0 and last_store == "graph":
        if chunks_allowed and not chunks_used and chunk_fallback_enabled:
            logger.info(
                "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_chunks reason=expand_graph_zero_yield",
                trace_id, (intent or "").upper(), active_domains,
            )
            action = SearchChunksAction(
                k=min(max_items_per_type * max(3, chunk_fallback_k_multiplier), 180),
                owner_type=getattr(pack, "owner_type", None),
            )
            try:
                setattr(pack, "chunk_fallback_used", True)
            except Exception:
                logger.exception("_decide_zero_yield_fallback: failed to mark chunk_fallback_used")
                raise
            return ControllerDecision(actions=[action])
        logger.info(
            "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_semantic reason=expand_graph_zero_yield",
            trace_id, (intent or "").upper(), active_domains,
        )
        return ControllerDecision(actions=[SearchSemanticAction(
            k=max_items_per_type, owner_type=getattr(pack, "owner_type", None),
        )])

    return None


def _decide_semantic(pack: Any, coverage: Any, cfg: dict[str, Any]) -> list[RetrievalAction]:
    if not getattr(coverage, "needs_semantic", False):
        return []
    max_items_per_type = int(cfg.get("max_items_per_type", 30))
    graph_predicate_limit = max(1, int(cfg.get("graph_predicate_limit", 2)))
    predicate_allowlist = cfg.get("predicate_allowlist") if isinstance(cfg, dict) else None
    trace_id = (cfg.get("trace_id") if isinstance(cfg, dict) else None) or getattr(pack, "trace_id", None)
    intent = str(getattr(pack, "intent", "") or "").strip().lower()
    active_domains = list(getattr(pack, "active_domains", []) or [])
    actions: list[RetrievalAction] = []
    facts = getattr(pack, "facts", []) or []
    if facts:
        eligible, best_pred, best_score, reasons = _debug_score_predicates(
            pack=pack,
            facts=list(facts),
            graph_predicate_limit=graph_predicate_limit,
            predicate_allowlist=predicate_allowlist,
        )
        predicate, score = _select_predicate_for_expansion(
            pack=pack,
            facts=list(facts),
            graph_predicate_limit=graph_predicate_limit,
            predicate_allowlist=predicate_allowlist,
        )
        logger.info(
            "RLM_DECISION trace_id=%s intent=%s domains=%s eligible_predicates=%s selected_predicate=%s score=%d reasons=%s",
            trace_id, (intent or "").upper(), active_domains, eligible, predicate, int(score or 0), reasons,
        )
        if predicate and score > 0:
            offset = getattr(pack, "get_predicate_offset")(predicate)
            actions.append(FetchMoreFactsAction(
                predicate=predicate,
                k=max_items_per_type,
                filters={"offset": offset},
                owner_type=getattr(pack, "owner_type", None),
            ))
            getattr(pack, "bump_predicate_offset")(predicate, max_items_per_type)
        else:
            # No predicate is relevant to the query => broaden search instead of expanding by predicate.
            logger.info(
                "RLM_DECISION trace_id=%s intent=%s domains=%s eligible_predicates=%s fallback=search_semantic reason=no_relevant_predicates",
                trace_id, (intent or "").upper(), active_domains, eligible,
            )
            actions.append(SearchSemanticAction(
                k=max_items_per_type,
                owner_type=getattr(pack, "owner_type", None),
            ))
    else:
        logger.info(
            "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_semantic reason=no_facts_in_pack",
            trace_id, (intent or "").upper(), active_domains,
        )
        actions.append(SearchSemanticAction(
            k=max_items_per_type,
            owner_type=getattr(pack, "owner_type", None),
        ))
    return actions


def _decide_chunk_fallback(pack: Any, cfg: dict[str, Any]) -> list[RetrievalAction]:
    if not bool(cfg.get("chunk_fallback_enabled", True)):
        return []
    if bool(getattr(pack, "chunk_fallback_used", False)):
        return []
    if len(getattr(pack, "chunks", []) or []) > 0:
        return []
    max_items_per_type = int(cfg.get("max_items_per_type", 30))
    chunk_fallback_k_multiplier = max(1, int(cfg.get("chunk_fallback_k_multiplier", 2)))
    try:
        setattr(pack, "chunk_fallback_used", True)
    except Exception:
        logger.exception("_decide_chunk_fallback: failed to mark chunk_fallback_used")
        raise
    return [SearchChunksAction(
        k=min(max_items_per_type * chunk_fallback_k_multiplier, 120),
        owner_type=getattr(pack, "owner_type", None),
    )]


def _decide_episodic_clusters(pack: Any, coverage: Any, cfg: dict[str, Any]) -> list[RetrievalAction]:
    if not getattr(coverage, "needs_clusters", False):
        return []
    cluster_k = max(1, int(cfg.get("cluster_k", 3)))
    salience_threshold = float(cfg.get("salience_threshold", 0.6))
    max_items_per_type = int(cfg.get("max_items_per_type", 30))
    owner_type = getattr(pack, "owner_type", None)

    if bool(cfg.get("episodic_clustering_available", False)):
        # Enterprise: compiled cluster summaries are the primary path.
        episodes = getattr(pack, "episodes", []) or []
        has_cluster = any(isinstance(ep, dict) and "episode_ids" in ep for ep in episodes)
        actions: list[RetrievalAction] = [FetchEpisodeClustersAction(
            k=cluster_k,
            time_range=None,
            min_salience=salience_threshold,
            owner_type=owner_type,
        )]
        if not has_cluster and len(getattr(pack, "steps", []) or []) >= 2:
            actions.append(SearchEpisodicAction(
                k=max_items_per_type,
                owner_type=owner_type,
            ))
        return actions

    # Lite/cont: no consolidation — direct vector search over raw episodes.
    return [SearchEpisodicAction(
        k=max_items_per_type,
        owner_type=owner_type,
    )]


def _decide_graph(pack: Any, coverage: Any, cfg: dict[str, Any]) -> list[RetrievalAction]:
    if not bool(cfg.get("graph_expansion_available", False)):
        return []
    graph = getattr(pack, "graph", []) or []
    facts = getattr(pack, "facts", []) or []
    chunks = getattr(pack, "chunks", []) or []
    if (
        graph
        or not (facts or chunks)
        or getattr(coverage, "needs_semantic", False)
        or getattr(coverage, "needs_clusters", False)
    ):
        return []
    max_items_per_type = int(cfg.get("max_items_per_type", 30))
    graph_predicate_limit = max(1, int(cfg.get("graph_predicate_limit", 2)))
    predicate_allowlist = cfg.get("predicate_allowlist") if isinstance(cfg, dict) else None
    trace_id = (cfg.get("trace_id") if isinstance(cfg, dict) else None) or getattr(pack, "trace_id", None)
    intent = str(getattr(pack, "intent", "") or "").strip().lower()
    active_domains = list(getattr(pack, "active_domains", []) or [])
    actions: list[RetrievalAction] = []

    # PERSONAL intent: user-based expansion (LIKES/PREFERS lane).
    if intent == "personal" and getattr(pack, "owner_type", None) == "user":
        next_scope = cfg.get("next_predicate_scope")
        predicate_scope = next_scope(pack, graph_predicate_limit) if callable(next_scope) else []
        if predicate_scope:
            logger.info(
                "RLM_DECISION trace_id=%s intent=%s domains=%s graph_seed=user_id predicate_scope=%s",
                trace_id, (intent or "").upper(), active_domains,
                predicate_scope[: max(1, int(graph_predicate_limit))],
            )
            actions.append(ExpandGraphAction(
                subject=getattr(pack, "user_id", None),
                predicate=predicate_scope[0],
                domain_scope=["user_profile"],
                hops=1,
                direction="outbound",
                k=min(max_items_per_type, 20),
                owner_type=getattr(pack, "owner_type", None),
            ))
        return actions

    # TOPICAL / MIXED: seed graph expansion from evidence-derived entities, not user_id.
    kb_facts = filter_facts_by_domains(list(facts), allowed_domains={"kb_doc"})
    eligible_g, _best_g, _best_score_g, reasons_g = _debug_score_predicates(
        pack=pack,
        facts=kb_facts,
        graph_predicate_limit=graph_predicate_limit,
        predicate_allowlist=predicate_allowlist,
    )
    predicate, score = _select_predicate_for_expansion(
        pack=pack,
        facts=kb_facts,
        graph_predicate_limit=graph_predicate_limit,
        predicate_allowlist=predicate_allowlist,
    )
    fallback_reason: Optional[str] = None
    if not predicate or score <= 0:
        # Fall back to a stable observed kb_doc predicate scope (never user_profile).
        preds = _top_predicates_from_facts(kb_facts, graph_predicate_limit)
        predicate = preds[0] if preds else None
        fallback_reason = "no_relevant_predicates_for_graph"

    entities = extract_candidate_entities(
        query_text=getattr(pack, "query_text", "") or "",
        facts=list(facts),
        chunks=list(chunks),
        limit=5,
    )
    excluded_reasons: list[str] = []
    if intent in {"topical", "mixed"} and "user_profile" not in set(active_domains or []):
        try:
            observed = _observed_predicates(list(facts))
            if any(p in PREFERENCE_PREDICATES for p in observed):
                excluded_reasons.append("excluded_user_profile_predicates_due_to_intent")
        except Exception:  # nosec B110
            logger.debug("decisions: intent predicate filter failed", exc_info=True)
            pass
    combined_reasons: Optional[list[str]] = None
    try:
        rs: list[str] = []
        if isinstance(reasons_g, list):
            rs.extend([r for r in reasons_g if isinstance(r, str) and r])
        rs.extend([r for r in excluded_reasons if isinstance(r, str) and r])
        combined_reasons = rs or None
    except Exception:
        combined_reasons = excluded_reasons or None
    logger.info(
        "RLM_DECISION trace_id=%s intent=%s domains=%s graph_seed=entities eligible_predicates=%s selected_predicate=%s score=%d selected_entities=%s fallback=%s reasons=%s",
        trace_id, (intent or "").upper(), active_domains, eligible_g, predicate,
        int(score or 0), (entities or [])[:5], fallback_reason, combined_reasons,
    )
    # Keep bounded: expand around at most 2 topical entities per step.
    for ent in (entities or [])[:2]:
        actions.append(ExpandGraphAction(
            subject=ent,
            predicate=predicate,
            domain_scope=["kb_doc"],
            hops=1,
            direction="both",
            k=min(max_items_per_type, 20),
            owner_type=getattr(pack, "owner_type", None),
        ))
    return actions


def _extract_query_terms(query_text: str) -> list[str]:
    try:
        extracted = extract_keywords_and_phrases(query_text or "")
    except Exception:
        extracted = {"keywords": [], "keyphrases": []}
    terms = []
    for key in ("keyphrases", "keywords"):
        vals = extracted.get(key) if isinstance(extracted, dict) else None
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, str) and v.strip():
                    terms.append(v.strip().lower())
    # unique, preserve order
    seen = set()
    out: list[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _fact_blob(fact: Any) -> str:
    parts: list[str] = []
    try:
        subj = fact.get("subject") if isinstance(fact, dict) else getattr(fact, "subject", None)
        if subj:
            parts.append(str(subj))
    except Exception:  # nosec B110 — optional field; missing subject contributes nothing to blob
        pass
    try:
        obj = fact.get("object") if isinstance(fact, dict) else getattr(fact, "object", None)
        if obj:
            parts.append(str(obj))
    except Exception:  # nosec B110 — optional field; missing object contributes nothing to blob
        pass
    try:
        meta = fact.get("meta") if isinstance(fact, dict) else getattr(fact, "meta", None)
        if isinstance(meta, dict):
            ft = meta.get("fact_text")
            sp = meta.get("source_path")
            if ft:
                parts.append(str(ft))
            if sp:
                parts.append(str(sp))
    except Exception:  # nosec B110 — optional field; missing meta contributes nothing to blob
        pass
    return " ".join(parts).lower()


def _word_root(word: str) -> str:
    """Strip common inflectional suffixes for fuzzy term matching."""
    w = word.lower()
    for suffix in ("ing", "tion", "sion", "ness", "ment", "ed", "er", "ly", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _predicate_score(predicate: str, facts: list[Any], terms: list[str]) -> int:
    if not predicate or not terms or not facts:
        return 0
    blob = " ".join(_fact_blob(f) for f in facts if f)[:20000]
    score = 0
    for t in terms:
        if not t:
            continue
        if t in blob or _word_root(t) in blob:
            score += 1
    return score


def _select_predicate_for_expansion(
    *,
    pack: Any,
    facts: list[Any],
    graph_predicate_limit: int,
    predicate_allowlist: Any,
) -> tuple[Optional[str], int]:
    """
    Select the most query-relevant predicate for semantic expansion.

    Rules (from the provided instructions):
    1) Candidate facts are those within active domains.
    2) Candidate predicates are unique predicates observed in those facts.
    3) Score each predicate by query term overlap against a fact blob.
    4) Pick top predicate(s).
    5) If no predicate scores > 0, do not expand by predicate.
    """
    active_domains = set(getattr(pack, "active_domains", []) or [])
    candidate_facts = filter_facts_by_domains(list(facts or []), active_domains) if active_domains else list(facts or [])

    pred_to_facts: dict[str, list[Any]] = {}
    for f in candidate_facts:
        try:
            pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
            if not pred:
                continue
            key = str(pred).strip().upper()
            if not key:
                continue
            pred_to_facts.setdefault(key, []).append(f)
        except Exception:  # nosec B112 — malformed fact skipped; scoring continues
            logger.debug("decisions: predicate extraction failed for fact, skipping", exc_info=True)
            continue

    candidates = list(pred_to_facts.keys())
    if not candidates:
        return None, 0

    # Optional domain-aware allowlist: intersect candidates with configured allowlist.
    observed = list(candidates)
    allowed: set[str] = set()
    if isinstance(predicate_allowlist, dict) and active_domains:
        for dom in active_domains:
            preds = predicate_allowlist.get(dom)
            if isinstance(preds, list):
                for p in preds:
                    if isinstance(p, str) and p.strip():
                        allowed.add(p.strip().upper())
    if allowed:
        candidates = [p for p in candidates if p in allowed]
        # If intersection is empty, fall back to observed predicates.
        if not candidates:
            candidates = observed

    terms = _extract_query_terms(getattr(pack, "query_text", "") or "")
    if not terms:
        return None, 0

    scored: list[tuple[str, int]] = []
    for p in candidates:
        score = _predicate_score(p, pred_to_facts.get(p, []), terms)
        scored.append((p, int(score)))

    scored.sort(key=lambda x: (-x[1], x[0]))
    top_k = max(1, int(graph_predicate_limit))
    best_pred, best_score = scored[0]
    # (We compute top_k for future multi-predicate expansions; currently we return the best.)
    _ = scored[:top_k]
    return best_pred, best_score


def _observed_predicates(facts: list[Any]) -> set[str]:
    out: set[str] = set()
    for f in facts or []:
        try:
            pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
            if not pred:
                continue
            s = str(pred).strip().upper()
            if s:
                out.add(s)
        except Exception:  # nosec B112 — malformed fact skipped; set continues building
            logger.debug("decisions: predicate observation failed for fact, skipping", exc_info=True)
            continue
    return out


def _debug_score_predicates(
    *,
    pack: Any,
    facts: list[Any],
    graph_predicate_limit: int,
    predicate_allowlist: Any,
    top_n: int = 8,
) -> tuple[list[str], Optional[str], int, Optional[list[str]]]:
    """
    Debug-only predicate scoring for observability logs.

    Returns:
    - eligible_predicates (top N, ordered)
    - best_predicate
    - best_score
    - reasons (optional)
    """
    reasons: list[str] = []
    active_domains = set(getattr(pack, "active_domains", []) or [])
    if active_domains and "user_profile" not in active_domains:
        reasons.append("excluded_user_profile_predicates_due_to_domains")

    candidate_facts = (
        filter_facts_by_domains(list(facts or []), active_domains)
        if active_domains
        else list(facts or [])
    )

    pred_to_facts: dict[str, list[Any]] = {}
    for f in candidate_facts:
        try:
            pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
            if not pred:
                continue
            key = str(pred).strip().upper()
            if not key:
                continue
            pred_to_facts.setdefault(key, []).append(f)
        except Exception:  # nosec B112 — malformed fact skipped; debug scoring continues
            logger.debug("decisions: debug predicate extraction failed for fact, skipping", exc_info=True)
            continue

    candidates = list(pred_to_facts.keys())
    if not candidates:
        return [], None, 0, reasons or None

    observed = list(candidates)
    allowed: set[str] = set()
    if isinstance(predicate_allowlist, dict) and active_domains:
        for dom in active_domains:
            preds = predicate_allowlist.get(dom)
            if isinstance(preds, list):
                for p in preds:
                    if isinstance(p, str) and p.strip():
                        allowed.add(p.strip().upper())
    if allowed:
        candidates = [p for p in candidates if p in allowed]
        if not candidates:
            candidates = observed
        else:
            reasons.append("predicate_allowlist_applied")

    terms = _extract_query_terms(getattr(pack, "query_text", "") or "")
    if not terms:
        return sorted(candidates)[: max(1, int(top_n))], None, 0, reasons or None

    scored: list[tuple[str, int]] = []
    for p in candidates:
        score = _predicate_score(p, pred_to_facts.get(p, []), terms)
        scored.append((p, int(score)))
    scored.sort(key=lambda x: (-x[1], x[0]))

    eligible_predicates = [p for (p, _s) in scored[: max(1, int(top_n))]]
    best_pred, best_score = scored[0]
    # Keep deterministic ordering, but limit the returned scored set.
    _ = scored[: max(1, int(graph_predicate_limit))]
    return eligible_predicates, best_pred, int(best_score), reasons or None


def _last_action_result(steps: list[Any]) -> Optional[dict[str, Any]]:
    for s in reversed(steps or []):
        if isinstance(s, dict) and s.get("event") == "action_result":
            return s
    return None


def _did_action(steps: list[Any], action_name: str) -> bool:
    name = str(action_name or "").strip()
    if not name:
        return False
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        if s.get("event") != "action_result":
            continue
        if str(s.get("action") or "").strip() == name:
            return True
    return False


def next_predicate_scope(
    *,
    facts: list[Any],
    predicate_weights: Optional[dict[str, float]],
    graph_predicate_limit: int,
) -> list[str]:
    """
    Determine a bounded predicate exploration scope based on:
    - configured predicate weights (highest first)
    - observed predicate frequency in current facts
    """
    ordered: list[str] = []

    if predicate_weights:
        for p, _ in sorted(predicate_weights.items(), key=lambda kv: (-float(kv[1]), str(kv[0]))):
            ordered.append(str(p).upper())

    counts: dict[str, int] = {}
    for f in facts or []:
        pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
        if pred:
            key = str(pred).upper()
            counts[key] = counts.get(key, 0) + 1

    for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if p not in ordered:
            ordered.append(p)

    if not ordered:
        ordered = ["RELATED_TO"]

    try:
        limit = max(1, int(graph_predicate_limit))
    except Exception:
        limit = 2
    return ordered[:limit]


def _top_predicates_from_facts(facts: list[Any], limit: int) -> list[str]:
    """
    Deterministic predicate scope derived from the observed facts (no config weights).

    Used to ensure topical graph expansion uses kb_doc predicates even when MIXED intent
    includes user_profile facts.
    """
    counts: dict[str, int] = {}
    for f in facts or []:
        try:
            pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
            if not pred:
                continue
            key = str(pred).strip().upper()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
        except Exception:  # nosec B112 — malformed fact skipped; frequency count continues
            logger.debug("decisions: predicate frequency count failed for fact, skipping", exc_info=True)
            continue
    ordered = [p for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if not ordered:
        ordered = ["RELATED_TO"]
    return ordered[: max(1, int(limit))]


#
# Action execution is centralized in UMAMemoryEnvironment.execute_action.
