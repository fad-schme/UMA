from __future__ import annotations

"""
uma.retrieve.rlm.decisions
================================

This module defines the *only* action space the RLM controller is allowed to use.

Design goals
------------
- Store-native: actions map directly to safe, bounded memory store operations.
- Production-safe: strict validation of parameters, no arbitrary queries.
- No backwards-compat: UMA-RLM is v1 in active development.

Action space
------------
The controller may:
- vector-search semantic/episodic/procedural (bounded k)
- fetch by ids (bounded list length)
- graph neighbor expansion (bounded k, bounded depth)
- episodic cluster summaries (bounded k)
- stop
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional, NoReturn, Set, Tuple

from pydantic import BaseModel, Field, ValidationError, model_validator

from .domain import PREFERENCE_PREDICATES, filter_facts_by_domains
from .entity_seed import extract_candidate_entities
from uma.retrieve.user_query_helper import extract_keywords_and_phrases

logger = logging.getLogger(__name__)


MAX_FACT_IDS = 50

ActionType = Literal[
    "search_semantic",
    "fetch_more_facts",              # ← NEW
    "search_episodic",
    "episodic_clusters",
    "fetch_episode_clusters",
    "search_procedural",
    "fetch_facts",
    "fetch_chunks",
    "search_chunks",
    "graph_neighbors",
    "expand_graph",
    "stop",
]


class RetrievalAction(BaseModel):
    """
    A single bounded retrieval action.

    IMPORTANT:
    - Each action has a strict contract.
    - Invalid combinations are rejected.
    """

    action: ActionType
    reason: str = ""

    # Common bounds
    k: Optional[int] = Field(default=None, ge=1, le=500)

    # Semantic / procedural
    filters: Optional[Dict[str, Any]] = None

    # Semantic refinement (NEW)
    predicate: Optional[str] = None
    owner_type: Optional[Literal["user", "agent"]] = None

    # Episodic
    time_range: Optional[Dict[str, Any]] = None
    min_salience: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Fetch-by-id
    ids: Optional[List[str]] = None

    # Graph
    node_id: Optional[str] = None
    predicate_scope: Optional[List[str]] = None
    domain_scope: Optional[List[str]] = None
    depth: Optional[int] = Field(default=None, ge=1, le=3)
    subject: Optional[str] = None
    direction: Optional[str] = None
    hops: Optional[int] = Field(default=None, ge=1, le=3)

    # Conflict resolution
    fact_ids: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_action(self) -> "RetrievalAction":
        """
        Enforce per-action contracts.

        This prevents the controller from emitting ambiguous or unsafe actions.
        """
        def _raise(msg: str) -> "NoReturn":
            logger.error("RetrievalAction.validate_action failed: %s", msg)
            raise ValueError(msg)

        a = self.action

        # --- STOP ---
        if a == "stop":
            if any(
                [
                    self.k,
                    self.filters,
                    self.time_range,
                    self.ids,
                    self.node_id,
                    self.predicate_scope,
                    self.domain_scope,
                    self.depth,
                ]
            ):
                _raise("stop action must not include any parameters")
            return self

        # --- SEARCH SEMANTIC ---
        if a == "search_semantic":
            if self.k is None:
                _raise("search_semantic requires k")
            if self.ids or self.node_id:
                _raise("search_semantic does not accept ids or node_id")
            return self

        # --- SEARCH EPISODIC ---
        if a == "search_episodic":
            if self.k is None:
                _raise("search_episodic requires k")
            if self.ids or self.node_id:
                _raise("search_episodic does not accept ids or node_id")
            return self

        # --- EPISODIC CLUSTERS ---
        if a == "episodic_clusters":
            if self.k is None:
                _raise("episodic_clusters requires k")
            if self.filters or self.ids or self.node_id:
                _raise("episodic_clusters does not accept filters, ids, or node_id")
            return self

        if a == "fetch_episode_clusters":
            if self.k is None:
                _raise("fetch_episode_clusters requires k")
            if self.filters or self.ids or self.node_id:
                _raise("fetch_episode_clusters does not accept filters, ids, or node_id")
            if self.min_salience is not None and not (0 <= self.min_salience <= 1):
                _raise("min_salience must be between 0 and 1")
            return self

        # --- SEARCH PROCEDURAL ---
        if a == "search_procedural":
            if self.k is None:
                _raise("search_procedural requires k")
            if self.ids or self.node_id:
                _raise("search_procedural does not accept ids or node_id")
            return self

        # --- FETCH FACTS ---
        if a == "fetch_facts":
            if not self.ids:
                _raise("fetch_facts requires ids")
            if self.k or self.filters or self.node_id:
                _raise("fetch_facts does not accept k, filters, or node_id")
            return self

        # --- FETCH CHUNKS ---
        if a == "fetch_chunks":
            if not self.ids:
                _raise("fetch_chunks requires ids")
            if self.k or self.filters or self.node_id or self.time_range:
                _raise("fetch_chunks does not accept k, filters, node_id, or time_range")
            return self

        # --- SEARCH CHUNKS ---
        if a == "search_chunks":
            if self.k is None:
                _raise("search_chunks requires k")
            if self.ids or self.node_id or self.time_range:
                _raise("search_chunks does not accept ids, node_id, or time_range")
            return self
        
        # --- FETCH MORE FACTS (predicate-scoped semantic expansion) ---
        if a == "fetch_more_facts":
            if self.k is None:
                _raise("fetch_more_facts requires k")
            if not self.predicate or not isinstance(self.predicate, str):
                _raise("fetch_more_facts requires a non-empty predicate")
            if self.ids or self.node_id or self.time_range:
                _raise("fetch_more_facts does not accept ids, node_id, or time_range")
            if self.owner_type and self.owner_type not in {"user", "agent"}:
                _raise("owner_type must be one of: user, agent")
            return self
        
        if a == "expand_graph":
            if not self.subject or not isinstance(self.subject, str):
                _raise("expand_graph requires a non-empty subject")
            if self.k is None:
                _raise("expand_graph requires k")
            if self.filters or self.ids or self.time_range:
                _raise("expand_graph does not accept filters, ids, or time_range")
            if self.node_id:
                _raise("expand_graph does not accept node_id")
            if self.domain_scope is not None:
                if not isinstance(self.domain_scope, list):
                    _raise("domain_scope must be a list")
                if len(self.domain_scope) > 10:
                    _raise("domain_scope too large (max 10)")
            if self.direction:
                dir_val = str(self.direction).lower()
                if dir_val not in {"inbound", "outbound", "both"}:
                    _raise("direction must be one of: inbound, outbound, both")
                self.direction = dir_val
            if self.hops is None:
                self.hops = 1
            return self


        # --- GRAPH NEIGHBORS ---
        if a == "graph_neighbors":
            if not self.node_id:
                _raise("graph_neighbors requires node_id")
            if self.k is None:
                _raise("graph_neighbors requires k")
            if self.depth is None:
                self.depth = 1
            if self.predicate_scope:
                if not isinstance(self.predicate_scope, list):
                    _raise("predicate_scope must be a list")
                if len(self.predicate_scope) > 20:
                    _raise("predicate_scope too large (max 20)")
            if self.domain_scope is not None:
                if not isinstance(self.domain_scope, list):
                    _raise("domain_scope must be a list")
                if len(self.domain_scope) > 10:
                    _raise("domain_scope too large (max 10)")
            return self

        _raise(f"Unknown action: {a}")


class ControllerDecision(BaseModel):
    """
    Output schema produced by the controller.

    - actions: ordered list of RetrievalAction
    - done: if true, controller terminates immediately
    """

    actions: List[RetrievalAction] = Field(default_factory=list)
    done: bool = False

    @classmethod
    def from_json(cls, raw: str) -> "ControllerDecision":
        """
        Parse controller output robustly.

        Controller MUST emit JSON. We attempt recovery but do not
        accept structurally invalid decisions.
        """
        try:
            return cls.model_validate_json(raw)
        except ValidationError:
            cleaned = (raw or "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(cleaned[start : end + 1])
                return cls.model_validate(parsed)
            logger.exception("ControllerDecision.from_json failed to parse decision.")
            raise


def deterministic_decision(
    pack: Any,
    coverage: Any,
    *,
    cfg: Dict[str, Any],
) -> Optional[ControllerDecision]:
    """
    Deterministic decision policy for RLM retrieval.

    This function contains no I/O. It translates the current state
    (ContextPack + CoverageReport) into a bounded list of RetrievalAction(s).
    """
    actions: List[RetrievalAction] = []

    max_items_per_type = int(cfg.get("max_items_per_type", 30))
    cluster_k = int(cfg.get("cluster_k", 3))
    salience_threshold = float(cfg.get("salience_threshold", 0.6))
    graph_predicate_limit = int(cfg.get("graph_predicate_limit", 2))

    chunk_fallback_enabled = bool(cfg.get("chunk_fallback_enabled", True))
    chunk_fallback_k_multiplier = max(1, int(cfg.get("chunk_fallback_k_multiplier", 2)))
    predicate_allowlist = cfg.get("predicate_allowlist") if isinstance(cfg, dict) else None
    trace_id = (cfg.get("trace_id") if isinstance(cfg, dict) else None) or getattr(pack, "trace_id", None)
    intent = str(getattr(pack, "intent", "") or "").strip().lower()
    active_domains = list(getattr(pack, "active_domains", []) or [])

    # ------------------------------------------------------------------
    # Zero-yield fallback ladder (Phase 4)
    # ------------------------------------------------------------------
    last = _last_action_result(getattr(pack, "steps", []) or [])
    if isinstance(last, dict):
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

        # 1) fetch_more_facts 0 novelty => try chunks once; else broaden semantic.
        if last_action == "fetch_more_facts" and last_novelty == 0 and last_store == "facts":
            if chunks_allowed and not chunks_used and chunk_fallback_enabled:
                logger.info(
                    "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_chunks reason=fetch_more_facts_zero_yield",
                    trace_id,
                    (intent or "").upper(),
                    active_domains,
                )
                actions.append(
                    RetrievalAction(
                        action="search_chunks",
                        k=min(max_items_per_type * max(3, chunk_fallback_k_multiplier), 180),
                        owner_type=getattr(pack, "owner_type", None),
                    )
                )
                try:
                    setattr(pack, "chunk_fallback_used", True)
                except Exception:
                    logger.exception("deterministic_decision: failed to mark chunk_fallback_used (fallback ladder)")
                    raise
                return ControllerDecision(actions=actions)

            logger.info(
                "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_semantic reason=fetch_more_facts_zero_yield",
                trace_id,
                (intent or "").upper(),
                active_domains,
            )
            actions.append(
                RetrievalAction(
                    action="search_semantic",
                    k=max_items_per_type,
                    owner_type=getattr(pack, "owner_type", None),
                )
            )
            return ControllerDecision(actions=actions)

        # 2) expand_graph 0 novelty => try semantic or chunks.
        if last_action == "expand_graph" and last_novelty == 0 and last_store == "graph":
            if chunks_allowed and not chunks_used and chunk_fallback_enabled:
                logger.info(
                    "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_chunks reason=expand_graph_zero_yield",
                    trace_id,
                    (intent or "").upper(),
                    active_domains,
                )
                actions.append(
                    RetrievalAction(
                        action="search_chunks",
                        k=min(max_items_per_type * max(3, chunk_fallback_k_multiplier), 180),
                        owner_type=getattr(pack, "owner_type", None),
                    )
                )
                try:
                    setattr(pack, "chunk_fallback_used", True)
                except Exception:
                    logger.exception("deterministic_decision: failed to mark chunk_fallback_used (fallback ladder)")
                    raise
                return ControllerDecision(actions=actions)

            logger.info(
                "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_semantic reason=expand_graph_zero_yield",
                trace_id,
                (intent or "").upper(),
                active_domains,
            )
            actions.append(
                RetrievalAction(
                    action="search_semantic",
                    k=max_items_per_type,
                    owner_type=getattr(pack, "owner_type", None),
                )
            )
            return ControllerDecision(actions=actions)

    # --- Semantic ---
    if getattr(coverage, "needs_semantic", False):
        facts = getattr(pack, "facts", []) or []
        if facts:
            eligible, best_pred, best_score, reasons = _debug_score_predicates(
                pack=pack,
                facts=list(facts or []),
                graph_predicate_limit=graph_predicate_limit,
                predicate_allowlist=predicate_allowlist,
            )
            predicate, score = _select_predicate_for_expansion(
                pack=pack,
                facts=list(facts or []),
                graph_predicate_limit=graph_predicate_limit,
                predicate_allowlist=predicate_allowlist,
            )
            logger.info(
                "RLM_DECISION trace_id=%s intent=%s domains=%s eligible_predicates=%s selected_predicate=%s score=%d reasons=%s",
                trace_id,
                (intent or "").upper(),
                active_domains,
                eligible,
                predicate,
                int(score or 0),
                reasons,
            )
            if predicate and score > 0:
                offset = getattr(pack, "get_predicate_offset")(predicate)
                actions.append(
                    RetrievalAction(
                        action="fetch_more_facts",
                        predicate=predicate,
                        k=max_items_per_type,
                        filters={"offset": offset},
                        owner_type=getattr(pack, "owner_type", None),
                    )
                )
                getattr(pack, "bump_predicate_offset")(predicate, max_items_per_type)
            else:
                # No predicate is relevant to the query => broaden search instead of expanding by predicate.
                logger.info(
                    "RLM_DECISION trace_id=%s intent=%s domains=%s eligible_predicates=%s fallback=search_semantic reason=no_relevant_predicates",
                    trace_id,
                    (intent or "").upper(),
                    active_domains,
                    eligible,
                )
                actions.append(
                    RetrievalAction(
                        action="search_semantic",
                        k=max_items_per_type,
                        owner_type=getattr(pack, "owner_type", None),
                    )
                )
        else:
            logger.info(
                "RLM_DECISION trace_id=%s intent=%s domains=%s fallback=search_semantic reason=no_facts_in_pack",
                trace_id,
                (intent or "").upper(),
                active_domains,
            )
            actions.append(
                RetrievalAction(
                    action="search_semantic",
                    k=max_items_per_type,
                    owner_type=getattr(pack, "owner_type", None),
                )
            )

    # --- Chunk fallback (at most once) ---
    if chunk_fallback_enabled and not bool(getattr(pack, "chunk_fallback_used", False)):
        chunks = getattr(pack, "chunks", []) or []
        if len(chunks) == 0:
            actions.append(
                RetrievalAction(
                    action="search_chunks",
                    k=min(max_items_per_type * chunk_fallback_k_multiplier, 120),
                    owner_type=getattr(pack, "owner_type", None),
                )
            )
            try:
                setattr(pack, "chunk_fallback_used", True)
            except Exception:
                logger.exception("deterministic_decision: failed to mark chunk_fallback_used")
                raise

    # --- Episodic / clusters ---
    if getattr(coverage, "needs_clusters", False):
        episodes = getattr(pack, "episodes", []) or []
        has_cluster = any(isinstance(ep, dict) and "episode_ids" in ep for ep in episodes)
        actions.append(
            RetrievalAction(
                action="fetch_episode_clusters",
                k=cluster_k,
                time_range=None,
                min_salience=salience_threshold,
                owner_type=getattr(pack, "owner_type", None),
            )
        )
        if not has_cluster and len(getattr(pack, "steps", []) or []) >= 2:
            actions.append(
                RetrievalAction(
                    action="search_episodic",
                    k=max_items_per_type,
                    owner_type=getattr(pack, "owner_type", None),
                )
            )

    # --- Graph last-mile ---
    graph = getattr(pack, "graph", []) or []
    facts = getattr(pack, "facts", []) or []
    chunks = getattr(pack, "chunks", []) or []
    if (
        not graph
        and (facts or chunks)
        and not getattr(coverage, "needs_semantic", False)
        and not getattr(coverage, "needs_clusters", False)
    ):
        # PERSONAL intent keeps user-based expansion (LIKES/PREFERS lane).
        if intent == "personal" and getattr(pack, "owner_type", None) == "user":
            next_scope = cfg.get("next_predicate_scope")
            predicate_scope = next_scope(pack, graph_predicate_limit) if callable(next_scope) else []
            if predicate_scope:
                logger.info(
                    "RLM_DECISION trace_id=%s intent=%s domains=%s graph_seed=user_id predicate_scope=%s",
                    trace_id,
                    (intent or "").upper(),
                    active_domains,
                    predicate_scope[: max(1, int(graph_predicate_limit))],
                )
                actions.append(
                    RetrievalAction(
                        action="expand_graph",
                        subject=getattr(pack, "user_id", None),
                        predicate=predicate_scope[0],
                        domain_scope=["user_profile"],
                        hops=1,
                        direction="outbound",
                        k=min(max_items_per_type, 20),
                        owner_type=getattr(pack, "owner_type", None),
                    )
                )

        # TOPICAL / MIXED: seed graph expansion from evidence-derived entities, not user_id.
        else:
            kb_facts = filter_facts_by_domains(list(facts or []), allowed_domains={"kb_doc"})
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
                facts=list(facts or []),
                chunks=list(chunks or []),
                limit=5,
            )
            excluded_reasons: List[str] = []
            if intent in {"topical", "mixed"} and "user_profile" not in set(active_domains or []):
                try:
                    observed = _observed_predicates(list(facts or []))
                    if any(p in PREFERENCE_PREDICATES for p in observed):
                        excluded_reasons.append("excluded_user_profile_predicates_due_to_intent")
                except Exception:
                    pass
            combined_reasons: Optional[List[str]] = None
            try:
                rs: List[str] = []
                if isinstance(reasons_g, list):
                    rs.extend([r for r in reasons_g if isinstance(r, str) and r])
                rs.extend([r for r in excluded_reasons if isinstance(r, str) and r])
                combined_reasons = rs or None
            except Exception:
                combined_reasons = excluded_reasons or None
            logger.info(
                "RLM_DECISION trace_id=%s intent=%s domains=%s graph_seed=entities eligible_predicates=%s selected_predicate=%s score=%d selected_entities=%s fallback=%s reasons=%s",
                trace_id,
                (intent or "").upper(),
                active_domains,
                eligible_g,
                predicate,
                int(score or 0),
                (entities or [])[:5],
                fallback_reason,
                combined_reasons,
            )

            # Keep bounded: expand around at most 2 topical entities per step.
            for ent in (entities or [])[:2]:
                actions.append(
                    RetrievalAction(
                        action="expand_graph",
                        subject=ent,
                        predicate=predicate,
                        domain_scope=["kb_doc"],
                        hops=1,
                        direction="both",
                        k=min(max_items_per_type, 20),
                        owner_type=getattr(pack, "owner_type", None),
                    )
                )

    return ControllerDecision(actions=actions) if actions else None


def _extract_query_terms(query_text: str) -> List[str]:
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
    out: List[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _fact_blob(fact: Any) -> str:
    parts: List[str] = []
    try:
        subj = fact.get("subject") if isinstance(fact, dict) else getattr(fact, "subject", None)
        if subj:
            parts.append(str(subj))
    except Exception:
        pass
    try:
        obj = fact.get("object") if isinstance(fact, dict) else getattr(fact, "object", None)
        if obj:
            parts.append(str(obj))
    except Exception:
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
    except Exception:
        pass
    return " ".join(parts).lower()


def _predicate_score(predicate: str, facts: List[Any], terms: List[str]) -> int:
    if not predicate or not terms or not facts:
        return 0
    blob = " ".join(_fact_blob(f) for f in facts if f)[:20000]
    return sum(1 for t in terms if t and t in blob)


def _select_predicate_for_expansion(
    *,
    pack: Any,
    facts: List[Any],
    graph_predicate_limit: int,
    predicate_allowlist: Any,
) -> Tuple[Optional[str], int]:
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

    pred_to_facts: Dict[str, List[Any]] = {}
    for f in candidate_facts:
        try:
            pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
            if not pred:
                continue
            key = str(pred).strip().upper()
            if not key:
                continue
            pred_to_facts.setdefault(key, []).append(f)
        except Exception:
            continue

    candidates = list(pred_to_facts.keys())
    if not candidates:
        return None, 0

    # Optional domain-aware allowlist: intersect candidates with configured allowlist.
    observed = list(candidates)
    allowed: Set[str] = set()
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

    scored: List[Tuple[str, int]] = []
    for p in candidates:
        score = _predicate_score(p, pred_to_facts.get(p, []), terms)
        scored.append((p, int(score)))

    scored.sort(key=lambda x: (-x[1], x[0]))
    top_k = max(1, int(graph_predicate_limit))
    best_pred, best_score = scored[0]
    # (We compute top_k for future multi-predicate expansions; currently we return the best.)
    _ = scored[:top_k]
    return best_pred, best_score


def _observed_predicates(facts: List[Any]) -> Set[str]:
    out: Set[str] = set()
    for f in facts or []:
        try:
            pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
            if not pred:
                continue
            s = str(pred).strip().upper()
            if s:
                out.add(s)
        except Exception:
            continue
    return out


def _debug_score_predicates(
    *,
    pack: Any,
    facts: List[Any],
    graph_predicate_limit: int,
    predicate_allowlist: Any,
    top_n: int = 8,
) -> Tuple[List[str], Optional[str], int, Optional[List[str]]]:
    """
    Debug-only predicate scoring for observability logs.

    Returns:
    - eligible_predicates (top N, ordered)
    - best_predicate
    - best_score
    - reasons (optional)
    """
    reasons: List[str] = []
    active_domains = set(getattr(pack, "active_domains", []) or [])
    if active_domains and "user_profile" not in active_domains:
        reasons.append("excluded_user_profile_predicates_due_to_domains")

    candidate_facts = (
        filter_facts_by_domains(list(facts or []), active_domains)
        if active_domains
        else list(facts or [])
    )

    pred_to_facts: Dict[str, List[Any]] = {}
    for f in candidate_facts:
        try:
            pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
            if not pred:
                continue
            key = str(pred).strip().upper()
            if not key:
                continue
            pred_to_facts.setdefault(key, []).append(f)
        except Exception:
            continue

    candidates = list(pred_to_facts.keys())
    if not candidates:
        return [], None, 0, reasons or None

    observed = list(candidates)
    allowed: Set[str] = set()
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

    scored: List[Tuple[str, int]] = []
    for p in candidates:
        score = _predicate_score(p, pred_to_facts.get(p, []), terms)
        scored.append((p, int(score)))
    scored.sort(key=lambda x: (-x[1], x[0]))

    eligible_predicates = [p for (p, _s) in scored[: max(1, int(top_n))]]
    best_pred, best_score = scored[0]
    # Keep deterministic ordering, but limit the returned scored set.
    _ = scored[: max(1, int(graph_predicate_limit))]
    return eligible_predicates, best_pred, int(best_score), reasons or None


def _last_action_result(steps: List[Any]) -> Optional[Dict[str, Any]]:
    for s in reversed(steps or []):
        if isinstance(s, dict) and s.get("event") == "action_result":
            return s
    return None


def _did_action(steps: List[Any], action_name: str) -> bool:
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
    facts: List[Any],
    predicate_weights: Optional[Dict[str, float]],
    graph_predicate_limit: int,
) -> List[str]:
    """
    Determine a bounded predicate exploration scope based on:
    - configured predicate weights (highest first)
    - observed predicate frequency in current facts
    """
    ordered: List[str] = []

    if predicate_weights:
        for p, _ in sorted(predicate_weights.items(), key=lambda kv: (-float(kv[1]), str(kv[0]))):
            ordered.append(str(p).upper())

    counts: Dict[str, int] = {}
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


def _top_predicates_from_facts(facts: List[Any], limit: int) -> List[str]:
    """
    Deterministic predicate scope derived from the observed facts (no config weights).

    Used to ensure topical graph expansion uses kb_doc predicates even when MIXED intent
    includes user_profile facts.
    """
    counts: Dict[str, int] = {}
    for f in facts or []:
        try:
            pred = f.get("predicate") if isinstance(f, dict) else getattr(f, "predicate", None)
            if not pred:
                continue
            key = str(pred).strip().upper()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
        except Exception:
            continue
    ordered = [p for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if not ordered:
        ordered = ["RELATED_TO"]
    return ordered[: max(1, int(limit))]


#
# Action execution is centralized in UMAMemoryEnvironment.execute_action.
