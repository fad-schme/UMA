from __future__ import annotations

import logging
from typing import Any, Optional

from uma.retrieve.rlm.context_pack import ContextPack
from uma.retrieve.rlm.request import RetrievalRequest

logger = logging.getLogger(__name__)


async def expand_from_graph(
    *,
    env: Any,
    request: RetrievalRequest,
    pack: ContextPack,
    max_items_per_type: int,
    ranker: Any = None,
) -> list[Any]:
    """Route graph-found nodes to ordinary facts and episodes (bounded).

    `pack.graph` holds raw Cypher node dicts from graph traversal — a
    supporting/routing lane, not evidence in its own right (ARCHITECTURE.md:
    "Relationship routing and entity expansion"). Each owned node kind
    enriches retrieval through its own authoritative, owner-scoped,
    quarantine-filtered fetch:

      - `Fact` node    -> that fact                    -> `pack.facts`
      - `Entity` node  -> the owner's facts naming it  -> `pack.facts`
      - `Episode` node -> that episode                 -> `pack.episodes`

    Nodes are grouped by their *own* (owner_type, owner_id) properties and
    fetched per scope; a node without explicit owner scope routes nowhere.
    If `ranker` is given, routed facts and episodes go through the same
    trust-weighted blend and `min_trust_score` filter as every other fetch
    path — the graph layer applies no trust scoring itself (ticket 14).

    Returns the routed facts and episodes that survived ranking (may be empty).
    """
    if not pack.graph:
        return []
    max_items = max(0, int(max_items_per_type))
    if not max_items:
        return []

    Scope = tuple[str, str]
    by_kind: dict[str, dict[Scope, list[str]]] = {"Fact": {}, "Entity": {}, "Episode": {}}
    counts = {kind: 0 for kind in by_kind}

    for item in pack.graph:
        if not isinstance(item, dict):
            continue
        kind = next((label for label in item.get("labels") or [] if label in by_kind), None)
        props = item.get("properties") or {}
        node_id = str(props.get("id") or "").strip()
        if kind is None or not node_id or counts[kind] >= max_items:
            continue
        owner_type, owner_id = props.get("owner_type"), props.get("owner_id")
        if not owner_type or not owner_id:
            logger.debug("expand_from_graph: skipped %s node without explicit owner scope", kind)
            continue
        ids = by_kind[kind].setdefault((str(owner_type), str(owner_id)), [])
        if node_id not in ids:
            ids.append(node_id)
            counts[kind] += 1

    # Entities route to the facts that name them; merge those ids into the
    # Fact routing so each fact is fetched (and scope-checked) exactly once.
    if by_kind["Entity"] and hasattr(env, "graph_fact_ids_for_entities"):
        for (s_owner_type, s_owner_id), entity_ids in by_kind["Entity"].items():
            try:
                named = await env.graph_fact_ids_for_entities(
                    request,
                    entity_ids=entity_ids,
                    owner_type=s_owner_type,
                    owner_id=s_owner_id,
                    limit=max_items,
                )
            except Exception:
                logger.exception(
                    "expand_from_graph: entity->fact resolution failed owner=%s:%s entities=%d",
                    s_owner_type,
                    s_owner_id,
                    len(entity_ids),
                )
                named = []
            fact_ids = by_kind["Fact"].setdefault((s_owner_type, s_owner_id), [])
            for fact_id in named:
                if fact_id not in fact_ids and len(fact_ids) < max_items:
                    fact_ids.append(fact_id)

    async def _fetch_per_scope(fetch: Any, grouped: dict[Scope, list[str]], label: str) -> list[Any]:
        out: list[Any] = []
        for (s_owner_type, s_owner_id), ids in grouped.items():
            if not ids:
                continue
            try:
                got = await fetch(request=request, ids=ids, owner_type=s_owner_type, owner_id=s_owner_id)
            except Exception:
                logger.exception(
                    "expand_from_graph: %s fetch failed owner=%s:%s ids=%d",
                    label,
                    s_owner_type,
                    s_owner_id,
                    len(ids),
                )
                got = []
            out.extend(list(got or []))
        return out

    facts: list[Any] = []
    if by_kind["Fact"] and hasattr(env, "fetch_facts_by_ids"):
        facts = await _fetch_per_scope(env.fetch_facts_by_ids, by_kind["Fact"], "fact")
    episodes: list[Any] = []
    if by_kind["Episode"] and hasattr(env, "fetch_episodes_by_ids"):
        episodes = await _fetch_per_scope(env.fetch_episodes_by_ids, by_kind["Episode"], "episode")

    logger.debug(
        "expand_from_graph: fact_nodes=%d entity_nodes=%d episode_nodes=%d facts=%d episodes=%d",
        counts["Fact"],
        counts["Entity"],
        counts["Episode"],
        len(facts),
        len(episodes),
    )
    if not facts and not episodes:
        return []

    for routed in (*facts, *episodes):
        if isinstance(routed, dict):
            continue
        meta = getattr(routed, "meta", None)
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("retrieval_route", "graph_expand")
        meta.setdefault("retrieval_stage", "graph_node_resolve")
        routed.meta = meta

    query_text = getattr(pack, "query_text", "") or ""
    debug = getattr(pack, "debug", False)
    if ranker is not None:
        before_facts, before_episodes = len(facts), len(episodes)
        if facts:
            facts = ranker.rank_facts(facts, query_text=query_text, debug=debug)
        if episodes:
            episodes = ranker.rank_episodes(episodes, query_text=query_text, debug=debug)
        dropped = (before_facts - len(facts)) + (before_episodes - len(episodes))
        if dropped:
            logger.info(
                "expand_from_graph: min_trust_score dropped %d of %d graph-routed item(s)",
                dropped,
                before_facts + before_episodes,
            )

    from uma.common.dedupe import dedupe_by_id

    if facts:
        pack.facts = dedupe_by_id(list(getattr(pack, "facts", []) or []) + list(facts))[:max_items]
    if episodes:
        pack.episodes = dedupe_by_id(list(getattr(pack, "episodes", []) or []) + list(episodes))[:max_items]
    return list(facts) + list(episodes)


async def expand_evidence_chunks_from_facts(
    *,
    env: Any,
    request: RetrievalRequest,
    pack: ContextPack,
    max_items_per_type: int,
    ranker: Any = None,
) -> list[Any]:
    """Evidence expansion: fetch chunks referenced by fact.source_ids (bounded).

    IMPORTANT (Ownership / Lane Strategy)
    ------------------------------------
    In KB lane, `pack.facts` can contain facts from multiple scopes (agent-owned KB facts
    and user-owned personal KB facts). Evidence expansion MUST therefore fetch chunks
    using the *fact's* owner scope, not a single global (owner_type, owner_id) pair.

    This function:
      1) Collects cited chunk ids from `fact.source_ids`.
      2) Groups them by (fact.owner_type, fact.owner_id) if available.
      3) Fetches per-scope via `env.fetch_chunks(... owner_type, owner_id ...)`.
      4) If `ranker` is given, runs the fetched chunks through
         `ranker.rank_chunks` — the same trust-weighted blend and
         `min_trust_score` filter every search-found chunk goes through. A
         chunk's own `trust_score` is independent of the citing fact's — the
         fact having passed `min_trust_score` does not vouch for the chunk's
         own score, so this is checked separately rather than inherited.
      5) Merges deterministically into `pack.chunks` and tags provenance in
         meta.

    Returns the fetched evidence chunks that survived ranking/filtering (may
    be empty).
    """

    def _get_fact_owner(f: Any) -> tuple[Optional[str], Optional[str]]:
        if isinstance(f, dict):
            return (f.get("owner_type"), f.get("owner_id"))
        return (getattr(f, "owner_type", None), getattr(f, "owner_id", None))

    def _get_source_ids(f: Any) -> list[str]:
        src = f.get("source_ids") if isinstance(f, dict) else getattr(f, "source_ids", None)
        if isinstance(src, list):
            return [str(x) for x in src if x]
        return []

    try:
        max_ev = int(getattr(getattr(env, "_memory", None), "retrieval_cfg", None).max_evidence_chunks)
    except Exception:
        max_ev = 6
    max_ev = max(0, max_ev)
    if not max_ev:
        return []
    if not hasattr(env, "fetch_chunks"):
        return []
    if not pack.facts:
        return []

    # Build a bounded, deterministic list of cited (scope, chunk_id) pairs.
    cited_pairs: list[tuple[tuple[str, str], str]] = []
    seen_global: set[str] = set()

    for f in pack.facts:
        f_owner_type, f_owner_id = _get_fact_owner(f)
        if not f_owner_type or not f_owner_id:
            logger.debug("expand_evidence_chunks_from_facts: skipped fact without explicit owner scope")
            continue
        scope = (str(f_owner_type), str(f_owner_id))
        for sid in _get_source_ids(f):
            # Global dedupe by chunk id to keep bounded.
            if sid in seen_global:
                continue
            seen_global.add(sid)
            cited_pairs.append((scope, sid))
            if len(cited_pairs) >= max_ev:
                break
        if len(cited_pairs) >= max_ev:
            break

    if not cited_pairs:
        logger.debug("expand_evidence_chunks_from_facts: skipped (no source_ids after prune)")
        return []

    # Group by scope while preserving first-seen order of scopes.
    scope_order: list[tuple[str, str]] = []
    by_scope: dict[tuple[str, str], list[str]] = {}
    for scope, sid in cited_pairs:
        if scope not in by_scope:
            by_scope[scope] = []
            scope_order.append(scope)
        by_scope[scope].append(sid)

    logger.debug(
        "expand_evidence_chunks_from_facts: ids=%d scopes=%d",
        len(cited_pairs),
        len(scope_order),
    )

    chunks_ev_all: list[Any] = []

    # Fetch per scope; this prevents KB agent facts from being expanded using user scope (and vice versa).
    for (s_owner_type, s_owner_id) in scope_order:
        ids = by_scope.get((s_owner_type, s_owner_id)) or []
        if not ids:
            continue
        logger.debug(
            "expand_evidence_chunks_from_facts: fetch scope owner=%s:%s ids=%d",
            s_owner_type,
            s_owner_id,
            len(ids),
        )
        try:
            got = await env.fetch_chunks(
                request=request,
                ids=ids,
                owner_type=s_owner_type,
                owner_id=s_owner_id,
            )
        except Exception:
            logger.exception(
                "expand_evidence_chunks_from_facts: fetch_chunks failed owner=%s:%s",
                s_owner_type,
                s_owner_id,
            )
            got = []
        if got:
            chunks_ev_all.extend(list(got))

    # Attach evidence provenance.
    try:
        for ch in chunks_ev_all or []:
            if isinstance(ch, dict):
                continue
            meta = getattr(ch, "meta", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            meta.setdefault("retrieval_route", "evidence")
            meta.setdefault("retrieval_stage", "evidence_expand")
            ch.meta = meta
    except Exception:
        logger.exception("expand_evidence_chunks_from_facts: failed to attach evidence metadata")
        raise

    if ranker is not None and chunks_ev_all:
        before = len(chunks_ev_all)
        chunks_ev_all = ranker.rank_chunks(
            chunks_ev_all,
            query_text=getattr(pack, "query_text", "") or "",
            debug=getattr(pack, "debug", False),
        )
        dropped = before - len(chunks_ev_all)
        if dropped:
            logger.info(
                "expand_evidence_chunks_from_facts: min_trust_score dropped %d of %d evidence chunk(s)",
                dropped,
                before,
            )

    from uma.common.dedupe import dedupe_by_id

    merged = dedupe_by_id(list(getattr(pack, "chunks", []) or []) + list(chunks_ev_all or []))
    pack.chunks = merged[: max(0, int(max_items_per_type))]
    return list(chunks_ev_all or [])
