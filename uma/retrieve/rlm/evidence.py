from __future__ import annotations

import logging
from typing import Any, Optional

from uma.retrieve.rlm.context_pack import ContextPack
from uma.retrieve.rlm.request import RetrievalRequest

logger = logging.getLogger(__name__)


async def expand_facts_from_graph(
    *,
    env: Any,
    request: RetrievalRequest,
    pack: ContextPack,
    max_items_per_type: int,
    ranker: Any = None,
) -> list[Any]:
    """Route graph-found Fact nodes to ordinary facts (bounded).

    `pack.graph` holds raw Cypher node dicts from graph traversal — a
    supporting/routing lane, not evidence in its own right (ARCHITECTURE.md:
    "Relationship routing and entity expansion"). A `Fact`-labeled node names
    a real fact by id; this resolves those ids through the authoritative,
    owner-scoped, quarantine-filtered fetch path and merges the result into
    `pack.facts`, so graph-found knowledge reaches both retrieval products as
    ordinary facts rather than sitting unread in `pack.graph`.

    Entity-labeled nodes carry no fact id and are skipped — they have
    nothing to resolve to.

    This function:
      1) Collects Fact-node ids from `pack.graph`, grouped by the node's own
         (owner_type, owner_id) properties.
      2) Fetches per-scope via `env.fetch_facts_by_ids(... owner_type,
         owner_id ...)` — the same store path `search_semantic` uses, which
         already enforces `quarantined_at IS NULL` and ownership scoping.
      3) If `ranker` is given, runs the fetched facts through
         `ranker.rank_facts` — the same trust-weighted blend and
         `min_trust_score` filter every other fact-fetch path applies. The
         graph layer itself applies no trust scoring (ticket 14), so this is
         the first trust check these facts see.
      4) Merges deterministically into `pack.facts` and tags provenance in
         meta.

    Returns the fetched facts that survived ranking/filtering (may be empty).
    """

    def _fact_node_id(item: Any) -> Optional[str]:
        if not isinstance(item, dict):
            return None
        labels = item.get("labels") or []
        if "Fact" not in labels:
            return None
        props = item.get("properties") or {}
        node_id = props.get("id")
        return str(node_id) if node_id else None

    def _fact_node_owner(item: Any) -> tuple[Optional[str], Optional[str]]:
        props = item.get("properties") or {} if isinstance(item, dict) else {}
        return (props.get("owner_type"), props.get("owner_id"))

    if not hasattr(env, "fetch_facts_by_ids"):
        return []
    if not pack.graph:
        return []

    max_items = max(0, int(max_items_per_type))
    if not max_items:
        return []

    # Build a bounded, deterministic list of (scope, fact_id) pairs.
    cited_pairs: list[tuple[tuple[str, str], str]] = []
    seen_global: set[str] = set()

    for item in pack.graph:
        fact_id = _fact_node_id(item)
        if not fact_id or fact_id in seen_global:
            continue
        owner_type, owner_id = _fact_node_owner(item)
        if not owner_type or not owner_id:
            logger.debug("expand_facts_from_graph: skipped Fact node without explicit owner scope")
            continue
        seen_global.add(fact_id)
        cited_pairs.append(((str(owner_type), str(owner_id)), fact_id))
        if len(cited_pairs) >= max_items:
            break

    if not cited_pairs:
        return []

    # Group by scope while preserving first-seen order of scopes.
    scope_order: list[tuple[str, str]] = []
    by_scope: dict[tuple[str, str], list[str]] = {}
    for scope, fact_id in cited_pairs:
        if scope not in by_scope:
            by_scope[scope] = []
            scope_order.append(scope)
        by_scope[scope].append(fact_id)

    logger.debug(
        "expand_facts_from_graph: ids=%d scopes=%d",
        len(cited_pairs),
        len(scope_order),
    )

    facts_from_graph: list[Any] = []

    for (s_owner_type, s_owner_id) in scope_order:
        ids = by_scope.get((s_owner_type, s_owner_id)) or []
        if not ids:
            continue
        try:
            got = await env.fetch_facts_by_ids(
                request=request,
                ids=ids,
                owner_type=s_owner_type,
                owner_id=s_owner_id,
            )
        except Exception:
            logger.exception(
                "expand_facts_from_graph: fetch_facts_by_ids failed owner=%s:%s",
                s_owner_type,
                s_owner_id,
            )
            got = []
        if got:
            facts_from_graph.extend(list(got))

    if not facts_from_graph:
        return []

    # Attach provenance.
    try:
        for f in facts_from_graph:
            if isinstance(f, dict):
                continue
            meta = getattr(f, "meta", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            meta.setdefault("retrieval_route", "graph_expand")
            meta.setdefault("retrieval_stage", "graph_fact_resolve")
            f.meta = meta
    except Exception:
        logger.exception("expand_facts_from_graph: failed to attach provenance metadata")
        raise

    if ranker is not None:
        before = len(facts_from_graph)
        facts_from_graph = ranker.rank_facts(
            facts_from_graph,
            query_text=getattr(pack, "query_text", "") or "",
            debug=getattr(pack, "debug", False),
        )
        dropped = before - len(facts_from_graph)
        if dropped:
            logger.info(
                "expand_facts_from_graph: min_trust_score dropped %d of %d graph-resolved fact(s)",
                dropped,
                before,
            )
        if not facts_from_graph:
            return []

    from uma.common.dedupe import dedupe_by_id

    merged = dedupe_by_id(list(getattr(pack, "facts", []) or []) + list(facts_from_graph))
    pack.facts = merged[:max_items]
    return facts_from_graph


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
