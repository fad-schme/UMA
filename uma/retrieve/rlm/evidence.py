from __future__ import annotations

import logging
from typing import Any, Optional

from uma.retrieve.rlm.context_pack import ContextPack
from uma.retrieve.rlm.request import RetrievalRequest

logger = logging.getLogger(__name__)


async def expand_evidence_chunks_from_facts(
    *,
    env: Any,
    request: RetrievalRequest,
    pack: ContextPack,
    max_items_per_type: int,
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
      4) Merges deterministically into `pack.chunks` and tags provenance in meta.

    Returns the fetched evidence chunks (may be empty).
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

    from uma.common.dedupe import dedupe_by_id

    merged = dedupe_by_id(list(getattr(pack, "chunks", []) or []) + list(chunks_ev_all or []))
    pack.chunks = merged[: max(0, int(max_items_per_type))]
    return list(chunks_ev_all or [])
