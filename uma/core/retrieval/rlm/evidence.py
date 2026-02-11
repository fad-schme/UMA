from __future__ import annotations

import logging
from typing import Any, List, Optional

from uma.core.retrieval.rlm.context_pack import ContextPack

logger = logging.getLogger(__name__)


async def expand_evidence_chunks_from_facts(
    *,
    env: Any,
    pack: ContextPack,
    owner_type: str,
    owner_id: Optional[str],
    max_items_per_type: int,
) -> List[Any]:
    """
    Evidence expansion: fetch chunks referenced by fact.source_ids (bounded).

    Called after pruning so only relevant facts can pull additional chunks into
    the candidate set.

    Returns the fetched evidence chunks (may be empty).
    """
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

    cited: List[str] = []
    for f in pack.facts:
        src = f.get("source_ids") if isinstance(f, dict) else getattr(f, "source_ids", None)
        if isinstance(src, list):
            for sid in src:
                if sid:
                    cited.append(str(sid))
    cited = list(dict.fromkeys(cited))[:max_ev]
    if not cited:
        logger.debug("expand_evidence_chunks_from_facts: skipped (no source_ids after prune)")
        return []

    logger.debug(
        "expand_evidence_chunks_from_facts: ids=%d owner=%s:%s",
        len(cited),
        owner_type,
        owner_id,
    )
    chunks_ev = await env.fetch_chunks(
        user_id=pack.user_id,
        ids=cited,
        owner_type=owner_type,
        owner_id=owner_id,
    )

    try:
        for ch in chunks_ev or []:
            if isinstance(ch, dict):
                continue
            meta = getattr(ch, "meta", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            meta.setdefault("retrieval_route", "evidence")
            meta.setdefault("retrieval_stage", "evidence_expand")
            ch.meta = meta
    except Exception:
        pass
    from uma.core.utils.dedupe import dedupe_by_id

    merged = dedupe_by_id(list(getattr(pack, "chunks", []) or []) + list(chunks_ev or []))
    pack.chunks = merged[: max(0, int(max_items_per_type))]
    return list(chunks_ev or [])
