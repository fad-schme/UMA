
from __future__ import annotations

import asyncio
import inspect
import logging
import json
from typing import Any, List, Optional

from uma.common.types import Fact

logger = logging.getLogger(__name__)


def _get_source_chunk_id(fact: Fact) -> Optional[str]:
    """Best-effort extraction of a single source chunk id from a Fact.

    Returns the first element of fact.source_ids if present, else None.
    """
    try:
        src_ids = getattr(fact, "source_ids", None)
        if isinstance(src_ids, list) and src_ids:
            first = src_ids[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    except Exception:
        logger.exception("_get_source_chunk_id: failed to read fact.source_ids")
        raise

    return None


def _validate_fact_for_graph(fact: Fact) -> None:
    """Validate minimal invariants required for graph provenance."""
    missing: List[str] = []

    if not getattr(fact, "id", None):
        missing.append("id")
    if not getattr(fact, "subject", None):
        missing.append("subject")
    if not getattr(fact, "predicate", None):
        missing.append("predicate")
    if getattr(fact, "object", None) is None:
        missing.append("object")

    # Ownership is mandatory for graph to remain partition-safe.
    if not getattr(fact, "owner_type", None):
        missing.append("owner_type")
    if not getattr(fact, "owner_id", None):
        missing.append("owner_id")

    if missing:
        logger.error("Fact missing required fields for graph update: %s", missing)
        raise ValueError(f"Fact missing required fields for graph update: {missing}")


async def _maybe_await(result: Any) -> Any:
    """Await coroutine results; otherwise return as-is."""
    if inspect.isawaitable(result):
        return await result
    return result


async def update_graph(
    facts: List[Fact],
    *,
    graph_core: Any,
    concurrency: int = 8,
) -> int:
    """Update the temporal graph with newly extracted semantic facts.

    This function closes the **fact → graph provenance gap** by ensuring that
    each upsert includes:
      - fact_id
      - owner_type / owner_id
      - source_chunk_id (best-effort)
      - timestamps (if graph core supports it)

    Graph is a *derived* store. Failures should not break ingestion, but must be
    observable.

    Parameters
    ----------
    facts:
        List of semantic Fact objects (authoritative in SQL).
    graph_core:
        Graph core instance. Preferred API:
            insert_fact_triplet(
                *,
                fact_id: str,
                subject: str,
                predicate: str,
                object: str,
                owner_type: str,
                owner_id: str,
                source_chunk_id: str | None = None,
                created_at: datetime | None = None,
                updated_at: datetime | None = None,
                meta: dict | None = None,
            )

        No fallback is supported.

    Returns
    -------
    int
        Number of fact edges attempted.
    """
    if not facts:
        return 0

    if graph_core is None:
        logger.warning("update_graph: graph_core missing; skipping")
        return 0

    attempted = 0
    failed = 0
    skipped = 0

    try:
        concurrency = int(concurrency)
    except Exception:
        concurrency = 8
    concurrency = max(1, min(concurrency, 32))
    sem = asyncio.Semaphore(concurrency)

    async def _upsert_one(fact: Fact) -> None:
        nonlocal attempted, failed, skipped
        async with sem:
            try:
                _validate_fact_for_graph(fact)
                source_chunk_id = _get_source_chunk_id(fact)

                meta_json = None
                try:
                    if isinstance(getattr(fact, "meta", None), dict) and fact.meta:
                        meta_json = json.dumps(fact.meta, default=str)
                except Exception:
                    meta_json = None

                domain = None
                try:
                    if isinstance(getattr(fact, "meta", None), dict):
                        d = fact.meta.get("domain")
                        if isinstance(d, str) and d.strip():
                            domain = d.strip().lower()
                except Exception:
                    domain = None

                res = graph_core.insert_fact_triplet(
                    fact_id=str(fact.id),
                    subject=str(fact.subject),
                    predicate=str(fact.predicate),
                    object=str(fact.object),
                    owner_type=str(fact.owner_type),
                    owner_id=str(fact.owner_id),
                    source_chunk_id=source_chunk_id,
                    created_at=getattr(fact, "created_at", None),
                    updated_at=getattr(fact, "updated_at", None),
                    meta_json=meta_json,
                    domain=domain,
                )
                await _maybe_await(res)
                attempted += 1
            except ValueError:
                skipped += 1
                logger.exception(
                    "update_graph: skipped invalid fact for graph (fact_id=%s)",
                    getattr(fact, "id", "<missing>"),
                )
            except Exception:
                failed += 1
                logger.exception(
                    "update_graph: failed to upsert fact into graph (fact_id=%s)",
                    getattr(fact, "id", "<missing>"),
                )

    await asyncio.gather(*[_upsert_one(f) for f in facts], return_exceptions=False)

    if failed or skipped:
        logger.warning(
            "update_graph: completed with issues attempted=%d failed=%d skipped=%d",
            attempted,
            failed,
            skipped,
        )
    else:
        logger.info("update_graph: upserted %d fact(s) into graph", attempted)

    return attempted
