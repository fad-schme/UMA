"""Gap analysis over a compiled memory result.

Surfaces facts whose *support* is weak, using signals the retrieval product
already carries. This is reporting, not filtering: a flagged fact is still
returned, and nothing here changes ranking, trust, or provenance validity.

Two signals, both computed from the supporting chunks that came back with the
result:

``stale_support``
    The newest chunk supporting the fact is older than the age threshold. The
    fact may still be true, but nothing recent corroborates it.

``weak_support``
    The fact rests on a single supporting chunk and that chunk's trust score is
    below the threshold. One weak source is the shape most likely to be wrong.

A fact with no resolvable supporting chunk is deliberately **not** reported
here — unsupported compiled claims are already handled by provenance
invalidation (``provenance_valid`` / ``invalid_reasons``), and duplicating that
signal would give an operator two places to look for one problem.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from uma.common.accessors import get_attr_or_key
from uma.common.provenance import collect_source_chunk_ids

logger = logging.getLogger(__name__)

# A fact whose freshest corroboration is older than this is worth flagging.
# Deliberately generous: durable facts are supposed to outlive their sources,
# so this catches "nothing has restated this in half a year", not ordinary age.
DEFAULT_MAX_SUPPORT_AGE_DAYS = 180

# Chunks below the retrieval trust floor are already filtered out, so this sits
# above it — the band that survives retrieval but is still the system's weakest
# evidence.
DEFAULT_MIN_SUPPORT_TRUST = 0.6

GAP_STALE_SUPPORT = "stale_support"
GAP_WEAK_SUPPORT = "weak_support"


def assess_gaps(
    *,
    facts: Iterable[Any],
    chunks: Iterable[Any],
    max_support_age_days: int = DEFAULT_MAX_SUPPORT_AGE_DAYS,
    min_support_trust: float = DEFAULT_MIN_SUPPORT_TRUST,
    now: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Return one entry per (fact, weak-support reason).

    A fact can appear twice when it is both stale and weakly supported; the
    reasons are independent and an operator acts on them differently.

    Pure and total: never raises, never performs I/O. A fact whose support
    cannot be interpreted is skipped rather than reported, so a malformed
    record cannot manufacture a false gap.
    """
    chunk_index = _index_chunks(chunks)
    if not chunk_index:
        return []

    reference_time = now or datetime.now(timezone.utc)
    gaps: list[dict[str, Any]] = []

    for fact in facts or []:
        try:
            supporting = [
                chunk_index[chunk_id]
                for chunk_id in collect_source_chunk_ids(fact)
                if chunk_id in chunk_index
            ]
        except Exception:
            logger.debug("assess_gaps: unreadable fact provenance; skipping", exc_info=True)
            continue

        if not supporting:
            # Covered by provenance invalidation — see module docstring.
            continue

        ages = [
            age
            for age in (_age_days(chunk, reference_time) for chunk in supporting)
            if age is not None
        ]
        if ages:
            newest_age = min(ages)
            if newest_age > max_support_age_days:
                gaps.append(
                    _entry(
                        fact,
                        reason=GAP_STALE_SUPPORT,
                        support_count=len(supporting),
                        newest_support_age_days=newest_age,
                    )
                )

        if len(supporting) == 1:
            trust = _trust(supporting[0])
            if trust is not None and trust < min_support_trust:
                gaps.append(
                    _entry(
                        fact,
                        reason=GAP_WEAK_SUPPORT,
                        support_count=1,
                        support_trust=round(trust, 4),
                    )
                )

    if gaps:
        logger.debug("assess_gaps: %d gap(s) across %d chunk(s)", len(gaps), len(chunk_index))
    return gaps


def _index_chunks(chunks: Iterable[Any]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for chunk in chunks or []:
        chunk_id = get_attr_or_key(chunk, "id", None)
        if chunk_id:
            index[str(chunk_id)] = chunk
    return index


def _entry(fact: Any, *, reason: str, **detail: Any) -> dict[str, Any]:
    subject = str(get_attr_or_key(fact, "subject", "") or "").strip()
    predicate = str(get_attr_or_key(fact, "predicate", "") or "").strip()
    object_text = str(get_attr_or_key(fact, "object", "") or "").strip()
    fact_id = get_attr_or_key(fact, "id", None)
    return {
        "fact_id": str(fact_id) if fact_id else None,
        "text": " ".join(part for part in (subject, predicate, object_text) if part),
        "reason": reason,
        **detail,
    }


def _age_days(chunk: Any, reference_time: datetime) -> Optional[int]:
    created_at = get_attr_or_key(chunk, "created_at", None)
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            return None
    if not isinstance(created_at, datetime):
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta_days = (reference_time - created_at).total_seconds() / 86400.0
    # Clamp: a clock-skewed future timestamp is not a staleness signal.
    return max(0, int(delta_days))


def _trust(chunk: Any) -> Optional[float]:
    raw = get_attr_or_key(chunk, "trust_score", None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def gap_thresholds(retrieval_cfg: Any) -> tuple[int, float]:
    """Read gap thresholds off a retrieval config, falling back to defaults."""
    try:
        max_age = int(
            getattr(retrieval_cfg, "gap_max_support_age_days", DEFAULT_MAX_SUPPORT_AGE_DAYS)
        )
        min_trust = float(
            getattr(retrieval_cfg, "gap_min_support_trust", DEFAULT_MIN_SUPPORT_TRUST)
        )
    except (TypeError, ValueError):
        return DEFAULT_MAX_SUPPORT_AGE_DAYS, DEFAULT_MIN_SUPPORT_TRUST
    return max(0, max_age), max(0.0, min(1.0, min_trust))


__all__ = [
    "DEFAULT_MAX_SUPPORT_AGE_DAYS",
    "DEFAULT_MIN_SUPPORT_TRUST",
    "GAP_STALE_SUPPORT",
    "GAP_WEAK_SUPPORT",
    "assess_gaps",
    "gap_thresholds",
]
