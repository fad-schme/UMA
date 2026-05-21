"""
Source trust classifier — PR2.

Maps a SourceDescriptor to a trust score in [0.0, 1.0].
Policy values are defined inline; no config surface in this PR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SourceDescriptor:
    """Minimal provenance description needed for trust scoring."""
    kind: str  # "turn_user" | "turn_assistant" | "document" | "bootstrap_memory" | "bootstrap_diary" | "tool_output" | "promotion"
    session_id: Optional[str] = None
    import_mode: Optional[str] = None
    parent_trust_score: Optional[float] = None


def score_source(source: SourceDescriptor) -> float:
    """
    Return a trust score in [0.0, 1.0] for the described source.

    Policy v1
    ---------
    turn_user (authenticated session)        → 0.9
    turn_assistant (authenticated session)   → 0.7
    document (via ingest_document)           → 0.7
    bootstrap (import_mode == "manual")      → 0.8
    bootstrap (default)                      → 0.6
    tool_output                              → 0.5
    promotion                                → inherit parent, default 0.5
    anything else                            → 0.5

    "Authenticated session" = session_id is present and non-empty.
    UMA does not model auth strength beyond session presence in v1.
    """
    kind = source.kind
    authenticated = bool(source.session_id and source.session_id.strip())

    if kind == "turn_user":
        return 0.9 if authenticated else 0.5
    if kind == "turn_assistant":
        return 0.7 if authenticated else 0.5
    if kind == "document":
        return 0.7
    if kind in ("bootstrap_memory", "bootstrap_diary"):
        return 0.8 if source.import_mode == "manual" else 0.6
    if kind == "tool_output":
        return 0.5
    if kind == "promotion":
        return source.parent_trust_score if source.parent_trust_score is not None else 0.5
    return 0.5
