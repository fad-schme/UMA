from __future__ import annotations

import re


def trim_to_sentence_boundary(text: str, *, max_chars: int) -> str:
    """
    Trim text to a maximum character budget, preferring to cut at a sentence boundary.

    This is a presentation-only helper used by snippet rendering/refinement.
    """
    if not text:
        return ""
    try:
        max_chars_i = int(max_chars)
    except Exception:
        max_chars_i = 0
    if max_chars_i <= 0:
        return ""
    if len(text) <= max_chars_i:
        return text.strip()

    cut = text[:max_chars_i]
    # Prefer the last real sentence boundary: .!? followed by whitespace (not a decimal or abbreviation mid-word).
    boundaries = [m.end() for m in re.finditer(r"[.!?](?=\s)", cut)]
    if cut and cut[-1] in ".!?":
        boundaries.append(len(cut))
    if boundaries:
        return cut[: boundaries[-1]].strip()
    # Fallback: last .!? of any kind (abbreviation, decimal, etc.)
    m = re.search(r"[.!?](?!.*[.!?])", cut)
    if m:
        return cut[: m.end()].strip()
    return cut.strip()

