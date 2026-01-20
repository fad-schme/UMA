"""
Utility helpers for UMA-3 consolidation subsystem.

Here you can add:
- date range utilities
- summarization normalization
- canonicalization of plan/fact output

Coding agent instructions:
--------------------------
- Keep this file for small pure utilities only.
"""

def deduplicate_strings(strings):
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for s in strings:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out