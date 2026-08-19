"""
_regex_backend.py — regex engine selector for security-critical scanning.

Prefers ``google-re2`` (linear-time execution, ReDoS-proof by construction)
when installed; falls back to Python's ``re`` module with a WARNING at first
import.

Install the RE2 backend for production::

    pip install uma-mem[security]

Only used by security-critical scan paths (``injection_scan``,
``rule_functions``). Other UMA code that uses regex is not affected — this
module is deliberately scoped to the write-time attack surface.

Contract
--------
- ``compile(pattern: str, flags: int = 0) -> Pattern`` — same call shape as
  ``re.compile``. The returned object exposes ``.search``, ``.match``,
  ``.finditer``, ``.findall``.
- ``MULTILINE``, ``IGNORECASE``, ``DOTALL`` — flag constants of the active
  backend (identical semantics for both engines on the patterns UMA uses).
- ``error`` — the compile-error exception class of the active backend.
- ``USING_RE2: bool`` — True iff the RE2 backend is active. Tests may
  assert on this to enforce a production posture.

Design notes
------------
The warning fires exactly once, at first import. Callers do not need to
check ``USING_RE2`` at runtime — the fallback is functionally correct for
every pattern currently in UMA's catalog (all 251 patterns across
``injection_patterns.yaml``, ``injection_patterns.l10n.yaml``, and
``sqli_patterns.yaml`` are RE2-compatible; verified). RE2 upgrades ReDoS
from "guarded by pattern review discipline" to "impossible by construction."

Note that ``google-re2`` wheels are not reliably available on Windows, so
the ``re`` fallback is the practical default there.
"""
from __future__ import annotations

import logging
import re as _stdlib_re

logger = logging.getLogger(__name__)

try:
    import re2 as _backend  # type: ignore[import-not-found]
    USING_RE2 = True
except ImportError:
    _backend = _stdlib_re
    USING_RE2 = False
    logger.warning(
        "uma.common._regex_backend: google-re2 not installed; falling back "
        "to Python `re`. Functionally correct for the shipped pattern "
        "catalog, but future patterns are not protected against ReDoS. "
        "Install `pip install uma-mem[security]` for the linear-time RE2 "
        "backend in production."
    )

# Re-export flag constants and the compile-error exception from whichever
# backend is active. Both engines expose these with identical semantics for
# the pattern subset UMA uses.
MULTILINE = _backend.MULTILINE
IGNORECASE = _backend.IGNORECASE
DOTALL = _backend.DOTALL
error = _backend.error
compile = _backend.compile

__all__ = [
    "USING_RE2",
    "MULTILINE",
    "IGNORECASE",
    "DOTALL",
    "error",
    "compile",
]
