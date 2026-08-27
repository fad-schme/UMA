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
- ``MULTILINE``, ``IGNORECASE``, ``DOTALL`` — flag constants owned by this
  module (not re-exported from either backend — ``google-re2`` has no such
  constants; it takes an ``Options`` object instead). ``compile()``
  translates these bits to whichever backend is active so callers see
  identical semantics either way.
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

# Flag bits are ours, not either backend's — ``google-re2`` has no MULTILINE
# / IGNORECASE / DOTALL constants at all, so there is nothing to re-export.
IGNORECASE = 1 << 0
MULTILINE = 1 << 1
DOTALL = 1 << 2

try:
    import re2 as _re2  # type: ignore[import-not-found]
    USING_RE2 = True
except ImportError:
    _re2 = None
    USING_RE2 = False
    logger.warning(
        "uma.common._regex_backend: google-re2 not installed; falling back "
        "to Python `re`. Functionally correct for the shipped pattern "
        "catalog, but future patterns are not protected against ReDoS. "
        "Install `pip install uma-mem[security]` for the linear-time RE2 "
        "backend in production."
    )

if USING_RE2:
    error = _re2.error

    def compile(pattern, flags=0):
        options = _re2.Options()
        options.case_sensitive = not bool(flags & IGNORECASE)
        # RE2's `one_line` defaults to False (^/$ match at line breaks),
        # the opposite of stdlib `re`'s MULTILINE-off default. Pin it
        # explicitly every call so both backends agree regardless of which
        # one is installed.
        options.one_line = not bool(flags & MULTILINE)
        options.dot_nl = bool(flags & DOTALL)
        return _re2.compile(pattern, options=options)
else:
    error = _stdlib_re.error

    def compile(pattern, flags=0):
        stdlib_flags = 0
        if flags & IGNORECASE:
            stdlib_flags |= _stdlib_re.IGNORECASE
        if flags & MULTILINE:
            stdlib_flags |= _stdlib_re.MULTILINE
        if flags & DOTALL:
            stdlib_flags |= _stdlib_re.DOTALL
        return _stdlib_re.compile(pattern, stdlib_flags)

__all__ = [
    "USING_RE2",
    "MULTILINE",
    "IGNORECASE",
    "DOTALL",
    "error",
    "compile",
]
