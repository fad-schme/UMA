"""
Canonical working-memory package surface.

PR 8 removes the stale package-level user-keyed API. The package now re-exports
the canonical session-scoped implementation from `uma.memory.working_memory.core`.
"""

from .buffer import WorkingMemoryBuffer, WorkingMemoryMessage
from .core import (
    WorkingMemoryCore,
    legacy_session_scope_for_user,
    session_scope_from_runtime_context,
)

__all__ = [
    "WorkingMemoryBuffer",
    "WorkingMemoryCore",
    "WorkingMemoryMessage",
    "legacy_session_scope_for_user",
    "session_scope_from_runtime_context",
]
