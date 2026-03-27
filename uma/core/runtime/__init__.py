"""
Runtime boundary types for UMA.

PR 2 introduces a shared runtime abstraction plus an immutable bound handle.
Execution cutover remains out of scope for this package in the current PR.
"""

from .runtime import UMARuntime, UMARequestHandle, UMABoundMemory

__all__ = [
    "UMARuntime",
    "UMARequestHandle",
    "UMABoundMemory",
]
