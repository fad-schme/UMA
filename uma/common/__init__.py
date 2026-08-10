"""Shared UMA types and configuration exports.

Exports are loaded lazily so importing a focused common submodule does not
initialize the full storage, retrieval, and memory dependency graph.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import UMAConfig
    from .types import (
        Chunk,
        Episode,
        Fact,
        OwnershipRef,
        OwnerType,
        RuntimeContext,
        SCOPE_MODEL_VERSION,
        SessionScope,
        Skill,
    )

__all__ = [
    "Chunk",
    "Episode",
    "Fact",
    "OwnershipRef",
    "OwnerType",
    "RuntimeContext",
    "SCOPE_MODEL_VERSION",
    "SessionScope",
    "Skill",
    "UMAConfig",
]

_PUBLIC_EXPORTS = {
    "UMAConfig": ("uma.common.config", "UMAConfig"),
    "Chunk": ("uma.common.types", "Chunk"),
    "Episode": ("uma.common.types", "Episode"),
    "Fact": ("uma.common.types", "Fact"),
    "OwnershipRef": ("uma.common.types", "OwnershipRef"),
    "OwnerType": ("uma.common.types", "OwnerType"),
    "RuntimeContext": ("uma.common.types", "RuntimeContext"),
    "SCOPE_MODEL_VERSION": ("uma.common.types", "SCOPE_MODEL_VERSION"),
    "SessionScope": ("uma.common.types", "SessionScope"),
    "Skill": ("uma.common.types", "Skill"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
