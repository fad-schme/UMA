"""Public UMA SDK exports.

The public objects are loaded on first access.  Keeping package import light is
important for tooling such as ``uma version`` and for applications that only
need a narrow UMA submodule; optional provider and security dependencies should
not be imported until their functionality is requested.
"""

from __future__ import annotations

from importlib import import_module
import logging
from typing import TYPE_CHECKING, Any

logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from .api import UMAMemory
    from .adapters.scanner.injection_scan import InjectionDetectedError
    from .common.results import (
        CompiledMemory,
        Confidence,
        ContextBundle,
        DebugInfo,
        HealthStatus,
        MemoryResult,
        Provenance,
    )

__all__ = [
    "CompiledMemory",
    "Confidence",
    "ContextBundle",
    "DebugInfo",
    "HealthStatus",
    "InjectionDetectedError",
    "MemoryResult",
    "Provenance",
    "UMAMemory",
]

_PUBLIC_EXPORTS = {
    "UMAMemory": ("uma.api", "UMAMemory"),
    "InjectionDetectedError": (
        "uma.adapters.scanner.injection_scan",
        "InjectionDetectedError",
    ),
    "CompiledMemory": ("uma.common.results", "CompiledMemory"),
    "Confidence": ("uma.common.results", "Confidence"),
    "ContextBundle": ("uma.common.results", "ContextBundle"),
    "DebugInfo": ("uma.common.results", "DebugInfo"),
    "HealthStatus": ("uma.common.results", "HealthStatus"),
    "MemoryResult": ("uma.common.results", "MemoryResult"),
    "Provenance": ("uma.common.results", "Provenance"),
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
