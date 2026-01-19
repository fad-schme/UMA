"""
UMA-3 Feature Registry System (Production Version)
==================================================

This module defines:
    - UMA3Feature: Abstract base class for pluggable UMA-3 features
    - FeatureRegistry: Factory/registry for optional feature plug-ins

This remains part of the UMA-3 CORE because:
    • All future optional capabilities (procedural memory, consolidation,
      advanced graph tools, user profiles, etc.) must register here.
    • UMA3Memory.enable_feature() depends on this registry.

Coding Agent Instructions
-------------------------
- Each optional feature MUST subclass UMA3Feature.
- Features MUST implement .attach(memory_client) safely.
- Exceptions in attach() MUST BE logged AND must NOT break UMA-3.
- The registry MUST validate that feature constructors return UMA3Feature.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


# ======================================================================
#  UMA3Feature (ABC)
# ======================================================================

class UMA3Feature(ABC):
    """
    Abstract base class for UMA-3 optional plugin features.

    Requirements
    ------------
    - Must set class attribute `name` (string)
    - Must implement .attach(memory_client)
    - Must NOT raise exceptions during attach()

    attach(memory_client):
        Use this to extend UMA3Memory with additional helper functions,
        hook registrations, or specialized subsystems.
    """

    #: Name used to reference this feature inside UMA3Memory.features
    name: str = "unnamed_feature"

    @abstractmethod
    def attach(self, memory_client: Any) -> None:
        """
        Attach feature to UMA3Memory instance.

        Must:
          - Register itself under memory_client.features[self.name]
          - Initialize internal resources
          - Not raise exceptions (log instead)

        Parameters
        ----------
        memory_client : UMA3Memory
            The runtime memory container object.
        """
        raise NotImplementedError


# ======================================================================
#  FeatureRegistry
# ======================================================================

class FeatureRegistry:
    """
    UMA-3 Feature Registry

    Registers feature factories (constructors) and instantiates them
    on demand with dependency injection.

    Typical Usage:
    --------------
        reg = FeatureRegistry()
        reg.register("procedural", ProceduralFeature)
        feature = reg.create("procedural", llm=..., tools=...)
        memory.enable_feature(feature)
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., UMA3Feature]] = {}
        logger.info("FeatureRegistry initialized.")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, factory: Callable[..., UMA3Feature]) -> None:
        """
        Register a feature constructor/factory.

        Parameters
        ----------
        name : str
            Identifier for the feature (e.g., 'consolidation').
        factory : Callable[..., UMA3Feature]
            Constructor that returns a UMA3Feature instance.
        """
        if not callable(factory):
            raise TypeError(f"Factory for feature '{name}' must be callable.")

        self._factories[name] = factory
        logger.info("FeatureRegistry: registered feature '%s'.", name)

    # ------------------------------------------------------------------
    # Instance Creation
    # ------------------------------------------------------------------

    def create(self, name: str, **kwargs) -> UMA3Feature:
        """
        Construct an instance of a registered feature.

        Returns
        -------
        UMA3Feature

        Raises
        ------
        KeyError
            If feature is not registered.
        RuntimeError
            If constructor fails.
        TypeError
            If returned instance is not a UMA3Feature.
        """
        if name not in self._factories:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(
                f"Feature '{name}' not registered. Available: {available}"
            )

        ctor = self._factories[name]

        try:
            feature = ctor(**kwargs)
        except Exception:
            logger.exception("FeatureRegistry: failed to construct '%s'.", name)
            raise RuntimeError(
                f"Feature '{name}' failed to initialize. See logs for details."
            )

        if not isinstance(feature, UMA3Feature):
            raise TypeError(
                f"Factory for '{name}' did not return a UMA3Feature subclass."
            )

        logger.info("FeatureRegistry: created feature '%s'.", name)
        return feature