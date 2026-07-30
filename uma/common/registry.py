"""
UMA Feature Attachment Interface (Production Version)
=======================================================

This module defines:
    - UMAFeature: Abstract base class for pluggable UMA features

This remains part of the UMA CORE because optional capabilities
must share a consistent attachment interface.

Coding Agent Instructions
-------------------------
- Each optional feature MUST subclass UMAFeature.
- Features MUST implement .attach(memory_client) safely.
- Exceptions in attach() MUST BE logged AND must NOT break UMA.
- Optional features attach directly to UMAMemory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from importlib import metadata
import inspect
import logging
from typing import Any, Iterable, Optional, Union


# ======================================================================
#  UMAFeature (ABC)
# ======================================================================

logger = logging.getLogger(__name__)


class UMAFeature(ABC):
    """
    Abstract base class for UMA optional plugin features.

    Requirements
    ------------
    - Must set class attribute `name` (string)
    - Must implement .attach(memory_client)
    - Must NOT raise exceptions during attach()

    attach(memory_client):
        Use this to extend UMAMemory with additional helper functions,
        hook registrations, or specialized subsystems.
    """

    #: Name used to reference this feature inside UMAMemory.features
    name: str = "unnamed_feature"

    @abstractmethod
    def attach(self, context: Any) -> Optional["FeatureHandle"]:
        """
        Attach feature to UMAMemory instance.

        Must:
          - Register itself under memory_client.features[self.name]
          - Initialize internal resources
          - Not raise exceptions (log instead)

        Parameters
        ----------
        context : FeatureContext
            The runtime feature context.
        """
        raise NotImplementedError

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> None:
        """Validate feature config (override in subclasses if needed)."""


@dataclass(frozen=True)
class FeatureHandle:
    name: str
    methods: tuple[str, ...] = ()
    version: Optional[str] = None


@dataclass(frozen=True)
class FeatureResult:
    ok: bool
    data: Any = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @classmethod
    def success(cls, data: Any = None, warnings: Optional[list[str]] = None) -> "FeatureResult":
        return cls(ok=True, data=data, warnings=tuple(warnings or ()))

    @classmethod
    def failure(cls, errors: Optional[list[str]] = None, data: Any = None) -> "FeatureResult":
        return cls(ok=False, data=data, errors=tuple(errors or ()))


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    provider: Union[str, type[UMAFeature]]
    default_config: dict[str, Any] = field(default_factory=dict)
    required: bool = False


@dataclass(frozen=True)
class FeaturePolicy:
    on_attach_error: str = "log_and_skip"  # "log_and_skip" or "raise"
    allow_method_override: bool = False


@dataclass(frozen=True)
class FeatureContext:
    memory: Any
    config: dict[str, Any]
    services: dict[str, Any]
    logger: logging.Logger


class FeatureRegistry:
    """Central registry for UMAFeature providers."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if not spec.name or not isinstance(spec.name, str):
            raise ValueError("FeatureSpec.name must be a non-empty string")
        if spec.name in self._specs:
            raise ValueError(f"Feature already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[FeatureSpec]:
        return self._specs.get(name)

    def specs(self) -> Iterable[FeatureSpec]:
        return self._specs.values()

    def register_entry_points(self, group: str = "uma.memory.features") -> None:
        try:
            eps = metadata.entry_points()
        except Exception:
            logger.exception("Failed to load entry points.")
            return

        matches = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
        for ep in matches:
            try:
                provider = ep.load()
                self.register(FeatureSpec(name=ep.name, provider=provider))
            except Exception:
                logger.exception("Failed to load feature entry point: %s", ep.name)


class FeatureLoader:
    """Config-driven feature loader with safe attachment."""

    def __init__(self, registry: FeatureRegistry, policy: FeaturePolicy) -> None:
        self.registry = registry
        self.policy = policy

    def load_from_config(
        self,
        memory_client: Any,
        feature_cfgs: list[dict[str, Any]],
        services: dict[str, Any],
    ) -> dict[str, UMAFeature]:
        loaded: dict[str, UMAFeature] = {}

        for cfg in feature_cfgs:
            if not isinstance(cfg, dict):
                logger.error("Feature config must be a mapping; got %r", cfg)
                continue

            name = cfg.get("name")
            if not name:
                logger.error("Feature config missing 'name': %r", cfg)
                continue

            if cfg.get("enabled") is False:
                logger.info("Feature %s disabled by config.", name)
                continue

            provider = cfg.get("provider")
            spec = self.registry.get(name)
            if provider is None and spec is None:
                logger.error("Feature %s has no provider and is not registered.", name)
                continue

            resolved_provider = provider or (spec.provider if spec else None)
            if resolved_provider is None:
                logger.error("Feature %s provider could not be resolved.", name)
                continue

            try:
                feature_cls = self._resolve_provider(resolved_provider)
            except Exception:
                logger.exception("Failed to resolve provider for %s.", name)
                if self.policy.on_attach_error == "raise":
                    raise
                continue

            merged_config = {}
            if spec and isinstance(spec.default_config, dict):
                merged_config.update(spec.default_config)
            if isinstance(cfg.get("config"), dict):
                merged_config.update(cfg["config"])

            try:
                if hasattr(feature_cls, "validate_config"):
                    feature_cls.validate_config(merged_config)
            except Exception:
                logger.exception("Config validation failed for feature %s.", name)
                if self.policy.on_attach_error == "raise":
                    raise
                continue

            try:
                feature = self._construct_feature(feature_cls, merged_config, services)
                handle = self._attach_feature(feature, memory_client, merged_config, services)
                loaded[name] = feature
                if handle and handle.methods:
                    logger.info("Feature %s attached with methods=%s", name, handle.methods)
                else:
                    logger.info("Feature %s attached.", name)
            except Exception:
                logger.exception("Failed to attach feature %s.", name)
                if self.policy.on_attach_error == "raise":
                    raise
                continue

        return loaded

    def _resolve_provider(self, provider: Union[str, type[UMAFeature]]) -> type[UMAFeature]:
        if inspect.isclass(provider):
            return provider
        if not isinstance(provider, str):
            raise TypeError(f"Invalid provider type: {type(provider)}")
        module_path, _, attr = provider.replace(":", ".").rpartition(".")
        if not module_path or not attr:
            raise ValueError(f"Provider must be a module path: {provider}")
        module = import_module(module_path)
        feature_cls = getattr(module, attr)
        if not inspect.isclass(feature_cls):
            raise TypeError(f"Provider is not a class: {provider}")
        return feature_cls

    def _construct_feature(
        self,
        feature_cls: type[UMAFeature],
        config: dict[str, Any],
        services: dict[str, Any],
    ) -> UMAFeature:
        kwargs = self._filter_kwargs(feature_cls, {**services, **config})
        return feature_cls(**kwargs)

    def _attach_feature(
        self,
        feature: UMAFeature,
        memory_client: Any,
        config: dict[str, Any],
        services: dict[str, Any],
    ) -> Optional[FeatureHandle]:
        context = FeatureContext(
            memory=memory_client,
            config=config,
            services=services,
            logger=logging.getLogger(feature.__class__.__name__),
        )

        attach_sig = inspect.signature(feature.attach)
        params = list(attach_sig.parameters.values())
        if len(params) == 2 and params[1].name in {"memory_client", "memory"}:
            return feature.attach(context.memory)  # type: ignore[misc]
        return feature.attach(context)  # type: ignore[misc]

    def _filter_kwargs(self, feature_cls: type[UMAFeature], payload: dict[str, Any]) -> dict[str, Any]:
        sig = inspect.signature(feature_cls.__init__)
        params = set(sig.parameters)
        params.discard("self")
        return {k: v for k, v in payload.items() if k in params}


def default_feature_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    registry.register(
        FeatureSpec(
            name="procedural",
            provider="uma.memory.procedural.feature:ProceduralFeature",
        )
    )
    registry.register(
        FeatureSpec(
            name="consolidation",
            provider="uma.memory.consolidation.feature:ConsolidationFeature",
        )
    )
    return registry
