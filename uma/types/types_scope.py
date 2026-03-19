"""
Runtime scope and ownership reference types for UMA.

This module defines the canonical immutable vocabulary for:
- runtime context (tenant / workspace / agent / user / session / request)
- session-local identity
- persistent ownership references
- explicit write targets

These types are intentionally additive in PR 1:
- they do not change runtime execution behavior
- they do not encode authorization or lane policy
- they validate only structural invariants
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types_owner import OwnerType


SCOPE_MODEL_VERSION = "v2"


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_optional(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_non_empty(value, field_name)


def validate_tenant_id(tenant_id: str) -> str:
    return _require_non_empty(tenant_id, "tenant_id")


def validate_agent_id(agent_id: str) -> str:
    return _require_non_empty(agent_id, "agent_id")


def validate_request_id(request_id: str) -> str:
    return _require_non_empty(request_id, "request_id")


def validate_user_id(user_id: Optional[str]) -> Optional[str]:
    return _validate_optional(user_id, "user_id")


def validate_workspace_id(workspace_id: Optional[str]) -> Optional[str]:
    return _validate_optional(workspace_id, "workspace_id")


def validate_session_id(session_id: Optional[str]) -> Optional[str]:
    return _validate_optional(session_id, "session_id")


def validate_owner_type(owner_type: str) -> OwnerType:
    normalized = _require_non_empty(owner_type, "owner_type")
    allowed = {"agent", "user", "workspace", "system"}
    if normalized not in allowed:
        allowed_display = ", ".join(sorted(allowed))
        raise ValueError(f"owner_type must be one of: {allowed_display}")
    return normalized  # type: ignore[return-value]


def validate_owner_id(owner_id: str) -> str:
    return _require_non_empty(owner_id, "owner_id")


@dataclass(frozen=True)
class RuntimeContext:
    tenant_id: str
    agent_id: str
    request_id: str
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", validate_tenant_id(self.tenant_id))
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        object.__setattr__(self, "request_id", validate_request_id(self.request_id))
        object.__setattr__(self, "user_id", validate_user_id(self.user_id))
        object.__setattr__(self, "workspace_id", validate_workspace_id(self.workspace_id))
        object.__setattr__(self, "session_id", validate_session_id(self.session_id))


@dataclass(frozen=True)
class SessionScope:
    tenant_id: str
    agent_id: str
    session_id: str
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", validate_tenant_id(self.tenant_id))
        object.__setattr__(self, "agent_id", validate_agent_id(self.agent_id))
        object.__setattr__(self, "session_id", _require_non_empty(self.session_id, "session_id"))
        object.__setattr__(self, "user_id", validate_user_id(self.user_id))
        object.__setattr__(self, "workspace_id", validate_workspace_id(self.workspace_id))


@dataclass(frozen=True)
class OwnershipRef:
    tenant_id: str
    owner_type: OwnerType
    owner_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", validate_tenant_id(self.tenant_id))
        object.__setattr__(self, "owner_type", validate_owner_type(self.owner_type))
        object.__setattr__(self, "owner_id", validate_owner_id(self.owner_id))


@dataclass(frozen=True)
class TargetOwner:
    tenant_id: str
    owner_type: OwnerType
    owner_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", validate_tenant_id(self.tenant_id))
        object.__setattr__(self, "owner_type", validate_owner_type(self.owner_type))
        object.__setattr__(self, "owner_id", validate_owner_id(self.owner_id))
