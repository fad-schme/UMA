from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from uma.common.types import RuntimeContext
from uma.common.identity import normalize_user_id
from uma.retrieve.planner import RetrievalPlan


ScopedOwnerType = Literal["agent", "user"]


def _validate_owner_type(owner_type: str) -> ScopedOwnerType:
    normalized = str(owner_type or "").strip().lower()
    if normalized not in {"agent", "user"}:
        raise ValueError("owner_type must be one of: agent, user")
    return normalized  # type: ignore[return-value]


def _validate_owner_id(owner_id: str) -> str:
    value = str(owner_id or "").strip()
    if not value:
        raise ValueError("owner_id must be a non-empty string")
    return value


@dataclass(frozen=True)
class RetrievalScope:
    owner_type: ScopedOwnerType
    owner_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_type", _validate_owner_type(self.owner_type))
        object.__setattr__(self, "owner_id", _validate_owner_id(self.owner_id))


@dataclass(frozen=True)
class RetrievalRequest:
    context: RuntimeContext
    normalized_user_id: str
    scopes: Tuple[RetrievalScope, ...]
    trace_id: Optional[str] = None
    include_legacy_turn_data: bool = False
    plan: Optional[RetrievalPlan] = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, RuntimeContext):
            raise TypeError("RetrievalRequest.context must be a RuntimeContext")
        if not self.context.user_id:
            raise ValueError("RetrievalRequest requires RuntimeContext.user_id")
        normalized = normalize_user_id(self.context.user_id)
        object.__setattr__(self, "normalized_user_id", normalized)
        normalized_scopes = tuple(self.scopes or ())
        if not normalized_scopes:
            raise ValueError("RetrievalRequest.scopes must be non-empty")
        object.__setattr__(self, "scopes", normalized_scopes)
        if self.trace_id is not None:
            object.__setattr__(self, "trace_id", str(self.trace_id).strip() or None)

    @classmethod
    def from_runtime_context(
        cls,
        context: RuntimeContext,
        *,
        trace_id: Optional[str] = None,
        include_legacy_turn_data: bool = False,
        plan: Optional[RetrievalPlan] = None,
    ) -> "RetrievalRequest":
        normalized_user_id = normalize_user_id(context.user_id or "")
        return cls(
            context=context,
            normalized_user_id=normalized_user_id,
            scopes=(
                RetrievalScope(owner_type="agent", owner_id=context.agent_id),
                RetrievalScope(owner_type="user", owner_id=normalized_user_id),
            ),
            trace_id=trace_id or context.request_id,
            include_legacy_turn_data=bool(include_legacy_turn_data),
            plan=plan,
        )

    def scopes_for_owner_type(self, owner_type: Optional[str] = None) -> Tuple[RetrievalScope, ...]:
        if owner_type is None:
            return self.scopes
        normalized = _validate_owner_type(owner_type)
        return tuple(scope for scope in self.scopes if scope.owner_type == normalized)
