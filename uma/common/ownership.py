from __future__ import annotations

from typing import Collection, Optional

from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types import OwnershipRef, TargetOwner, make_target_owner
from uma.common.types.types_scope import validate_owner_type
from uma.common.identity import normalize_user_id


def _normalize_owner_id(owner_type: str, owner_id: str) -> str:
    normalized_owner_type = validate_owner_type(owner_type)
    if normalized_owner_type == "user":
        return normalize_user_id(owner_id)
    return owner_id


def resolve_target_owner(
    *,
    target_owner: Optional[TargetOwner] = None,
    tenant_id: Optional[str] = None,
    owner_type: Optional[str] = None,
    owner_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    allowed_owner_types: Optional[Collection[str]] = None,
) -> TargetOwner:
    owner_type_value = target_owner.owner_type if target_owner is not None else str(owner_type or "")
    owner_id_value = target_owner.owner_id if target_owner is not None else str(owner_id or "")
    workspace_id_value = target_owner.workspace_id if target_owner is not None else workspace_id
    tenant_id_value = (
        target_owner.tenant_id
        if target_owner is not None
        else (tenant_id or DEFAULT_TENANT_ID)
    )
    normalized_owner_id = _normalize_owner_id(owner_type_value, owner_id_value)
    normalized_owner_type = validate_owner_type(owner_type_value)
    if normalized_owner_type == "workspace" and not workspace_id_value:
        workspace_id_value = normalized_owner_id
    return make_target_owner(
        tenant_id=tenant_id_value,
        owner_type=normalized_owner_type,
        owner_id=normalized_owner_id,
        workspace_id=workspace_id_value,
        allowed_owner_types=allowed_owner_types,
    )


def resolve_ownership_ref(
    *,
    owner: Optional[OwnershipRef] = None,
    tenant_id: Optional[str] = None,
    owner_type: Optional[str] = None,
    owner_id: Optional[str] = None,
    allowed_owner_types: Optional[Collection[str]] = None,
) -> OwnershipRef:
    owner_type_value = owner.owner_type if owner is not None else str(owner_type or "")
    owner_id_value = owner.owner_id if owner is not None else str(owner_id or "")
    tenant_id_value = owner.tenant_id if owner is not None else (tenant_id or DEFAULT_TENANT_ID)
    normalized_owner_type = validate_owner_type(owner_type_value)
    if allowed_owner_types is not None:
        allowed = {validate_owner_type(value) for value in allowed_owner_types}
        if normalized_owner_type not in allowed:
            allowed_display = ", ".join(sorted(allowed))
            raise ValueError(f"owner_type must be one of: {allowed_display}")
    return OwnershipRef(
        tenant_id=tenant_id_value,
        owner_type=normalized_owner_type,
        owner_id=_normalize_owner_id(normalized_owner_type, owner_id_value),
    )
