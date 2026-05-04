from __future__ import annotations

from typing import Optional

from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.common.types.types_scope import validate_owner_type, validate_tenant_id, validate_workspace_id
from uma.common.identity import normalize_user_id


def _normalize_owner_id(owner_type: str, owner_id: str) -> str:
    normalized_owner_type = validate_owner_type(owner_type)
    if normalized_owner_type == "user":
        return normalize_user_id(owner_id)
    return owner_id


def validate_explicit_owner(
    *,
    tenant_id: Optional[str] = None,
    owner_type: str,
    owner_id: str,
    workspace_id: Optional[str] = None,
) -> dict[str, str | None]:
    normalized_owner_type = validate_owner_type(owner_type)
    normalized_owner_id = _normalize_owner_id(normalized_owner_type, owner_id)
    normalized_workspace_id = validate_workspace_id(workspace_id)
    if normalized_owner_type == "workspace" and normalized_workspace_id is None:
        normalized_workspace_id = normalized_owner_id
    return {
        "tenant_id": validate_tenant_id(str(tenant_id or DEFAULT_TENANT_ID)),
        "owner_type": normalized_owner_type,
        "owner_id": normalized_owner_id,
        "workspace_id": normalized_workspace_id,
    }
