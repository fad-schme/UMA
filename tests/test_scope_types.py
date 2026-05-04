from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from uma.common.types import (
    OwnerType,
    OwnershipRef,
    RuntimeContext,
    SCOPE_MODEL_VERSION,
    SessionScope,
)
from uma.common.types.types_scope import (
    validate_agent_id,
    validate_owner_id,
    validate_owner_type,
    validate_request_id,
    validate_session_id,
    validate_tenant_id,
    validate_user_id,
    validate_workspace_id,
)


def test_scope_model_version_is_exported() -> None:
    assert SCOPE_MODEL_VERSION == "v2"


def test_runtime_context_construction_succeeds() -> None:
    ctx = RuntimeContext(
        tenant_id="tenant-1",
        agent_id="agent-1",
        request_id="req-1",
        user_id="user-1",
        workspace_id="workspace-1",
        session_id="session-1",
    )

    assert ctx.tenant_id == "tenant-1"
    assert ctx.agent_id == "agent-1"
    assert ctx.request_id == "req-1"
    assert ctx.user_id == "user-1"
    assert ctx.workspace_id == "workspace-1"
    assert ctx.session_id == "session-1"


def test_session_scope_construction_succeeds() -> None:
    scope = SessionScope(
        tenant_id="tenant-1",
        agent_id="agent-1",
        session_id="session-1",
        user_id="user-1",
        workspace_id="workspace-1",
    )

    assert scope.tenant_id == "tenant-1"
    assert scope.agent_id == "agent-1"
    assert scope.session_id == "session-1"


@pytest.mark.parametrize("owner_type", ["agent", "user", "workspace", "system"])
def test_ownership_ref_accepts_supported_owner_types(owner_type: str) -> None:
    ref = OwnershipRef(
        tenant_id="tenant-1",
        owner_type=owner_type,
        owner_id="owner-1",
    )
    assert ref.owner_type == owner_type


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tenant_id": "", "agent_id": "agent-1", "request_id": "req-1"}, "tenant_id"),
        ({"tenant_id": "tenant-1", "agent_id": "", "request_id": "req-1"}, "agent_id"),
        ({"tenant_id": "tenant-1", "agent_id": "agent-1", "request_id": ""}, "request_id"),
        ({"tenant_id": "tenant-1", "agent_id": "agent-1", "request_id": "req-1", "user_id": ""}, "user_id"),
        (
            {"tenant_id": "tenant-1", "agent_id": "agent-1", "request_id": "req-1", "workspace_id": ""},
            "workspace_id",
        ),
        (
            {"tenant_id": "tenant-1", "agent_id": "agent-1", "request_id": "req-1", "session_id": ""},
            "session_id",
        ),
    ],
)
def test_runtime_context_rejects_invalid_values(kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RuntimeContext(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tenant_id": "", "agent_id": "agent-1", "session_id": "session-1"}, "tenant_id"),
        ({"tenant_id": "tenant-1", "agent_id": "", "session_id": "session-1"}, "agent_id"),
        ({"tenant_id": "tenant-1", "agent_id": "agent-1", "session_id": ""}, "session_id"),
        (
            {"tenant_id": "tenant-1", "agent_id": "agent-1", "session_id": "session-1", "user_id": ""},
            "user_id",
        ),
        (
            {
                "tenant_id": "tenant-1",
                "agent_id": "agent-1",
                "session_id": "session-1",
                "workspace_id": "",
            },
            "workspace_id",
        ),
    ],
)
def test_session_scope_rejects_invalid_values(kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SessionScope(**kwargs)


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (OwnershipRef, {"tenant_id": "", "owner_type": "user", "owner_id": "owner-1"}, "tenant_id"),
        (OwnershipRef, {"tenant_id": "tenant-1", "owner_type": "invalid", "owner_id": "owner-1"}, "owner_type"),
        (OwnershipRef, {"tenant_id": "tenant-1", "owner_type": "user", "owner_id": ""}, "owner_id"),
    ],
)
def test_ownership_types_reject_invalid_values(factory, kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


def test_runtime_scope_types_are_immutable() -> None:
    ctx = RuntimeContext(tenant_id="tenant-1", agent_id="agent-1", request_id="req-1")
    with pytest.raises(FrozenInstanceError):
        ctx.agent_id = "agent-2"  # type: ignore[misc]


def test_persistent_ownership_types_are_immutable() -> None:
    owner = OwnershipRef(tenant_id="tenant-1", owner_type="user", owner_id="owner-1")
    with pytest.raises(FrozenInstanceError):
        owner.owner_id = "owner-2"  # type: ignore[misc]


def test_validation_helpers_accept_none_for_optional_ids() -> None:
    assert validate_user_id(None) is None
    assert validate_workspace_id(None) is None
    assert validate_session_id(None) is None


@pytest.mark.parametrize(
    ("validator", "value", "expected"),
    [
        (validate_tenant_id, "tenant-1", "tenant-1"),
        (validate_agent_id, "agent-1", "agent-1"),
        (validate_request_id, "req-1", "req-1"),
        (validate_user_id, "user-1", "user-1"),
        (validate_workspace_id, "workspace-1", "workspace-1"),
        (validate_session_id, "session-1", "session-1"),
        (validate_owner_type, "workspace", "workspace"),
        (validate_owner_id, "owner-1", "owner-1"),
    ],
)
def test_validation_helpers_accept_valid_strings(validator, value: str, expected: str) -> None:
    assert validator(value) == expected


@pytest.mark.parametrize(
    ("validator", "value", "message"),
    [
        (validate_tenant_id, "", "tenant_id"),
        (validate_agent_id, "", "agent_id"),
        (validate_request_id, "", "request_id"),
        (validate_user_id, "", "user_id"),
        (validate_workspace_id, "", "workspace_id"),
        (validate_session_id, "", "session_id"),
        (validate_owner_type, "project", "owner_type"),
        (validate_owner_id, "", "owner_id"),
    ],
)
def test_validation_helpers_reject_invalid_strings(validator, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validator(value)


def test_owner_type_literal_matches_supported_vocabulary() -> None:
    assert set(get_args(OwnerType)) == {"agent", "user", "workspace", "system"}


def test_new_types_are_exported_from_uma_types() -> None:
    assert RuntimeContext.__module__ == "uma.common.types.types_scope"
    assert SessionScope.__module__ == "uma.common.types.types_scope"
    assert OwnershipRef.__module__ == "uma.common.types.types_scope"
