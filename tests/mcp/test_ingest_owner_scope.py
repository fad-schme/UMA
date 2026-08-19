"""Owner-scope enforcement for the `ingest_document` MCP tool.

An ingested document is retrieved later as trusted context, so the owner
tuple decides whose memory a caller can write into. These tests cover the
boundary in both transports: stdio, where the calling application asserts
identity, and HTTP, where the bearer token is authoritative.

Skipped if the `mcp` optional extra isn't installed.
"""

from __future__ import annotations

import importlib.util

import pytest

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None
pytestmark = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason="mcp extra not installed (pip install 'uma-mem[mcp]')",
)

if MCP_AVAILABLE:
    from mcp.server.auth.provider import AccessToken

    from uma.mcp import server as srv
    from uma.mcp.auth import make_client_id


AGENT_ID = "agent-ingest"


@pytest.fixture
def http_mode(monkeypatch: pytest.MonkeyPatch):
    """Put `_resolve_scope` in HTTP mode with a bearer token for one user."""

    def _install(tenant_id: str = "default", user_id: str = "user:alpha"):
        token = AccessToken(
            token="opaque",
            client_id=make_client_id(tenant_id, user_id),
            scopes=["read", "write"],
            expires_at=None,
        )
        import mcp.server.auth.middleware.auth_context as auth_context

        monkeypatch.setattr(auth_context, "get_access_token", lambda: token)
        return tenant_id, user_id

    return _install


# ── stdio mode ────────────────────────────────────────────────────────


def test_agent_ingest_needs_no_user_in_stdio_mode() -> None:
    """The common call — agent-owned ingest, no user anywhere in sight."""
    assert srv._resolve_scope("", "default", require_user=False) == ("", "default")
    assert srv._resolve_ingest_owner(
        agent_id=AGENT_ID,
        owner_type="agent",
        owner_id="",
        caller_user_id="",
    ) == ("agent", AGENT_ID)


def test_agent_ingest_rejects_a_foreign_agent_owner() -> None:
    with pytest.raises(ValueError, match="does not match the calling agent"):
        srv._resolve_ingest_owner(
            agent_id=AGENT_ID,
            owner_type="agent",
            owner_id="some-other-agent",
            caller_user_id="",
        )


def test_user_ingest_uses_the_asserted_user_in_stdio_mode() -> None:
    """No token means the calling application is the authority on identity."""
    assert srv._resolve_ingest_owner(
        agent_id=AGENT_ID,
        owner_type="user",
        owner_id="",
        caller_user_id="alpha",
    ) == ("user", "user:alpha")


def test_user_ingest_without_any_user_raises() -> None:
    with pytest.raises(ValueError, match="user_id is required for a user-owned ingest"):
        srv._resolve_ingest_owner(
            agent_id=AGENT_ID,
            owner_type="user",
            owner_id="",
            caller_user_id="",
        )


@pytest.mark.parametrize("owner_type", ["workspace", "system", "", "  "])
def test_unsupported_owner_types_are_rejected(owner_type: str) -> None:
    """Only the two scopes retrieval reads back are writable over MCP."""
    with pytest.raises(ValueError, match="owner_type must be one of"):
        srv._resolve_ingest_owner(
            agent_id=AGENT_ID,
            owner_type=owner_type,
            owner_id="",
            caller_user_id="user:alpha",
        )


@pytest.mark.parametrize("owner_type", ["AGENT", " agent "])
def test_owner_type_is_case_and_whitespace_insensitive(owner_type: str) -> None:
    assert srv._resolve_ingest_owner(
        agent_id=AGENT_ID,
        owner_type=owner_type,
        owner_id="",
        caller_user_id="",
    ) == ("agent", AGENT_ID)


# ── HTTP mode ─────────────────────────────────────────────────────────


def test_agent_ingest_resolves_scope_in_http_mode(http_mode) -> None:
    """Regression: the old placeholder user_id made this raise every time."""
    tenant_id, user_id = http_mode()
    assert srv._resolve_scope("", tenant_id, require_user=False) == (user_id, tenant_id)


def test_user_ingest_defaults_to_the_token_user(http_mode) -> None:
    tenant_id, user_id = http_mode()
    caller_user_id, _ = srv._resolve_scope("", tenant_id, require_user=False)
    assert srv._resolve_ingest_owner(
        agent_id=AGENT_ID,
        owner_type="user",
        owner_id="",
        caller_user_id=caller_user_id,
    ) == ("user", user_id)


def test_user_ingest_cannot_target_another_user(http_mode) -> None:
    """The write primitive into someone else's private lane."""
    tenant_id, _ = http_mode()
    caller_user_id, _ = srv._resolve_scope("", tenant_id, require_user=False)
    with pytest.raises(ValueError, match="does not match the calling user"):
        srv._resolve_ingest_owner(
            agent_id=AGENT_ID,
            owner_type="user",
            owner_id="user:victim",
            caller_user_id=caller_user_id,
        )


def test_user_ingest_accepts_the_token_user_in_either_subject_form(http_mode) -> None:
    """"alpha" and "user:alpha" are the same principal, not two."""
    tenant_id, user_id = http_mode(user_id="alpha")
    caller_user_id, _ = srv._resolve_scope("", tenant_id, require_user=False)
    assert srv._resolve_ingest_owner(
        agent_id=AGENT_ID,
        owner_type="user",
        owner_id="user:alpha",
        caller_user_id=caller_user_id,
    ) == ("user", "user:alpha")


def test_explicit_user_id_argument_still_must_match_the_token(http_mode) -> None:
    tenant_id, _ = http_mode()
    with pytest.raises(ValueError, match="does not match authenticated user"):
        srv._resolve_scope("user:victim", tenant_id, require_user=False)


def test_tenant_mismatch_still_raises_for_ingest(http_mode) -> None:
    http_mode(tenant_id="tenant-a")
    with pytest.raises(ValueError, match="does not match authenticated tenant"):
        srv._resolve_scope("", "tenant-b", require_user=False)
