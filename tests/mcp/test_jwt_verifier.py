"""Tests for uma.mcp.auth.UMAJWTVerifier.

Real RSA signing via cryptography + PyJWT. The JWKS endpoint is stubbed —
we monkey-patch ``jwt.PyJWKClient`` so tests never make network calls.

Skipped when the `oauth` optional extra (PyJWT + cryptography) isn't
installed. Also skipped when the `mcp` extra isn't installed since the
verifier subclasses ``mcp.server.auth.provider.TokenVerifier``.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any

import pytest

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None
JWT_AVAILABLE = importlib.util.find_spec("jwt") is not None
CRYPTO_AVAILABLE = importlib.util.find_spec("cryptography") is not None

pytestmark = pytest.mark.skipif(
    not (MCP_AVAILABLE and JWT_AVAILABLE and CRYPTO_AVAILABLE),
    reason="mcp+oauth extras not installed (pip install 'uma-mem[mcp,oauth]')",
)


# ---------------------------------------------------------------------------
# Fixtures — real RSA keys, stubbed JWKS client
# ---------------------------------------------------------------------------
ISSUER = "https://acme.auth0.com/"  # Auth0-style, with trailing slash
AUDIENCE = "https://uma.example.com/mcp"
TENANT = "prod"


@pytest.fixture(scope="module")
def rsa_keypair():
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    )
    return private_key, private_key.public_key()


@pytest.fixture(scope="module")
def other_rsa_keypair():
    """Distinct keypair for signature-mismatch tests."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    )
    return private_key, private_key.public_key()


@pytest.fixture(autouse=True)
def stub_jwks(monkeypatch, rsa_keypair):
    """Replace jwt.PyJWKClient with a stub that returns our test public key.

    Auto-used by every test in this module so no test ever reaches the
    network. If a test wants to simulate JWKS returning a different key,
    it should re-stub after this fixture.
    """
    import jwt as pyjwt

    _, public_key = rsa_keypair

    class _StubSigningKey:
        def __init__(self, key: Any) -> None:
            self.key = key

    class _StubJWKClient:
        def __init__(self, uri: str) -> None:
            self.uri = uri

        def get_signing_key_from_jwt(self, token: str) -> _StubSigningKey:
            return _StubSigningKey(public_key)

    monkeypatch.setattr(pyjwt, "PyJWKClient", _StubJWKClient)


@pytest.fixture
def mint(rsa_keypair):
    """Return a helper that mints a signed JWT with overrideable claims."""
    private_key, _ = rsa_keypair

    def _mint(**overrides: Any) -> str:
        import jwt as pyjwt

        claims: dict[str, Any] = {
            "sub": "alice",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        claims.update(overrides)
        return pyjwt.encode(claims, private_key, algorithm="RS256")

    return _mint


@pytest.fixture
def verifier():
    from uma.mcp.auth import UMAJWTVerifier

    return UMAJWTVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        tenant_id=TENANT,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def test_constructor_rejects_empty_scope_values():
    from uma.mcp.auth import UMAJWTVerifier

    for kwargs in (
        {"issuer": "", "audience": "a", "tenant_id": "t"},
        {"issuer": "i", "audience": "", "tenant_id": "t"},
        {"issuer": "i", "audience": "a", "tenant_id": ""},
    ):
        with pytest.raises(ValueError):
            UMAJWTVerifier(**kwargs)


def test_default_jwks_uri_derivation():
    from uma.mcp.auth import UMAJWTVerifier

    v = UMAJWTVerifier(
        issuer="https://acme.auth0.com/",
        audience="aud",
        tenant_id="t",
    )
    # No double slash — a foot-gun a naive f-string join would trigger.
    assert v._jwks_uri == "https://acme.auth0.com/.well-known/jwks.json"  # noqa: SLF001


def test_default_jwks_uri_derivation_without_trailing_slash():
    from uma.mcp.auth import UMAJWTVerifier

    v = UMAJWTVerifier(
        issuer="https://acme.auth0.com",
        audience="aud",
        tenant_id="t",
    )
    assert v._jwks_uri == "https://acme.auth0.com/.well-known/jwks.json"  # noqa: SLF001


def test_custom_jwks_uri_honored():
    from uma.mcp.auth import UMAJWTVerifier

    v = UMAJWTVerifier(
        issuer="https://acme.auth0.com/",
        audience="aud",
        tenant_id="t",
        jwks_uri="https://custom/keys",
    )
    assert v._jwks_uri == "https://custom/keys"  # noqa: SLF001


def test_issuer_preserved_verbatim():
    """Trailing slash must be preserved — IdPs like Auth0 emit iss WITH
    the slash; PyJWT's iss check is exact-string. A silent strip here
    would produce silent InvalidIssuerError for every JWT."""
    from uma.mcp.auth import UMAJWTVerifier

    v_with = UMAJWTVerifier(
        issuer="https://acme.auth0.com/",
        audience="aud",
        tenant_id="t",
    )
    v_without = UMAJWTVerifier(
        issuer="https://acme.auth0.com",
        audience="aud",
        tenant_id="t",
    )
    assert v_with._issuer == "https://acme.auth0.com/"  # noqa: SLF001
    assert v_without._issuer == "https://acme.auth0.com"  # noqa: SLF001


# ---------------------------------------------------------------------------
# verify_token — happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_valid_jwt_produces_access_token(verifier, mint):
    from uma.mcp.auth import parse_client_id

    token = mint(sub="alice")
    at = await verifier.verify_token(token)

    assert at is not None
    tenant, user = parse_client_id(at.client_id)
    assert tenant == TENANT
    assert user == "alice"
    assert at.token == token
    assert at.expires_at > int(time.time())


@pytest.mark.asyncio
async def test_realistic_idp_sub_shapes_accepted(verifier, mint):
    """Real IdPs emit string subs with provider-specific shapes:
    Auth0 -> "auth0|65f2a3c8b1a2c3d4e5f6a7b8"
    Google -> "109357281928374651829"  (numeric-looking but still a string)
    Okta  -> "00uixa271s6x7qt8I0h7"

    All are strings per RFC 7519 §4.1.2 / OIDC Core §5.1. The verifier
    passes them through unchanged as the UMA user_id.
    """
    from uma.mcp.auth import parse_client_id

    for realistic_sub in (
        "auth0|65f2a3c8b1a2c3d4e5f6a7b8",
        "109357281928374651829",  # str, not int — Google-style
        "00uixa271s6x7qt8I0h7",
    ):
        token = mint(sub=realistic_sub)
        at = await verifier.verify_token(token)
        assert at is not None, f"realistic sub {realistic_sub!r} was rejected"
        _, user = parse_client_id(at.client_id)
        assert user == realistic_sub


# ---------------------------------------------------------------------------
# verify_token — rejection paths (the security-critical ones)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_expired_token_rejected(verifier, mint):
    token = mint(exp=int(time.time()) - 60)
    assert await verifier.verify_token(token) is None


@pytest.mark.asyncio
async def test_wrong_audience_rejected(verifier, mint):
    """Blocks token replay from other services in the same IdP."""
    token = mint(aud="different-service")
    assert await verifier.verify_token(token) is None


@pytest.mark.asyncio
async def test_wrong_issuer_rejected(verifier, mint):
    token = mint(iss="https://malicious.example.com/")
    assert await verifier.verify_token(token) is None


@pytest.mark.asyncio
async def test_issuer_trailing_slash_mismatch_rejected(verifier, mint):
    """Verifier configured with trailing slash rejects JWTs without one."""
    token = mint(iss="https://acme.auth0.com")  # no slash
    assert await verifier.verify_token(token) is None


@pytest.mark.asyncio
async def test_missing_sub_rejected(verifier, rsa_keypair):
    """A JWT without `sub` cannot identify a user — must not be
    accepted just because signature/iss/aud check out."""
    import jwt as pyjwt

    private_key, _ = rsa_keypair
    token = pyjwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": int(time.time()) + 3600,
        },
        private_key,
        algorithm="RS256",
    )
    assert await verifier.verify_token(token) is None


@pytest.mark.asyncio
async def test_bad_signature_rejected(
    verifier, other_rsa_keypair,
):
    """JWT signed with a different key must not verify."""
    import jwt as pyjwt

    other_private, _ = other_rsa_keypair
    token = pyjwt.encode(
        {
            "sub": "alice",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": int(time.time()) + 3600,
        },
        other_private,
        algorithm="RS256",
    )
    assert await verifier.verify_token(token) is None


@pytest.mark.asyncio
async def test_hs256_downgrade_rejected(verifier):
    """HS256 tokens must be rejected. Accepting HS256 alongside RS256
    opens algorithm-confusion attacks where the public key is used as
    the HMAC secret.

    The 32-byte key satisfies RFC 7518 §3.2 minimum length so PyJWT
    doesn't emit an InsecureKeyLengthWarning during test collection —
    what we're validating is our algorithm allowlist rejecting HS256,
    not PyJWT's key-length policy.
    """
    import jwt as pyjwt

    hs256_key = "a" * 32  # 32 bytes, RFC 7518 §3.2-compliant
    token = pyjwt.encode(
        {
            "sub": "alice",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": int(time.time()) + 3600,
        },
        hs256_key,
        algorithm="HS256",
    )
    assert await verifier.verify_token(token) is None


@pytest.mark.asyncio
async def test_malformed_token_rejected(verifier):
    assert await verifier.verify_token("garbage") is None
    assert await verifier.verify_token("") is None
    assert await verifier.verify_token("not.a.jwt.at.all") is None


# ---------------------------------------------------------------------------
# Scope enforcement
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_required_scopes_accept_when_all_present(mint):
    from uma.mcp.auth import UMAJWTVerifier

    v = UMAJWTVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        tenant_id=TENANT,
        required_scopes=["mcp:read", "mcp:write"],
    )
    token = mint(scope="mcp:read mcp:write extra:noise")
    assert await v.verify_token(token) is not None


@pytest.mark.asyncio
async def test_required_scopes_reject_when_missing(mint):
    from uma.mcp.auth import UMAJWTVerifier

    v = UMAJWTVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        tenant_id=TENANT,
        required_scopes=["mcp:read", "mcp:write"],
    )
    token = mint(scope="mcp:read")
    assert await v.verify_token(token) is None


@pytest.mark.asyncio
async def test_scope_list_form_accepted(mint):
    """Auth0 often emits `scope` as an array, not a space-string. The
    verifier must accept either shape."""
    from uma.mcp.auth import UMAJWTVerifier

    v = UMAJWTVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        tenant_id=TENANT,
        required_scopes=["mcp:read"],
    )
    token = mint(scope=["mcp:read", "mcp:write"])
    assert await v.verify_token(token) is not None


@pytest.mark.asyncio
async def test_scp_claim_also_accepted(mint):
    """Older IdPs emit `scp` instead of `scope`. Accept both."""
    from uma.mcp.auth import UMAJWTVerifier

    v = UMAJWTVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        tenant_id=TENANT,
        required_scopes=["mcp:read"],
    )
    token = mint(scp="mcp:read mcp:write")
    assert await v.verify_token(token) is not None


@pytest.mark.asyncio
async def test_no_required_scopes_allows_any_valid_jwt(verifier, mint):
    """A verifier constructed without required_scopes must accept a JWT
    that has no scope claim at all."""
    token = mint()  # no scope claim
    assert await verifier.verify_token(token) is not None
