"""MCP TokenVerifier implementations for UMA's HTTP mode.

Two verifiers ship, wired into ``uma-mcp --http`` via
``FastMCP(..., token_verifier=verifier)`` and mutually exclusive at the
CLI level:

``UMATokenVerifier``
    Consults the persistent ``TokenStore`` (SQLite). Opaque bearer
    tokens minted by ``uma auth create``. Used by clients that accept
    a paste-in bearer (Perplexity, Cowork, Claude Desktop remote). No
    external IdP required. Phase 2.

``UMAJWTVerifier``
    Validates JWTs signed by an external OAuth 2.1 authorization server
    (Auth0 / Microsoft Entra ID / Google / Okta / any RFC 8414-compliant
    IdP). Required for clients that only accept OAuth (ChatGPT). No UMA
    OAuth server is stood up — the SDK's MCP-side is a resource server
    only, per the current spec. Phase 3, ``[oauth]`` extra.

Both verifiers return an ``AccessToken`` whose ``client_id`` encodes
``"{tenant_id}:{user_id}"``. Tools resolve that back into a scope pair
via ``parse_client_id``. This keeps the DAT-invariant plumbing identical
across bearer-mode and JWT-mode.

Requires the ``mcp`` optional extra (``pip install 'uma-mem[mcp]'``).
The JWT verifier additionally requires the ``oauth`` extra
(``pip install 'uma-mem[mcp,oauth]'``). The stdlib-only ``TokenStore``
lives in ``uma.mcp.tokens`` so the CLI subcommands can operate the store
without needing this module.
"""

from __future__ import annotations

import logging
from typing import Optional

from uma.common.types.types_scope import DEFAULT_TENANT_ID

from mcp.server.auth.provider import AccessToken, TokenVerifier

from uma.mcp.tokens import TokenStore

logger = logging.getLogger("uma.mcp.auth")

# Every valid token grants the full read+write scope set at v0.2.0. Scope
# refinement (read-only tokens, admin tokens) is deliberately not shipped
# — no user has asked for it and adding it now is speculation. Every tool
# still enforces the DAT invariant via `client_id` parsing.
_DEFAULT_SCOPES = ("read", "write")


def make_client_id(tenant_id: str, user_id: str) -> str:
    """Encode the DAT scope pair into an OAuth-style client_id string."""
    # ":" is not valid in a UMA tenant_id or user_id (both are opaque
    # identifiers the caller supplies via CLI args that reject anything
    # other than the exact-scope-value pattern, per `uma/cli/scopes.py`).
    return f"{tenant_id}:{user_id}"


def parse_client_id(client_id: str) -> tuple[str, str]:
    """Reverse of ``make_client_id``. Raises ValueError on malformed input."""
    if ":" not in client_id:
        raise ValueError(
            f"malformed client_id {client_id!r}: expected 'tenant:user'"
        )
    tenant_id, _, user_id = client_id.partition(":")
    if not tenant_id or not user_id:
        raise ValueError(
            f"malformed client_id {client_id!r}: empty tenant or user"
        )
    return tenant_id, user_id


class UMATokenVerifier(TokenVerifier):
    """Bearer-token verifier backed by ``TokenStore``.

    verify_token is called by the MCP SDK on every request. A None return
    triggers a 401 with ``WWW-Authenticate: Bearer`` — the SDK handles
    the response shape.
    """

    def __init__(self, store: TokenStore) -> None:
        self._store = store

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        record = await self._store.verify(token)
        if record is None:
            # Do not log the presented token — even a truncated preview
            # gives an attacker a length signal. Just note the rejection.
            logger.info("mcp.auth: token rejected")
            return None

        logger.info(
            "mcp.auth: token accepted token_id=%s tenant_id=%s user_id=%s",
            record.token_id,
            record.tenant_id,
            record.user_id,
        )
        return AccessToken(
            token=token,
            client_id=make_client_id(record.tenant_id, record.user_id),
            scopes=list(_DEFAULT_SCOPES),
            expires_at=None,
        )


class UMAJWTVerifier(TokenVerifier):
    """JWT verifier for OAuth 2.1 flow (ChatGPT, Auth0, Entra, etc.).

    The MCP server acts as a pure OAuth 2.1 resource server per the
    modern MCP spec — it validates JWTs issued by an external
    authorization server and never issues tokens itself. Signature,
    issuer, audience, and expiry are checked by PyJWT; the ``sub`` claim
    becomes the UMA ``user_id`` and the operator-configured
    ``tenant_id`` becomes the tenant scope.

    Requires ``pip install 'uma-mem[mcp,oauth]'``.

    Constructor arguments:
        issuer: OAuth 2.1 issuer URL (matches the ``iss`` claim).
        audience: Expected ``aud`` claim value — typically the resource
            server URL. Prevents replay of tokens minted for other
            services in the same IdP.
        jwks_uri: JWKS endpoint the IdP publishes. Defaults to
            ``{issuer}/.well-known/jwks.json``.
        tenant_id: The tenant every authenticated user maps to. v0.3.0
            is single-tenant per OAuth deployment — multi-tenant
            mapping (via custom claims) is deferred until asked for.
        required_scopes: If set, the JWT's ``scope`` claim must contain
            every listed scope; otherwise the token is rejected.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_uri: Optional[str] = None,
        tenant_id: str = DEFAULT_TENANT_ID,
        required_scopes: Optional[list[str]] = None,
    ) -> None:
        # Deferred import so `uma[mcp]` (Phase 2) still works without
        # PyJWT installed. Only the [oauth] extra pulls it in.
        try:
            import jwt as _pyjwt  # noqa: F401 — imported for side-effect check
            from jwt import PyJWKClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "UMAJWTVerifier requires the 'oauth' extra. Install "
                "with: pip install 'uma-mem[mcp,oauth]'"
            ) from exc

        if not issuer.strip() or not audience.strip():
            raise ValueError("issuer and audience must be non-empty")
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")

        # NB: do NOT strip trailing slash from issuer. IdPs like Auth0
        # emit tokens with iss=`https://x.auth0.com/` (trailing slash)
        # and PyJWT's iss check is exact-string. Store as-configured so
        # the operator has one source of truth. The JWKS URI join below
        # tolerates either shape.
        self._issuer = issuer
        self._audience = audience
        self._tenant_id = tenant_id
        self._required_scopes = tuple(required_scopes or ())
        self._jwks_uri = jwks_uri or (
            f"{issuer.rstrip('/')}/.well-known/jwks.json"
        )
        # PyJWKClient caches keys with sane defaults and rotates
        # automatically when the JWKS endpoint publishes new kids.
        self._jwks_client = PyJWKClient(self._jwks_uri)

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        import jwt as pyjwt

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            claims = pyjwt.decode(
                token,
                signing_key.key,
                # Restrict to the two RSA/ECDSA algorithms every real IdP
                # uses. HS256 is deliberately not accepted — a shared
                # secret in JWKS is an anti-pattern that has produced
                # multiple real-world confusions.
                algorithms=["RS256", "ES256"],
                audience=self._audience,
                issuer=self._issuer,
                options={
                    "require": ["exp", "iss", "aud", "sub"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )
        except pyjwt.PyJWTError as exc:
            # Cover: expired, bad signature, wrong iss/aud, malformed,
            # unknown kid. All map to a 401 for the client.
            logger.info("mcp.auth: jwt rejected reason=%s", type(exc).__name__)
            return None

        subject = str(claims.get("sub", "")).strip()
        if not subject:
            logger.info("mcp.auth: jwt rejected reason=empty_sub_claim")
            return None

        scopes = self._extract_scopes(claims)
        if self._required_scopes:
            missing = [s for s in self._required_scopes if s not in scopes]
            if missing:
                # NB: RFC 6750 says insufficient_scope should be 403,
                # not 401. The SDK's TokenVerifier surface only lets us
                # accept or reject at the wire level (returning None
                # here → 401). Documented tradeoff in CHATGPT.md; if a
                # future SDK version exposes a distinct return value
                # for scope mismatch, revisit.
                logger.info(
                    "mcp.auth: jwt rejected reason=missing_scopes missing=%s",
                    missing,
                )
                return None

        logger.info(
            "mcp.auth: jwt accepted tenant_id=%s user_id=%s issuer=%s",
            self._tenant_id,
            subject,
            self._issuer,
        )
        return AccessToken(
            token=token,
            client_id=make_client_id(self._tenant_id, subject),
            scopes=scopes or list(_DEFAULT_SCOPES),
            expires_at=int(claims["exp"]),
        )

    @staticmethod
    def _extract_scopes(claims: dict) -> list[str]:
        """Handle both scope-string and scope-list JWT conventions.

        RFC 8693 uses ``scope`` as a space-separated string. Some IdPs
        (Auth0, older) use ``scp`` or an array. Accept all three shapes.
        """
        raw = claims.get("scope") or claims.get("scp") or []
        if isinstance(raw, str):
            return [s for s in raw.split() if s]
        if isinstance(raw, (list, tuple)):
            return [str(s) for s in raw if str(s).strip()]
        return []


__all__ = [
    "UMATokenVerifier",
    "UMAJWTVerifier",
    "make_client_id",
    "parse_client_id",
]
