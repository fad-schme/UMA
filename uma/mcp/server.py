"""UMA MCP Server — exposes UMA memory operations as MCP tools.

Entry point: ``uma-mcp`` (registered as a console_script in pyproject.toml).

Two transports, one binary:
    uma-mcp                                    # stdio (default)
    uma-mcp --http --port 3131                 # HTTP + opaque bearer tokens
    uma-mcp --http --port 3131 \
        --oauth-issuer https://your-idp/       # HTTP + OAuth 2.1 JWT
        --oauth-audience https://you/mcp

Configuration via environment variables (both modes):
    UMA_CONFIG_PATH   Absolute path to uma.yaml (required).

The server holds no identity. Every tool call carries its own agent_id,
user_id, and (optionally) tenant_id, so one process serves every agent.

HTTP-mode flags:
    --http                       Enable HTTP transport (streamable-http).
    --port PORT                  Bind port (default: 3131).
    --bind HOST                  Bind interface. Default 127.0.0.1; set
                                 to 0.0.0.0 when serving through a tunnel.
    --public-url URL             Externally-reachable URL. Required when
                                 --bind is not loopback (RFC 8414 §3.3).
                                 Used as the OAuth issuer and resource
                                 identifiers. Defaults to
                                 http://{bind}:{port}/mcp for loopback.

Opaque bearer token mode (Phase 2 default when --http is set):
    --tokens-db PATH             SQLite path for the bearer-token store
                                 (default: .uma/db/mcp_tokens.db).

OAuth 2.1 JWT mode (Phase 3, mutually exclusive with --tokens-db):
    --oauth-issuer URL           OAuth issuer URL (matches JWT `iss` claim).
                                 Enabling this switches from opaque bearer
                                 to JWT verification via UMAJWTVerifier.
    --oauth-audience STR         Expected JWT `aud` claim. Required with
                                 --oauth-issuer.
    --oauth-jwks-uri URL         JWKS endpoint. Defaults to
                                 {issuer}/.well-known/jwks.json.
    --oauth-tenant TENANT        UMA tenant every authenticated user is
                                 mapped to (default: "default"). v0.3.0
                                 is single-tenant per OAuth deployment.
    --oauth-required-scope SCOPE Required scope; may be passed multiple
                                 times. All must be present in the JWT.

Bearer tokens are issued out-of-band via ``uma auth create``. JWT tokens
are issued by the external IdP — UMA never mints them. Verification for
both goes through a ``TokenVerifier`` subclass in ``uma.mcp.auth``.

Result shape on the wire: every tool returns a JSON string. Results that
come back from UMA as Pydantic v2 models (ContextBundle, MemoryResult,
HealthStatus) are serialized with ``.model_dump_json()``. IngestReport
(frozen dataclass) is serialized with ``dataclasses.asdict`` + json.dumps.
process_turn returns a small ok/status dict — the only tool where the
underlying API returns None.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from uma.common.types.types_scope import DEFAULT_TENANT_ID, validate_agent_id

# ---------------------------------------------------------------------------
# Logging — stderr only. stdout is reserved for the MCP JSON-RPC transport
# in stdio mode; anything the server writes to stdout corrupts the frame.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("uma.mcp")


# ---------------------------------------------------------------------------
# Lazy UMA singleton
# ---------------------------------------------------------------------------
_memory = None


def _get_memory():
    """Return the process-global UMAMemory, creating it on first use.

    The instance holds no identity: one server process serves every agent,
    every user, and the single tenant. Scope arrives with each tool call.
    """
    global _memory
    if _memory is None:
        config_path = os.environ.get("UMA_CONFIG_PATH")
        if not config_path:
            raise RuntimeError(
                "UMA_CONFIG_PATH environment variable is required. "
                "Point it at your uma.yaml (absolute path). See "
                "docs/mcp/STDIO_CLIENTS.md."
            )

        from uma.api.memory import UMAMemory

        logger.info("Initializing UMAMemory from %s", config_path)
        _memory = UMAMemory.from_yaml(config_path)
    return _memory


# ---------------------------------------------------------------------------
# Scope resolution — DAT invariant enforcement at the tool boundary
# ---------------------------------------------------------------------------
def _resolve_scope(
    user_id: str,
    tenant_id: str,
    *,
    require_user: bool = True,
) -> tuple[str, str]:
    """Resolve (user_id, tenant_id) for the current tool call.

    stdio mode: no AccessToken. The passed args are authoritative and
    ``user_id`` must be non-empty.

    HTTP mode: the bearer token is authoritative. Its client_id encodes
    ``{tenant_id}:{user_id}`` (see ``uma.mcp.auth.make_client_id``). If
    the tool call also passed non-default values for these args, they
    must match the token's claim — otherwise we raise, which surfaces to
    the caller as a tool error. This is the ASI03 boundary.

    ``require_user=False`` is for ``ingest_document``, whose scope is a
    durable owner tuple rather than a request scope: it always needs the
    tenant and the caller's identity, but an agent-owned ingest has no
    user of its own. It never relaxes the token check — in HTTP mode the
    returned user is still the token's, which is what the owner resolver
    below constrains against.
    """
    token = None
    try:
        # Import inside the function so stdio mode never triggers the auth
        # middleware import path.
        from mcp.server.auth.middleware.auth_context import get_access_token

        token = get_access_token()
    except Exception:
        # Auth middleware isn't installed (stdio mode) — get_access_token
        # may raise or return None depending on the SDK version. Both mean
        # the same thing: no auth context.
        token = None

    if token is None:
        # stdio mode — args are authoritative
        if require_user and not (user_id or "").strip():
            raise ValueError(
                "user_id is required when running in stdio mode "
                "(the LLM must pass it in every tool call)"
            )
        return (user_id or "").strip(), tenant_id

    # HTTP mode — token wins, but reject explicit mismatches to keep the
    # DAT invariant loud rather than silent.
    from uma.mcp.auth import parse_client_id

    token_tenant, token_user = parse_client_id(token.client_id)
    if user_id and user_id != token_user:
        raise ValueError(
            f"user_id={user_id!r} does not match authenticated user "
            f"(token identifies user={token_user!r})"
        )
    if tenant_id and tenant_id != DEFAULT_TENANT_ID and tenant_id != token_tenant:
        raise ValueError(
            f"tenant_id={tenant_id!r} does not match authenticated tenant "
            f"(token identifies tenant={token_tenant!r})"
        )
    return token_user, token_tenant


# Documents are readable only through the two scopes retrieval builds
# (`RetrievalScope` is agent|user), so those are the only two an ingest
# tool may target. A workspace- or system-owned document would be
# write-only over MCP.
_INGEST_OWNER_TYPES = ("agent", "user")


def _resolve_ingest_owner(
    *,
    agent_id: str,
    owner_type: str,
    owner_id: str,
    caller_user_id: str,
) -> tuple[str, str]:
    """Return the (owner_type, owner_id) this ingest call may write to.

    The owner tuple is never taken on trust from the arguments alone.
    Ingested documents are retrieved later as trusted context, so letting
    a caller name an arbitrary owner is a write primitive into someone
    else's memory:

    - ``agent``: the owner is the calling ``agent_id``. Naming a different
      agent raises rather than being silently ignored.
    - ``user``: the owner is the caller's own identity. In HTTP mode
      ``caller_user_id`` comes from the bearer token, so a user-owned
      ingest can only ever target that user. In stdio mode there is no
      token and the calling application is the authority on identity, as
      it is everywhere else in UMA — the asserted value is used as-is.

    ``owner_id`` stays accepted for both types so an explicit call still
    reads the same way; it just has to agree with the caller.
    """
    normalized_owner_type = (owner_type or "").strip().lower()
    if normalized_owner_type not in _INGEST_OWNER_TYPES:
        raise ValueError(
            f"owner_type must be one of {list(_INGEST_OWNER_TYPES)}; "
            f"got {owner_type!r}"
        )

    requested_owner_id = (owner_id or "").strip()

    if normalized_owner_type == "agent":
        if requested_owner_id and requested_owner_id != agent_id:
            raise ValueError(
                f"owner_id={requested_owner_id!r} does not match the calling "
                f"agent (agent_id={agent_id!r}); an agent-owned document is "
                f"owned by the agent that ingests it"
            )
        return "agent", agent_id

    from uma.common.identity import normalize_user_id

    if not caller_user_id:
        raise ValueError(
            "user_id is required for a user-owned ingest "
            "(owner_type='user') when running in stdio mode"
        )
    resolved_owner_id = normalize_user_id(requested_owner_id or caller_user_id)
    if resolved_owner_id != normalize_user_id(caller_user_id):
        raise ValueError(
            f"owner_id={requested_owner_id!r} does not match the calling user "
            f"({caller_user_id!r}); a user-owned document is owned by the "
            f"user that ingests it"
        )
    return "user", resolved_owner_id


# ---------------------------------------------------------------------------
# Tool functions (bare — registered onto whichever FastMCP instance the
# transport mode selects). Signatures match the stdio contract; HTTP mode
# allows user_id/tenant_id to be empty since the token supplies them.
# ---------------------------------------------------------------------------
async def retrieve_context(
    query_text: str,
    agent_id: str,
    user_id: str = "",
    session_id: str = "",
    tenant_id: str = DEFAULT_TENANT_ID,
) -> str:
    """Retrieve curated RAG context from UMA for the given query.

    Returns a JSON ContextBundle: snippets, facts, working_memory, meta,
    and query_scan_severity. agent_id is required on every call and names
    the calling agent. In stdio mode user_id is required too; in HTTP mode
    the bearer token supplies it.
    """
    user_id, tenant_id = _resolve_scope(user_id, tenant_id)
    memory = _get_memory()
    result = await memory.retrieve_context(
        query_text=query_text,
        agent_id=validate_agent_id(agent_id),
        user_id=user_id,
        tenant_id=tenant_id,
        session_id=session_id or None,
    )
    return result.model_dump_json()


async def retrieve_memory(
    query_text: str,
    agent_id: str,
    user_id: str = "",
    session_id: str = "",
    tenant_id: str = DEFAULT_TENANT_ID,
    memory_intent: str = "continuity",
) -> str:
    """Retrieve compiled, evidence-backed memory from UMA.

    Returns a JSON MemoryResult: compiled_memory, facts (subject-predicate-
    object triples), evidence, provenance_valid. memory_intent is
    "continuity" (default) or "topical".
    """
    user_id, tenant_id = _resolve_scope(user_id, tenant_id)
    memory = _get_memory()
    result = await memory.retrieve_memory(
        query_text=query_text,
        agent_id=validate_agent_id(agent_id),
        user_id=user_id,
        tenant_id=tenant_id,
        session_id=session_id or None,
        memory_intent=memory_intent,
    )
    return result.model_dump_json()


async def process_turn(
    session_id: str,
    user_msg: str,
    assistant_reply: str,
    agent_id: str,
    user_id: str = "",
    tenant_id: str = DEFAULT_TENANT_ID,
) -> str:
    """Ingest a conversation turn into UMA memory.

    session_id is required (non-empty). Returns {"status": "ok", ...} on
    success or {"status": "injection_blocked", ...} if UMA's write-time
    scan rejected the user_msg with high severity — in that case nothing
    was stored.
    """
    user_id, tenant_id = _resolve_scope(user_id, tenant_id)
    from uma.adapters.scanner.injection_scan import InjectionDetectedError

    memory = _get_memory()
    try:
        await memory.process_turn(
            agent_id=validate_agent_id(agent_id),
            user_id=user_id,
            user_msg=user_msg,
            assistant_reply=assistant_reply,
            session_id=session_id,
            tenant_id=tenant_id,
        )
    except InjectionDetectedError as exc:
        return json.dumps(
            {
                "status": "injection_blocked",
                "severity": exc.severity,
                "matched_rules": list(exc.matched_rules),
                "score": exc.score,
                "user_id": user_id,
                "session_id": session_id,
            }
        )
    return json.dumps(
        {"status": "ok", "user_id": user_id, "session_id": session_id}
    )


async def ingest_document(
    file_path: str,
    agent_id: str,
    owner_type: str = "agent",
    owner_id: str = "",
    user_id: str = "",
    tenant_id: str = DEFAULT_TENANT_ID,
) -> str:
    """Ingest a document file into UMA's knowledge base.

    Chunks, embeds, and indexes the file. Returns a JSON IngestReport.

    Documents are scoped by (tenant_id, owner_type, owner_id), not by a
    request scope. owner_type is "agent" (default — the document joins the
    calling agent's shared KB) or "user" (the document is private to the
    ingesting user). owner_id defaults to the caller in both cases and may
    not name anyone else. In HTTP mode the bearer token supplies the user
    and the tenant.
    """
    resolved_agent_id = validate_agent_id(agent_id)
    # require_user=False: an agent-owned ingest has no user of its own, and
    # demanding one would make the common stdio call impossible.
    caller_user_id, tenant_id = _resolve_scope(
        user_id, tenant_id, require_user=False
    )
    resolved_owner_type, resolved_owner_id = _resolve_ingest_owner(
        agent_id=resolved_agent_id,
        owner_type=owner_type,
        owner_id=owner_id,
        caller_user_id=caller_user_id,
    )

    memory = _get_memory()
    report = await memory.ingest_document(
        file_path,
        owner_type=resolved_owner_type,
        owner_id=resolved_owner_id,
        tenant_id=tenant_id,
    )
    return json.dumps(dataclasses.asdict(report), default=str)


def health_check() -> str:
    """Return UMA runtime health status (HealthStatus JSON)."""
    memory = _get_memory()
    return memory.health_check().model_dump_json()


def _register_tools(server: FastMCP) -> None:
    """Register the five UMA tools on the given FastMCP instance.

    Same function bodies feed both the stdio and HTTP instances — the
    only difference between the two modes lives in ``_resolve_scope``,
    which reads the current AccessToken (or None) at call time.
    """
    server.tool()(retrieve_context)
    server.tool()(retrieve_memory)
    server.tool()(process_turn)
    server.tool()(ingest_document)
    server.tool()(health_check)


# ---------------------------------------------------------------------------
# Module-level stdio instance. HTTP mode builds its own instance in main().
# ---------------------------------------------------------------------------
mcp = FastMCP("uma")
_register_tools(mcp)


# ---------------------------------------------------------------------------
# CLI parsing & main()
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="uma-mcp",
        description=(
            "UMA Model Context Protocol server. Runs stdio by default; "
            "pass --http to serve streamable-http with bearer-token auth."
        ),
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Enable HTTP transport (streamable-http). Stdio is the default.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3131,
        help="Bind port for --http (default: 3131).",
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help=(
            "Bind interface for --http (default: 127.0.0.1). Set to 0.0.0.0 "
            "when serving through a tunnel."
        ),
    )
    parser.add_argument(
        "--public-url",
        default=None,
        help=(
            "Externally-reachable URL for --http mode. Required when --bind "
            "is not loopback. Used as the OAuth issuer/resource identifier."
        ),
    )
    parser.add_argument(
        "--tokens-db",
        type=Path,
        default=None,
        help=(
            "SQLite path for the bearer-token store "
            "(default: .uma/db/mcp_tokens.db). Mutually exclusive with "
            "--oauth-issuer."
        ),
    )
    parser.add_argument(
        "--oauth-issuer",
        default=None,
        help=(
            "OAuth 2.1 issuer URL. Enabling this switches from opaque "
            "bearer tokens (Phase 2) to JWT verification against your "
            "IdP (Auth0 / Entra / Google / Okta / any RFC 8414 provider). "
            "Required for ChatGPT."
        ),
    )
    parser.add_argument(
        "--oauth-audience",
        default=None,
        help=(
            "Expected JWT `aud` claim value — the resource server "
            "identifier registered with your IdP. Required with "
            "--oauth-issuer."
        ),
    )
    parser.add_argument(
        "--oauth-jwks-uri",
        default=None,
        help=(
            "JWKS endpoint URL. Defaults to "
            "{oauth-issuer}/.well-known/jwks.json — override if your IdP "
            "publishes JWKS at a non-standard path."
        ),
    )
    parser.add_argument(
        "--oauth-tenant",
        default=DEFAULT_TENANT_ID,
        help=(
            "UMA tenant every authenticated user maps to (default: "
            "'default'). v0.3.0 is single-tenant per OAuth deployment; "
            "multi-tenant claim mapping is deferred."
        ),
    )
    parser.add_argument(
        "--oauth-required-scope",
        action="append",
        default=None,
        dest="oauth_required_scopes",
        help=(
            "Require this scope in the JWT. Repeat for multiple. If "
            "omitted, no scope enforcement — any valid JWT passes."
        ),
    )
    return parser.parse_args(argv)


def _validate_http_args(args: argparse.Namespace) -> None:
    """Guard against the ECONNREFUSED trap (bind=0.0.0.0, no --public-url)
    plus OAuth flag consistency."""
    is_loopback = args.bind in ("127.0.0.1", "localhost", "::1")
    if not is_loopback and args.public_url is None:
        raise SystemExit(
            f"uma-mcp: --bind={args.bind} requires --public-url so the "
            "issuer and resource identifiers match the URL clients see. "
            "See docs/mcp/DEPLOY.md."
        )

    # OAuth flag consistency
    if args.oauth_issuer and not args.oauth_audience:
        raise SystemExit(
            "uma-mcp: --oauth-issuer requires --oauth-audience. See "
            "docs/mcp/CHATGPT.md."
        )
    if args.oauth_audience and not args.oauth_issuer:
        raise SystemExit(
            "uma-mcp: --oauth-audience without --oauth-issuer has no effect. "
            "Provide --oauth-issuer or drop --oauth-audience."
        )
    if args.oauth_issuer and args.tokens_db:
        raise SystemExit(
            "uma-mcp: --oauth-issuer and --tokens-db are mutually exclusive. "
            "OAuth mode validates JWTs from your IdP; bearer mode validates "
            "opaque tokens from the local store. Pick one."
        )


def _resolve_public_url(args: argparse.Namespace) -> str:
    if args.public_url:
        return args.public_url.rstrip("/")
    return f"http://{args.bind}:{args.port}"


def _build_http_server(args: argparse.Namespace) -> FastMCP:
    """Assemble a token-verifying FastMCP instance for HTTP mode.

    Two branches, selected by --oauth-issuer:

    OAuth JWT mode: UMAJWTVerifier validates JWTs from an external IdP.
    The MCP server acts as an OAuth 2.1 resource server; no local token
    store, no `uma auth create` needed. Required for ChatGPT.

    Bearer mode: UMATokenVerifier consults the local SQLite TokenStore.
    Tokens issued via `uma auth create`. For Perplexity, Cowork, and
    Claude Desktop's remote connector.
    """
    # Deferred imports so stdio mode never touches the auth code path.
    from pydantic import AnyHttpUrl
    from mcp.server.auth.settings import AuthSettings

    public_url = _resolve_public_url(args)
    resource_url = f"{public_url}/mcp"

    if args.oauth_issuer:
        # ---- OAuth JWT branch ----
        from uma.mcp.auth import UMAJWTVerifier

        verifier = UMAJWTVerifier(
            issuer=args.oauth_issuer,
            audience=args.oauth_audience,
            jwks_uri=args.oauth_jwks_uri,
            tenant_id=args.oauth_tenant,
            required_scopes=args.oauth_required_scopes,
        )
        # In OAuth mode, the issuer_url MUST be the external IdP — that's
        # what the SDK advertises in the RFC 9728 protected-resource
        # metadata and what clients discover. Do NOT point it at UMA
        # itself; UMA isn't an authorization server.
        auth = AuthSettings(
            issuer_url=AnyHttpUrl(args.oauth_issuer),
            resource_server_url=AnyHttpUrl(resource_url),
            required_scopes=(
                list(args.oauth_required_scopes)
                if args.oauth_required_scopes
                else None
            ),
        )
        logger.info(
            "uma-mcp http bind=%s port=%s public_url=%s mode=oauth "
            "issuer=%s audience=%s tenant=%s",
            args.bind,
            args.port,
            public_url,
            args.oauth_issuer,
            args.oauth_audience,
            args.oauth_tenant,
        )
    else:
        # ---- Opaque bearer branch (Phase 2 unchanged) ----
        from uma.mcp.auth import UMATokenVerifier
        from uma.mcp.tokens import DEFAULT_TOKENS_DB_PATH, TokenStore

        db_path = args.tokens_db if args.tokens_db else DEFAULT_TOKENS_DB_PATH
        store = TokenStore(db_path=db_path)
        # Init schema synchronously at server startup. asyncio.run isolates
        # the loop so it doesn't collide with mcp.run's own event loop.
        asyncio.run(store.init_schema())

        verifier = UMATokenVerifier(store)
        # In bearer mode, the issuer is UMA itself — the token was
        # locally minted, no external IdP.
        auth = AuthSettings(
            issuer_url=AnyHttpUrl(public_url),
            resource_server_url=AnyHttpUrl(resource_url),
        )
        logger.info(
            "uma-mcp http bind=%s port=%s public_url=%s mode=bearer "
            "tokens_db=%s",
            args.bind,
            args.port,
            public_url,
            db_path,
        )

    http_mcp = FastMCP(
        "uma",
        host=args.bind,
        port=args.port,
        token_verifier=verifier,
        auth=auth,
    )
    _register_tools(http_mcp)
    return http_mcp


def main(argv: Optional[list[str]] = None) -> None:
    """Console-script entry point.

    stdio (default): mcp.run() — blocks until the client disconnects.
    HTTP: build a token-verifying FastMCP, run streamable-http.
    """
    args = _parse_args(argv)

    if not args.http:
        # Stdio: use the module-level instance with the tools already on it.
        mcp.run()
        return

    _validate_http_args(args)
    http_mcp = _build_http_server(args)
    http_mcp.run(transport="streamable-http")


if __name__ == "__main__":  # pragma: no cover
    main()
