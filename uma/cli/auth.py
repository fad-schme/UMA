"""CLI handlers for ``uma auth {create,list,revoke}``.

Sync wrappers around ``uma.mcp.tokens.TokenStore`` — the token store is
async (to_thread wrapping around SQLite) but the CLI is sync, so each
handler uses ``asyncio.run``.

Auth operations do NOT load ``UMAConfig`` — the token store is
independent of the memory runtime and can be operated before UMA is
otherwise configured. This matches the convention that management-CLI
commands each pin their own scope explicitly.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from uma.mcp.tokens import DEFAULT_TOKENS_DB_PATH, TokenRecord, TokenStore


def _resolve_db_path(args: Any) -> Path:
    """--tokens-db argument, or the default under `.uma/db/`."""
    explicit = getattr(args, "tokens_db", None)
    return explicit if explicit else DEFAULT_TOKENS_DB_PATH


def _default_tenant() -> str:
    """Same fallback logic as `uma.cli.scopes._default_tenant`."""
    return os.environ.get("UMA_TENANT_ID", "default")


def _record_to_dict(record: TokenRecord) -> dict[str, Any]:
    return {
        "token_id": record.token_id,
        "tenant_id": record.tenant_id,
        "user_id": record.user_id,
        "label": record.label,
        "created_at": record.created_at,
        "revoked_at": record.revoked_at,
    }


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------
async def _create_async(
    *,
    db_path: Path,
    tenant_id: str,
    user_id: str,
    label: str,
) -> tuple[str, TokenRecord]:
    store = TokenStore(db_path=db_path)
    await store.init_schema()
    return await store.create(tenant_id=tenant_id, user_id=user_id, label=label)


def handle_create(args: Any) -> tuple[dict[str, Any], str, str, int]:
    """`uma auth create LABEL --user USER [--tenant TENANT] [--tokens-db PATH]`.

    Returns (data, text, status, exit_code) matching the ``_emit`` shape.
    The raw token is included in both the data payload and the text output
    — this is the ONLY time it exists in plaintext; every subsequent
    ``list`` returns just the metadata.
    """
    if not getattr(args, "user_id", None):
        return (
            {},
            "auth create requires --user",
            "error",
            2,
        )
    label = getattr(args, "label", None)
    if not label:
        return (
            {},
            "auth create requires a LABEL positional",
            "error",
            2,
        )

    tenant_id = getattr(args, "tenant_id", None) or _default_tenant()
    db_path = _resolve_db_path(args)

    raw_token, record = asyncio.run(
        _create_async(
            db_path=db_path,
            tenant_id=tenant_id,
            user_id=args.user_id,
            label=label,
        )
    )
    data = {
        **_record_to_dict(record),
        "token": raw_token,
        "tokens_db": str(db_path),
    }
    text = (
        f"Issued bearer token for tenant={record.tenant_id} "
        f"user={record.user_id} label={record.label}\n"
        f"  token_id: {record.token_id}\n"
        f"  token:    {raw_token}\n\n"
        "Store this token now — it is not recoverable. To revoke, run:\n"
        f"  uma auth revoke {record.token_id}"
    )
    return data, text, "ok", 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
async def _list_async(
    *,
    db_path: Path,
    tenant_id: str | None,
    include_revoked: bool,
) -> list[TokenRecord]:
    store = TokenStore(db_path=db_path)
    await store.init_schema()
    return await store.list(tenant_id=tenant_id, include_revoked=include_revoked)


def handle_list(args: Any) -> tuple[dict[str, Any], str, str, int]:
    """`uma auth list [--tenant TENANT] [--include-revoked] [--tokens-db PATH]`."""
    tenant_id = getattr(args, "tenant_id", None) or None
    include_revoked = bool(getattr(args, "include_revoked", False))
    db_path = _resolve_db_path(args)

    records = asyncio.run(
        _list_async(
            db_path=db_path,
            tenant_id=tenant_id,
            include_revoked=include_revoked,
        )
    )
    data = {
        "tokens": [_record_to_dict(r) for r in records],
        "count": len(records),
        "tokens_db": str(db_path),
    }
    if not records:
        text = f"No tokens{' (including revoked)' if include_revoked else ''}."
    else:
        header = (
            f"{'TOKEN_ID':14} {'TENANT':20} {'USER':20} "
            f"{'LABEL':20} {'CREATED_AT':20} STATUS"
        )
        rows = [header, "-" * len(header)]
        for r in records:
            status = "revoked" if r.revoked_at else "active"
            rows.append(
                f"{r.token_id:14} {r.tenant_id:20} {r.user_id:20} "
                f"{r.label:20} {r.created_at:20} {status}"
            )
        text = "\n".join(rows)
    return data, text, "ok", 0


# ---------------------------------------------------------------------------
# revoke
# ---------------------------------------------------------------------------
async def _revoke_async(*, db_path: Path, token_id: str) -> bool:
    store = TokenStore(db_path=db_path)
    await store.init_schema()
    return await store.revoke(token_id)


def handle_revoke(args: Any) -> tuple[dict[str, Any], str, str, int]:
    """`uma auth revoke TOKEN_ID [--tokens-db PATH]`."""
    token_id = getattr(args, "token_id", None)
    if not token_id:
        return (
            {},
            "auth revoke requires a TOKEN_ID positional",
            "error",
            2,
        )
    db_path = _resolve_db_path(args)
    revoked = asyncio.run(_revoke_async(db_path=db_path, token_id=token_id))
    data = {
        "token_id": token_id,
        "revoked": revoked,
        "tokens_db": str(db_path),
    }
    if revoked:
        text = f"Revoked token {token_id}."
        return data, text, "ok", 0
    text = (
        f"Token {token_id} not found or already revoked "
        f"(tokens_db={db_path})."
    )
    return data, text, "error", 1


__all__ = [
    "handle_create",
    "handle_list",
    "handle_revoke",
]
