"""Bearer token store for the UMA MCP HTTP server.

Stores SHA-256 hashes of tokens, never plaintext. The raw token is shown
once at issue time (``TokenStore.create``) and never persisted. Verification
hashes the presented token and looks it up in constant time via
``hmac.compare_digest``.

Schema (`.uma/db/mcp_tokens.db` by default, override with ``--tokens-db``):

    CREATE TABLE tokens (
        token_id     TEXT PRIMARY KEY,   -- server-generated short id
        token_hash   TEXT NOT NULL,      -- sha256 hex of the raw token
        tenant_id    TEXT NOT NULL,      -- DAT invariant carrier
        user_id      TEXT NOT NULL,      -- DAT invariant carrier
        label        TEXT NOT NULL,      -- caller-supplied ("perplexity", ...)
        created_at   TEXT NOT NULL,      -- ISO-8601 UTC
        revoked_at   TEXT                -- NULL while active
    )

This module has zero external dependencies (stdlib only) so the CLI can
use it without the ``mcp`` optional extra installed. The MCP SDK's
``TokenVerifier`` subclass lives in ``uma.mcp.auth``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Default DB path relative to CWD, matching UMA's `.uma/db/` convention.
DEFAULT_TOKENS_DB_PATH = Path(".uma/db/mcp_tokens.db")

# Raw token format: `umat_<44 url-safe bytes>` — `umat_` prefix helps
# operators recognize a UMA MCP token in logs; 44 url-safe bytes gives
# 264 bits of entropy which is well past cryptographic sufficiency.
_TOKEN_PREFIX = "umat_"


@dataclass(frozen=True)
class TokenRecord:
    """One row from the tokens table. Never carries the raw token."""

    token_id: str
    tenant_id: str
    user_id: str
    label: str
    created_at: str
    revoked_at: Optional[str] = None

    # token_hash is intentionally not surfaced — no code path outside
    # TokenStore should compare hashes directly. Prevents callers from
    # accidentally short-circuiting the constant-time compare.
    _internal_hash: str = field(default="", repr=False)


def _hash_token(raw: str) -> str:
    """Hex SHA-256 of the raw token. Matches UMA's sha256-only convention."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TokenStore:
    """SQLite-backed bearer token store.

    Every public method is async and wraps the sync sqlite3 body in
    ``asyncio.to_thread`` — matches the pattern used by every other UMA
    store (finding #1 P0). No event-loop blocking.

    Not thread-safe for concurrent writes to the same db_path; SQLite's
    own write serialization is relied on. Reads are safe.
    """

    def __init__(self, db_path: Path | str = DEFAULT_TOKENS_DB_PATH) -> None:
        self._db_path = Path(db_path)
        # Ensure the directory exists; SQLite will create the file itself.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_schema_sync(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tokens (
                    token_id    TEXT PRIMARY KEY,
                    token_hash  TEXT NOT NULL UNIQUE,
                    tenant_id   TEXT NOT NULL,
                    user_id     TEXT NOT NULL,
                    label       TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    revoked_at  TEXT
                )
                """
            )
            conn.commit()

    async def init_schema(self) -> None:
        await asyncio.to_thread(self._init_schema_sync)

    # ------------------------------------------------------------------
    # Create — returns the raw token string exactly once
    # ------------------------------------------------------------------
    def _create_sync(
        self,
        *,
        tenant_id: str,
        user_id: str,
        label: str,
    ) -> tuple[str, TokenRecord]:
        if not tenant_id.strip() or not user_id.strip() or not label.strip():
            raise ValueError(
                "tenant_id, user_id, and label must all be non-empty"
            )

        token_id = secrets.token_urlsafe(9)  # 12-char short id
        raw_token = _TOKEN_PREFIX + secrets.token_urlsafe(32)
        token_hash = _hash_token(raw_token)
        created_at = _now_utc_iso()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO tokens
                    (token_id, token_hash, tenant_id, user_id, label,
                     created_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    token_id,
                    token_hash,
                    tenant_id,
                    user_id,
                    label,
                    created_at,
                ),
            )
            conn.commit()

        record = TokenRecord(
            token_id=token_id,
            tenant_id=tenant_id,
            user_id=user_id,
            label=label,
            created_at=created_at,
            revoked_at=None,
            _internal_hash=token_hash,
        )
        return raw_token, record

    async def create(
        self,
        *,
        tenant_id: str,
        user_id: str,
        label: str,
    ) -> tuple[str, TokenRecord]:
        """Issue a new bearer token.

        Returns ``(raw_token, record)``. The raw_token is the only chance
        the caller has to see the plaintext — the DB stores only its
        hash. Log the token_id, not the raw_token.
        """
        return await asyncio.to_thread(
            self._create_sync,
            tenant_id=tenant_id,
            user_id=user_id,
            label=label,
        )

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------
    def _list_sync(
        self,
        *,
        tenant_id: Optional[str],
        include_revoked: bool,
    ) -> list[TokenRecord]:
        query = (
            "SELECT token_id, tenant_id, user_id, label, created_at, "
            "revoked_at FROM tokens"
        )
        params: list[str] = []
        conditions: list[str] = []
        if tenant_id is not None:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if not include_revoked:
            conditions.append("revoked_at IS NULL")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            TokenRecord(
                token_id=row[0],
                tenant_id=row[1],
                user_id=row[2],
                label=row[3],
                created_at=row[4],
                revoked_at=row[5],
            )
            for row in rows
        ]

    async def list(
        self,
        *,
        tenant_id: Optional[str] = None,
        include_revoked: bool = False,
    ) -> list[TokenRecord]:
        """List tokens, most recent first. Excludes revoked by default."""
        return await asyncio.to_thread(
            self._list_sync,
            tenant_id=tenant_id,
            include_revoked=include_revoked,
        )

    # ------------------------------------------------------------------
    # Revoke
    # ------------------------------------------------------------------
    def _revoke_sync(self, token_id: str) -> bool:
        revoked_at = _now_utc_iso()
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                """
                UPDATE tokens
                   SET revoked_at = ?
                 WHERE token_id = ?
                   AND revoked_at IS NULL
                """,
                (revoked_at, token_id),
            )
            conn.commit()
            return cur.rowcount > 0

    async def revoke(self, token_id: str) -> bool:
        """Mark a token revoked. Returns True if a row was affected."""
        return await asyncio.to_thread(self._revoke_sync, token_id)

    # ------------------------------------------------------------------
    # Verify — the hot path called on every MCP request
    # ------------------------------------------------------------------
    def _verify_sync(self, raw_token: str) -> Optional[TokenRecord]:
        if not raw_token or not raw_token.startswith(_TOKEN_PREFIX):
            # Wrong shape — reject early without touching the DB. Guards
            # against accidentally treating any long string as a candidate.
            return None

        presented_hash = _hash_token(raw_token)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT token_id, token_hash, tenant_id, user_id, label,
                       created_at, revoked_at
                  FROM tokens
                 WHERE revoked_at IS NULL
                """
            ).fetchall()

        # Constant-time compare against every active hash. O(n) in the
        # number of active tokens; for v0.2.0's operator-scale token
        # counts (dozens, not millions) this is fine and dodges the
        # timing side-channel that a WHERE token_hash = ? lookup would
        # have. Revisit with an indexed lookup only if benchmarks show
        # this on the hot path.
        for row in rows:
            stored_hash = row[1]
            if hmac.compare_digest(presented_hash, stored_hash):
                return TokenRecord(
                    token_id=row[0],
                    tenant_id=row[2],
                    user_id=row[3],
                    label=row[4],
                    created_at=row[5],
                    revoked_at=row[6],
                    _internal_hash=stored_hash,
                )
        return None

    async def verify(self, raw_token: str) -> Optional[TokenRecord]:
        """Return the record for a valid, non-revoked token, or None."""
        return await asyncio.to_thread(self._verify_sync, raw_token)


__all__ = [
    "DEFAULT_TOKENS_DB_PATH",
    "TokenRecord",
    "TokenStore",
]
