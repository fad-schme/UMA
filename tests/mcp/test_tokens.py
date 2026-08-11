"""Tests for uma.mcp.tokens.TokenStore.

Zero external dependencies — TokenStore is stdlib-only, so this suite runs
regardless of whether the `mcp` optional extra is installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uma.mcp.tokens import DEFAULT_TOKENS_DB_PATH, TokenRecord, TokenStore


@pytest.fixture
def store(tmp_path: Path) -> TokenStore:
    return TokenStore(db_path=tmp_path / "tokens.db")


@pytest.mark.asyncio
async def test_default_db_path_under_dot_uma():
    """Default path lives at .uma/db/mcp_tokens.db so it lands next to
    UMA's other SQLite artifacts and gets picked up by cleandir.py's
    --verify sweep like every other .uma-owned file."""
    assert DEFAULT_TOKENS_DB_PATH == Path(".uma/db/mcp_tokens.db")


@pytest.mark.asyncio
async def test_create_returns_raw_token_with_prefix(store: TokenStore):
    await store.init_schema()
    raw, record = await store.create(
        tenant_id="acme", user_id="alice", label="perplexity"
    )
    assert raw.startswith("umat_"), "raw token must carry the UMA prefix"
    assert len(raw) > 40, "raw token must have real entropy"
    assert record.tenant_id == "acme"
    assert record.user_id == "alice"
    assert record.label == "perplexity"
    assert record.revoked_at is None


@pytest.mark.asyncio
async def test_verify_matches_created_token(store: TokenStore):
    await store.init_schema()
    raw, record = await store.create(
        tenant_id="acme", user_id="alice", label="perplexity"
    )
    verified = await store.verify(raw)
    assert verified is not None
    assert verified.token_id == record.token_id
    assert verified.tenant_id == "acme"
    assert verified.user_id == "alice"


@pytest.mark.asyncio
async def test_verify_rejects_wrong_token(store: TokenStore):
    await store.init_schema()
    await store.create(tenant_id="acme", user_id="alice", label="perplexity")
    assert await store.verify("umat_wrongtoken") is None


@pytest.mark.asyncio
async def test_verify_rejects_wrong_prefix(store: TokenStore):
    """Early-reject anything that doesn't start with the UMA token prefix.
    Guards against treating arbitrary long strings as candidates."""
    await store.init_schema()
    assert await store.verify("bearer_xyz") is None
    assert await store.verify("some-other-token") is None


@pytest.mark.asyncio
async def test_verify_rejects_empty(store: TokenStore):
    await store.init_schema()
    assert await store.verify("") is None


@pytest.mark.asyncio
async def test_revoke_makes_token_unverifiable(store: TokenStore):
    await store.init_schema()
    raw, record = await store.create(
        tenant_id="acme", user_id="alice", label="perplexity"
    )
    assert await store.revoke(record.token_id) is True
    # Same raw token no longer verifies.
    assert await store.verify(raw) is None


@pytest.mark.asyncio
async def test_revoke_is_idempotent_returning_false_second_time(
    store: TokenStore,
):
    await store.init_schema()
    _, record = await store.create(
        tenant_id="acme", user_id="alice", label="perplexity"
    )
    assert await store.revoke(record.token_id) is True
    assert await store.revoke(record.token_id) is False


@pytest.mark.asyncio
async def test_list_excludes_revoked_by_default(store: TokenStore):
    await store.init_schema()
    _, r1 = await store.create(tenant_id="a", user_id="u", label="l1")
    _, r2 = await store.create(tenant_id="a", user_id="u", label="l2")
    await store.revoke(r1.token_id)
    active = await store.list()
    assert {t.token_id for t in active} == {r2.token_id}


@pytest.mark.asyncio
async def test_list_include_revoked_shows_everything(store: TokenStore):
    await store.init_schema()
    _, r1 = await store.create(tenant_id="a", user_id="u", label="l1")
    _, r2 = await store.create(tenant_id="a", user_id="u", label="l2")
    await store.revoke(r1.token_id)
    all_tokens = await store.list(include_revoked=True)
    assert {t.token_id for t in all_tokens} == {r1.token_id, r2.token_id}


@pytest.mark.asyncio
async def test_list_filters_by_tenant(store: TokenStore):
    """DAT enforcement in the CLI: `uma auth list --tenant X` MUST not
    surface tokens from other tenants, even to the operator."""
    await store.init_schema()
    _, a_token = await store.create(
        tenant_id="tenant-a", user_id="u", label="a"
    )
    _, b_token = await store.create(
        tenant_id="tenant-b", user_id="u", label="b"
    )
    a_only = await store.list(tenant_id="tenant-a")
    assert [t.token_id for t in a_only] == [a_token.token_id]
    b_only = await store.list(tenant_id="tenant-b")
    assert [t.token_id for t in b_only] == [b_token.token_id]


@pytest.mark.asyncio
async def test_create_rejects_empty_scope_values(store: TokenStore):
    """Empty tenant_id / user_id / label must be refused at the store
    boundary. Prevents tokens with unroutable scope from being minted."""
    await store.init_schema()
    for kwargs in (
        {"tenant_id": "", "user_id": "u", "label": "l"},
        {"tenant_id": "t", "user_id": "  ", "label": "l"},
        {"tenant_id": "t", "user_id": "u", "label": ""},
    ):
        with pytest.raises(ValueError):
            await store.create(**kwargs)


@pytest.mark.asyncio
async def test_plaintext_token_never_persisted(store: TokenStore, tmp_path):
    """The raw token must not appear in the SQLite file. Regression guard
    against any future 'let's just cache the token for convenience' PR."""
    import sqlite3

    await store.init_schema()
    raw, record = await store.create(
        tenant_id="acme", user_id="alice", label="perplexity"
    )
    with sqlite3.connect(store._db_path) as conn:  # noqa: SLF001
        rows = conn.execute(
            "SELECT token_id, token_hash, tenant_id, user_id, label FROM tokens"
        ).fetchall()
    for row in rows:
        # Every column value must not be the raw token
        for cell in row:
            assert cell != raw, (
                f"raw token found in DB column: {cell!r}"
            )
    # And the raw token must not appear anywhere in the DB file bytes
    # (belt-and-suspenders — catches accidental logging into aux tables).
    file_bytes = store._db_path.read_bytes()  # noqa: SLF001
    assert raw.encode("utf-8") not in file_bytes, (
        "raw token literal found in SQLite file bytes"
    )


@pytest.mark.asyncio
async def test_second_create_gets_distinct_token_id(store: TokenStore):
    """token_id collisions must never happen — different tokens must get
    different short ids."""
    await store.init_schema()
    _, r1 = await store.create(tenant_id="a", user_id="u", label="l1")
    _, r2 = await store.create(tenant_id="a", user_id="u", label="l2")
    assert r1.token_id != r2.token_id
