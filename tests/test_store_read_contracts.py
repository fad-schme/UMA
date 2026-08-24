"""Store-level read contracts: LIKE escaping and quarantine exclusion.

Both are invariants the type system and the happy-path suite cannot catch:
a LIKE pattern that silently turns user text into a wildcard still returns
rows, and a read that forgets its quarantine filter still returns rows —
just the wrong ones.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.helpers.runtime import TEST_AGENT_ID, init_uma_for_tests
from uma.common.types import Episode
from uma.common.types.types_scope import DEFAULT_TENANT_ID
from uma.stores.base_sql_store import LIKE_ESCAPE_SQL, escape_like


USER = "user:u1"


# ── LIKE escaping ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("term", "matches", "does_not_match"),
    [
        ("100%", "100% cotton", "100 percent cotton"),
        ("a_b", "a_b", "axb"),
        (r"back\slash", r"back\slash", "backslash"),
    ],
)
def test_escaped_terms_match_literally_not_as_wildcards(
    term: str, matches: str, does_not_match: str
) -> None:
    """The escape only works because the clause carries ESCAPE.

    Without `LIKE_ESCAPE_SQL`, SQLite has no escape character at all: the
    backslashes become literal, `%` stays a wildcard, and `does_not_match`
    starts matching.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (text TEXT)")
    conn.executemany("INSERT INTO t VALUES (?)", [(matches,), (does_not_match,)])

    rows = conn.execute(
        f"SELECT text FROM t WHERE text LIKE ?{LIKE_ESCAPE_SQL}",
        [f"%{escape_like(term)}%"],
    ).fetchall()

    assert [row[0] for row in rows] == [matches]


def test_escape_like_leaves_ordinary_text_alone() -> None:
    assert escape_like("plain words") == "plain words"


@pytest.mark.parametrize("query", ["100% cotton", "a_b architecture", "match%everything"])
def test_query_terms_are_wildcard_free_before_reaching_a_like_clause(query: str) -> None:
    """Why the store-level escape is defence in depth, not a live fix.

    Both lexical paths build their terms through the shared keyword
    extractor, which strips punctuation — so no `%` or `_` reaches a LIKE
    pattern today. That is upstream behaviour the store cannot see, and it
    is the layer this asserts: if extraction ever stops sanitizing, the
    store's own ESCAPE clause is what keeps the pattern literal.
    """
    from uma.common.text import build_query_term_set, extract_keywords_and_phrases

    extracted = extract_keywords_and_phrases(query)
    term_set = build_query_term_set(query, max_terms=12, max_phrases=12)
    reaching_sql = [
        *(extracted.get("keywords") or []),
        *(extracted.get("keyphrases") or []),
        *(list(term_set.terms) if term_set else []),
        *(list(term_set.phrases) if term_set else []),
    ]

    assert reaching_sql, "extraction produced nothing; the assertion below would be vacuous"
    assert not any("%" in term or "_" in term for term in reaching_sql)


# ── quarantine exclusion ──────────────────────────────────────────────


def _episode(episode_id: str, *, quarantined: bool) -> Episode:
    now = datetime.now(timezone.utc)
    return Episode(
        id=episode_id,
        user_id=USER,
        timestamp=now,
        summary=f"summary for {episode_id}",
        raw=f"raw transcript for {episode_id}",
        tenant_id=DEFAULT_TENANT_ID,
        owner_type="user",
        owner_id=USER,
        origin_agent_id=TEST_AGENT_ID,
        quarantined_at=now if quarantined else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["fetch_summaries", "fetch_transcripts"])
async def test_episode_reads_exclude_quarantined_rows(tmp_path: Path, method: str) -> None:
    """Quarantine means "do not use this artifact for anything."

    These two reads take explicit ids, so a caller that already holds a
    quarantined id would otherwise pull its summary or full transcript back
    out — the sibling reads (list_recent, fetch_by_ids) all filter it.
    """
    memory = await init_uma_for_tests(tmp_path)
    try:
        store = memory.episodic_core.store
        clean = _episode("ep_clean", quarantined=False)
        dirty = _episode("ep_quarantined", quarantined=True)
        embedding = (await memory.embedder.embed(["episode text"]))[0]
        await store.add_episode(clean, embedding)
        await store.add_episode(dirty, embedding)

        rows = await getattr(store, method)(
            [clean.id, dirty.id],
            tenant_id=DEFAULT_TENANT_ID,
            owner_type="user",
            owner_id=USER,
        )

        assert [row["id"] for row in rows] == [clean.id]
    finally:
        memory.shutdown()
