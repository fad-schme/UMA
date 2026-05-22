"""
test_pr7_lint_integration.py
==============================
PR7: lint_memory_drift routes typed lane records through verify_integrity.
Tampered records produce an integrity_failure finding and get quarantined.
"""
from __future__ import annotations

import json
import pytest

from tests.helpers.runtime import init_uma_for_tests

_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:pr7-lint")


# ---------------------------------------------------------------------------
# Helpers — direct SQL tampering without updating content_hash
# ---------------------------------------------------------------------------

def _tamper_fact(store, fact_id: str) -> None:
    conn = store._conn()
    try:
        conn.execute(
            "UPDATE facts SET object=? WHERE id=?",
            [json.dumps("tampered_value"), fact_id],
        )
        conn.commit()
    finally:
        conn.close()


def _tamper_episode(store, episode_id: str) -> None:
    conn = store._conn()
    try:
        conn.execute(
            "UPDATE episodes SET summary=? WHERE id=?",
            ["tampered summary", episode_id],
        )
        conn.commit()
    finally:
        conn.close()


def _tamper_skill(store, skill_id: str) -> None:
    conn = store._conn()
    try:
        conn.execute(
            "UPDATE skills SET plan=? WHERE id=?",
            [json.dumps({"steps": ["malicious step"]}), skill_id],
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lint_detects_tampered_fact(tmp_path):
    """lint_memory_drift reports integrity_failure when a fact has been tampered with."""
    from datetime import datetime, timezone
    from uma.common.types import Fact
    from uma.api.management import lint_memory_drift
    from uma.common.integrity import hash_fact_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["semantic"]

    clean = Fact(
        id="lint-fact-clean",
        subject="user:pr7-lint", predicate="LIKES", object="coffee",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7-lint", "LIKES", "coffee"),
        **_SCOPE,
    )
    tampered = Fact(
        id="lint-fact-tampered",
        subject="user:pr7-lint", predicate="OWNS", object="bike",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7-lint", "OWNS", "bike"),
        **_SCOPE,
    )
    await store.upsert_fact(clean, [0.1] * 64)
    await store.upsert_fact(tampered, [0.1] * 64)
    _tamper_fact(store, "lint-fact-tampered")

    result = await lint_memory_drift(
        memory,
        [clean, tampered],
        user_id="user:pr7-lint",
        tenant_id="default",
    )

    assert result["status"] == "issues_found"
    assert result["artifacts_scanned"] == 2
    failures = [f for f in result["findings"] if f["category"] == "integrity_failure"]
    assert len(failures) == 1
    assert failures[0]["record_id"] == "lint-fact-tampered"
    assert failures[0]["lane"] == "semantic"
    assert failures[0]["quarantined"] is True


@pytest.mark.asyncio
async def test_lint_clean_records_produce_no_findings(tmp_path):
    """lint_memory_drift returns status='ok' when no records have been tampered with."""
    from datetime import datetime, timezone
    from uma.common.types import Fact, Episode
    from uma.api.management import lint_memory_drift
    from uma.common.integrity import hash_fact_content, hash_episode_content

    memory = await init_uma_for_tests(tmp_path)
    sem_store = memory._stores["semantic"]
    epi_store = memory._stores["episodic"]

    fact = Fact(
        id="lint-clean-fact",
        subject="user:pr7-lint", predicate="READS", object="news",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7-lint", "READS", "news"),
        **_SCOPE,
    )
    ep = Episode(
        id="lint-clean-ep",
        timestamp=datetime.now(timezone.utc),
        summary="user went for a walk",
        user_id="user:pr7-lint",
        content_hash=hash_episode_content("user went for a walk"),
        **_SCOPE,
    )
    await sem_store.upsert_fact(fact, [0.1] * 64)
    await epi_store.add_episode(ep, [0.1] * 64)

    result = await lint_memory_drift(
        memory,
        [fact, ep],
        user_id="user:pr7-lint",
        tenant_id="default",
    )

    assert result["status"] == "ok"
    assert result["findings"] == []
    assert result["artifacts_scanned"] == 2


@pytest.mark.asyncio
async def test_lint_tampered_record_is_quarantined_after_lint(tmp_path):
    """After lint_memory_drift flags a tampered record, quarantined_at is set on that record."""
    from datetime import datetime, timezone
    from uma.common.types import Fact
    from uma.api.management import lint_memory_drift
    from uma.common.integrity import hash_fact_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["semantic"]

    fact = Fact(
        id="lint-quarantine-fact",
        subject="user:pr7-lint", predicate="WORKS_AT", object="company",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7-lint", "WORKS_AT", "company"),
        **_SCOPE,
    )
    await store.upsert_fact(fact, [0.1] * 64)
    _tamper_fact(store, "lint-quarantine-fact")

    await lint_memory_drift(
        memory,
        [fact],
        user_id="user:pr7-lint",
        tenant_id="default",
    )

    fetched = await store.get_fact("lint-quarantine-fact", **_SCOPE)
    assert fetched is not None
    assert fetched.quarantined_at is not None


@pytest.mark.asyncio
async def test_lint_mixed_lanes_detects_tampered_episode(tmp_path):
    """lint_memory_drift handles a mix of typed record types and pinpoints the tampered one."""
    from datetime import datetime, timezone
    from uma.common.types import Fact, Episode, Skill
    from uma.api.management import lint_memory_drift
    from uma.common.integrity import hash_fact_content, hash_episode_content, hash_skill_content

    memory = await init_uma_for_tests(tmp_path)
    sem_store = memory._stores["semantic"]
    epi_store = memory._stores["episodic"]
    pro_store = memory._stores["procedural"]

    fact = Fact(
        id="lint-mix-fact",
        subject="user:pr7-lint", predicate="USES", object="Python",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7-lint", "USES", "Python"),
        **_SCOPE,
    )
    ep = Episode(
        id="lint-mix-ep",
        timestamp=datetime.now(timezone.utc),
        summary="user attended a meetup",
        user_id="user:pr7-lint",
        content_hash=hash_episode_content("user attended a meetup"),
        **_SCOPE,
    )
    plan = {"steps": ["open", "process", "close"]}
    skill = Skill(
        id="lint-mix-skill",
        name="process_file",
        description="process a file",
        plan=plan,
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        content_hash=hash_skill_content("process_file", plan),
        **_SCOPE,
    )
    await sem_store.upsert_fact(fact, [0.1] * 64)
    await epi_store.add_episode(ep, [0.1] * 64)
    await pro_store.add_skill(skill, [0.1] * 64)

    _tamper_episode(epi_store, "lint-mix-ep")

    result = await lint_memory_drift(
        memory,
        [fact, ep, skill],
        user_id="user:pr7-lint",
        tenant_id="default",
    )

    assert result["status"] == "issues_found"
    assert result["artifacts_scanned"] == 3
    failures = [f for f in result["findings"] if f["category"] == "integrity_failure"]
    assert len(failures) == 1
    assert failures[0]["record_id"] == "lint-mix-ep"
    assert failures[0]["lane"] == "episodic"
