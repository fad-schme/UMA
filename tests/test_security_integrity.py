"""Content integrity verification: SHA-256 re-derivation, tamper detection, auto-quarantine on mismatch.

Covers verify_integrity (match and mismatch paths) across Fact/Episode/Skill,
audit log entries on failure, and lint_memory_drift integration.
"""
from __future__ import annotations
from tests.helpers.runtime import init_uma_for_tests
import json
import pytest

# ── test_pr7_verify_integrity_match ──────────────────────────────────────────





_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:pr7")


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_integrity_fact_match(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Fact
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_fact_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["semantic"]

    fact = Fact(
        id="fact-iv-1",
        subject="user:pr7",
        predicate="LIKES",
        object="hiking",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7", "LIKES", "hiking"),
        **_SCOPE,
    )
    await store.upsert_fact(fact, [0.1] * 64)

    result = await verify_integrity(memory, record_id="fact-iv-1", lane="semantic", **_SCOPE)

    assert result.status == "verified"
    assert result.record_id == "fact-iv-1"
    assert result.lane == "semantic"
    assert result.expected_hash is None
    assert result.actual_hash is None
    assert result.quarantined is False


@pytest.mark.asyncio
async def test_verify_integrity_fact_unchanged_after_verification(tmp_path):
    """verify_integrity must not mutate a record that passes."""
    from datetime import datetime, timezone
    from uma.common.types import Fact
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_fact_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["semantic"]

    fact = Fact(
        id="fact-iv-2",
        subject="user:pr7",
        predicate="KNOWS",
        object="Python",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7", "KNOWS", "Python"),
        **_SCOPE,
    )
    await store.upsert_fact(fact, [0.1] * 64)

    await verify_integrity(memory, record_id="fact-iv-2", lane="semantic", **_SCOPE)

    fetched = await store.get_fact("fact-iv-2", **_SCOPE)
    assert fetched is not None
    assert fetched.quarantined_at is None


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_integrity_episode_match(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Episode
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_episode_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["episodic"]

    ep = Episode(
        id="ep-iv-1",
        timestamp=datetime.now(timezone.utc),
        summary="user went hiking in the mountains",
        user_id="user:pr7",
        content_hash=hash_episode_content("user went hiking in the mountains"),
        **_SCOPE,
    )
    await store.add_episode(ep, [0.1] * 64)

    result = await verify_integrity(memory, record_id="ep-iv-1", lane="episodic", **_SCOPE)

    assert result.status == "verified"
    assert result.quarantined is False


@pytest.mark.asyncio
async def test_verify_integrity_episode_not_mutated(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Episode
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_episode_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["episodic"]

    ep = Episode(
        id="ep-iv-2",
        timestamp=datetime.now(timezone.utc),
        summary="user read a book",
        user_id="user:pr7",
        content_hash=hash_episode_content("user read a book"),
        **_SCOPE,
    )
    await store.add_episode(ep, [0.1] * 64)

    await verify_integrity(memory, record_id="ep-iv-2", lane="episodic", **_SCOPE)

    fetched = await store.get_episode("ep-iv-2", **_SCOPE)
    assert fetched.quarantined_at is None


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_integrity_skill_match(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Skill
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_skill_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["procedural"]

    plan = {"steps": ["open file", "read file", "close file"]}
    skill = Skill(
        id="skill-iv-1",
        name="read_file",
        description="how to read a file",
        plan=plan,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        content_hash=hash_skill_content("read_file", plan),
        **_SCOPE,
    )
    await store.add_skill(skill, [0.1] * 64)

    result = await verify_integrity(memory, record_id="skill-iv-1", lane="procedural", **_SCOPE)

    assert result.status == "verified"
    assert result.quarantined is False


@pytest.mark.asyncio
async def test_verify_integrity_unknown_lane_raises(tmp_path):
    from uma.api.management import verify_integrity

    memory = await init_uma_for_tests(tmp_path)

    with pytest.raises(ValueError, match="unknown lane"):
        await verify_integrity(memory, record_id="x", lane="unknown", **_SCOPE)


@pytest.mark.asyncio
async def test_verify_integrity_missing_record_raises(tmp_path):
    from uma.api.management import verify_integrity

    memory = await init_uma_for_tests(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        await verify_integrity(memory, record_id="does-not-exist", lane="semantic", **_SCOPE)


# ── test_pr7_verify_integrity_mismatch ──────────────────────────────────────────




_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:pr7-mm")


def _tamper_fact(store, fact_id: str) -> None:
    """Directly overwrite the fact object in SQL without updating content_hash."""
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
    """Directly overwrite the episode summary in SQL without updating content_hash."""
    conn = store._conn()
    try:
        conn.execute(
            "UPDATE episodes SET summary=? WHERE id=?",
            ["tampered summary text", episode_id],
        )
        conn.commit()
    finally:
        conn.close()


def _tamper_skill(store, skill_id: str) -> None:
    """Directly overwrite the skill plan in SQL without updating content_hash."""
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
# Fact mismatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tampered_fact_fails_verification(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Fact
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_fact_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["semantic"]

    original_object = "hiking"
    fact = Fact(
        id="fact-tamper-1",
        subject="user:pr7-mm",
        predicate="LIKES",
        object=original_object,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7-mm", "LIKES", original_object),
        **_SCOPE,
    )
    await store.upsert_fact(fact, [0.1] * 64)

    _tamper_fact(store, "fact-tamper-1")

    result = await verify_integrity(
        memory, record_id="fact-tamper-1", lane="semantic", **_SCOPE
    )

    assert result.status == "failed"
    assert result.record_id == "fact-tamper-1"
    assert result.lane == "semantic"
    assert result.expected_hash == hash_fact_content("user:pr7-mm", "LIKES", original_object)
    assert result.actual_hash != result.expected_hash
    assert result.quarantined is True


@pytest.mark.asyncio
async def test_tampered_fact_is_quarantined_in_store(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Fact
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_fact_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["semantic"]

    fact = Fact(
        id="fact-tamper-2",
        subject="user:pr7-mm",
        predicate="OWNS",
        object="car",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7-mm", "OWNS", "car"),
        **_SCOPE,
    )
    await store.upsert_fact(fact, [0.1] * 64)
    _tamper_fact(store, "fact-tamper-2")

    await verify_integrity(memory, record_id="fact-tamper-2", lane="semantic", **_SCOPE)

    # Record must now have quarantined_at set
    fetched = await store.get_fact("fact-tamper-2", **_SCOPE)
    assert fetched is not None
    assert fetched.quarantined_at is not None


@pytest.mark.asyncio
async def test_tampered_fact_has_audit_log_entry(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Fact
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_fact_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["semantic"]

    fact = Fact(
        id="fact-tamper-3",
        subject="user:pr7-mm",
        predicate="READS",
        object="books",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7-mm", "READS", "books"),
        **_SCOPE,
    )
    await store.upsert_fact(fact, [0.1] * 64)
    _tamper_fact(store, "fact-tamper-3")

    await verify_integrity(memory, record_id="fact-tamper-3", lane="semantic", **_SCOPE)

    fetched = await store.get_fact("fact-tamper-3", **_SCOPE)
    audit_log = (fetched.meta or {}).get("security", {}).get("audit_log", [])
    assert any(e.get("event") == "integrity_failure" for e in audit_log), (
        "audit_log must contain an integrity_failure entry"
    )


@pytest.mark.asyncio
async def test_tampered_fact_appears_in_list_quarantined(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Fact
    from uma.api.management import verify_integrity, list_quarantined
    from uma.common.integrity import hash_fact_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["semantic"]

    fact = Fact(
        id="fact-tamper-4",
        subject="user:pr7-mm",
        predicate="WORKS_AT",
        object="company",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        content_hash=hash_fact_content("user:pr7-mm", "WORKS_AT", "company"),
        **_SCOPE,
    )
    await store.upsert_fact(fact, [0.1] * 64)
    _tamper_fact(store, "fact-tamper-4")

    await verify_integrity(memory, record_id="fact-tamper-4", lane="semantic", **_SCOPE)

    quarantined = await list_quarantined(
        memory, lane="semantic", **_SCOPE
    )
    ids = [r.id for r in quarantined]
    assert "fact-tamper-4" in ids


# ---------------------------------------------------------------------------
# Episode mismatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tampered_episode_fails_verification(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Episode
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_episode_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["episodic"]

    original_summary = "user went hiking"
    ep = Episode(
        id="ep-tamper-1",
        timestamp=datetime.now(timezone.utc),
        summary=original_summary,
        user_id="user:pr7-mm",
        content_hash=hash_episode_content(original_summary),
        **_SCOPE,
    )
    await store.add_episode(ep, [0.1] * 64)
    _tamper_episode(store, "ep-tamper-1")

    result = await verify_integrity(
        memory, record_id="ep-tamper-1", lane="episodic", **_SCOPE
    )

    assert result.status == "failed"
    assert result.quarantined is True
    assert result.expected_hash == hash_episode_content(original_summary)
    assert result.actual_hash != result.expected_hash


@pytest.mark.asyncio
async def test_tampered_episode_audit_log(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Episode
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_episode_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["episodic"]

    ep = Episode(
        id="ep-tamper-2",
        timestamp=datetime.now(timezone.utc),
        summary="user read a book",
        user_id="user:pr7-mm",
        content_hash=hash_episode_content("user read a book"),
        **_SCOPE,
    )
    await store.add_episode(ep, [0.1] * 64)
    _tamper_episode(store, "ep-tamper-2")

    await verify_integrity(memory, record_id="ep-tamper-2", lane="episodic", **_SCOPE)

    fetched = await store.get_episode("ep-tamper-2", **_SCOPE)
    audit_log = (fetched.meta or {}).get("security", {}).get("audit_log", [])
    assert any(e.get("event") == "integrity_failure" for e in audit_log)


# ---------------------------------------------------------------------------
# Skill mismatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tampered_skill_fails_verification(tmp_path):
    from datetime import datetime, timezone
    from uma.common.types import Skill
    from uma.api.management import verify_integrity
    from uma.common.integrity import hash_skill_content

    memory = await init_uma_for_tests(tmp_path)
    store = memory._stores["procedural"]

    original_plan = {"steps": ["step1", "step2"]}
    skill = Skill(
        id="skill-tamper-1",
        name="do_thing",
        description="a test skill",
        plan=original_plan,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        content_hash=hash_skill_content("do_thing", original_plan),
        **_SCOPE,
    )
    await store.add_skill(skill, [0.1] * 64)
    _tamper_skill(store, "skill-tamper-1")

    result = await verify_integrity(
        memory, record_id="skill-tamper-1", lane="procedural", **_SCOPE
    )

    assert result.status == "failed"
    assert result.quarantined is True
    assert result.expected_hash == hash_skill_content("do_thing", original_plan)


# ── test_pr7_lint_integration ──────────────────────────────────────────




_SCOPE = dict(tenant_id="default", owner_type="user", owner_id="user:pr7-lint")


# ---------------------------------------------------------------------------
# Helpers — direct SQL tampering without updating content_hash
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
