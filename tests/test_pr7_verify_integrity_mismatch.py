"""
test_pr7_verify_integrity_mismatch.py
=======================================
PR7: verify_integrity detects content tampering, quarantines the record,
and surfaces it via list_quarantined.
"""
from __future__ import annotations

import json
import pytest

from tests.helpers.runtime import init_uma_for_tests

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
