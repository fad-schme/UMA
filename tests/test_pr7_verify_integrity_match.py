"""
test_pr7_verify_integrity_match.py
====================================
PR7: verify_integrity returns "verified" for unmodified records (facts, episodes, skills).
"""
from __future__ import annotations

import pytest

from tests.helpers.runtime import init_uma_for_tests


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
