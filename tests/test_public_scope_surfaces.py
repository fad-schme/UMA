from __future__ import annotations

from datetime import datetime, timezone
import warnings

import pytest
import yaml

from uma.core.uma_memory import UMAMemory
from uma.types import Skill

from tests.helpers.runtime import build_test_config


async def _init_memory_with_procedural_feature(tmp_path) -> UMAMemory:
    db_root = tmp_path / "db"
    db_root.mkdir(parents=True, exist_ok=True)

    cfg = build_test_config(db_root=db_root)
    cfg["features"] = {
        "load": [
            {
                "name": "procedural",
                "enabled": True,
                "provider": "uma.features.procedural.feature:ProceduralFeature",
                "config": {"max_k": 5},
            }
        ],
        "policy": {"on_attach_error": "log_and_skip", "allow_method_override": False},
    }

    cfg_path = tmp_path / "uma_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    memory = UMAMemory.from_yaml(str(cfg_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        memory.agent_id = "agent-default"
    memory._ensure_ingestion_ready()
    return memory


@pytest.mark.asyncio
async def test_public_procedural_reads_require_explicit_user_id(tmp_path) -> None:
    memory = await _init_memory_with_procedural_feature(tmp_path)
    try:
        now = datetime.now(timezone.utc)
        skill = Skill(
            id="skill_user_owned",
            name="Book a flight",
            description="Book a flight safely using the user-owned travel flow.",
            created_at=now,
            updated_at=now,
            owner_type="user",
            owner_id="user:u1",
            trigger_phrases=["book a flight"],
            trigger_patterns=[],
            plan={"steps": ["book"]},
            tools=["shell"],
            example="book a flight",
            meta={},
        )
        embedding = (await memory.embedder.embed([skill.description]))[0]
        add_result = await memory.procedural_add_skill(skill, embedding)
        assert add_result.ok

        find_result = await memory.procedural_find_skills(
            "book a flight",
            user_id="user:u1",
            k=5,
        )
        assert find_result.ok
        assert find_result.data
        assert find_result.data[0].id == "skill_user_owned"

        get_result = await memory.procedural_get_skill(
            "skill_user_owned",
            user_id="user:u1",
        )
        assert get_result.ok
        assert get_result.data is not None
        assert get_result.data.id == "skill_user_owned"
    finally:
        memory.shutdown()


@pytest.mark.asyncio
async def test_public_procedural_reads_no_longer_depend_on_ambient_memory_user_id(tmp_path) -> None:
    memory = await _init_memory_with_procedural_feature(tmp_path)
    try:
        result = await memory.procedural_find_skills("book a flight", user_id="", k=5)
        assert not result.ok
        assert "missing user_id" in result.errors
        assert not hasattr(memory, "user_id")
    finally:
        memory.shutdown()


def test_agent_id_setter_warns_as_deprecated_public_scope_api(uma_memory) -> None:
    with pytest.warns(DeprecationWarning, match="deprecated as a public scope API"):
        uma_memory.agent_id = "agent-deprecated-test"
