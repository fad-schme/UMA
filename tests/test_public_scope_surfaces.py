from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import warnings

import pytest
import yaml

from uma.core.uma_memory import UMAMemory
from uma.core.retrieval.rlm.request import RetrievalRequest
from uma.types import RuntimeContext, TargetOwner, Skill

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
    memory.set_context(
        user_id="user:u1",
        agent_id="agent-default",
        tenant_id="default",
        request_id="test:agent-default",
    )
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


@pytest.mark.asyncio
async def test_public_procedural_reads_accept_explicit_workspace_scope_without_broadening_retrieval(tmp_path) -> None:
    memory = await _init_memory_with_procedural_feature(tmp_path)
    try:
        now = datetime.now(timezone.utc)
        skill = Skill(
            id="skill_workspace_owned",
            name="Workspace rollout",
            description="Run the shared workspace rollout procedure safely.",
            created_at=now,
            updated_at=now,
            trigger_phrases=["workspace rollout"],
            trigger_patterns=[],
            plan={"steps": ["rollout"]},
            tools=["shell"],
            example="workspace rollout",
            meta={},
        )
        embedding = (await memory.embedder.embed([skill.description]))[0]
        add_result = await memory.procedural_add_skill(
            skill,
            embedding,
            target_owner=TargetOwner(
                tenant_id="default",
                owner_type="workspace",
                owner_id="workspace:alpha",
                workspace_id="workspace:alpha",
            ),
        )
        assert add_result.ok

        find_result = await memory.procedural_find_skills(
            "workspace rollout",
            owner_type="workspace",
            owner_id="workspace:alpha",
            k=5,
        )
        assert find_result.ok
        assert [item.id for item in find_result.data] == ["skill_workspace_owned"]

        get_result = await memory.procedural_get_skill(
            "skill_workspace_owned",
            owner_type="workspace",
            owner_id="workspace:alpha",
        )
        assert get_result.ok
        assert get_result.data is not None
        assert get_result.data.owner_type == "workspace"

        request = RetrievalRequest.from_runtime_context(
            RuntimeContext(
                tenant_id="default",
                agent_id="agent-default",
                request_id="req-procedural-workspace",
                user_id="user:u1",
                session_id="legacy-user:user:u1",
            )
        )
        assert [scope.owner_type for scope in request.scopes] == ["agent", "user"]
    finally:
        memory.shutdown()


def test_agent_id_setter_is_removed_from_public_surface(uma_memory) -> None:
    with pytest.raises(AttributeError):
        uma_memory.agent_id = "agent-deprecated-test"

