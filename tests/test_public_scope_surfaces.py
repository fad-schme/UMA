from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import warnings

import pytest
import yaml

from uma.core.uma_memory import UMAMemory
from uma.core.retrieval.rlm.request import RetrievalRequest
from uma.types import TargetOwner, Skill

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
            memory._build_runtime_context_for_retrieval(user_id="user:u1")
        )
        assert [scope.owner_type for scope in request.scopes] == ["agent", "user"]
    finally:
        memory.shutdown()


def test_agent_id_setter_warns_as_deprecated_public_scope_api(uma_memory) -> None:
    with pytest.warns(DeprecationWarning, match="deprecated as a public scope API"):
        uma_memory.agent_id = "agent-deprecated-test"


@pytest.mark.asyncio
async def test_process_turn_shim_warns_and_routes_to_canonical_pipeline(uma_memory) -> None:
    calls = []

    class _Pipeline:
        async def process_turn(self, *, user_id, user_msg, assistant_reply, extra_meta=None):
            calls.append(
                {
                    "user_id": user_id,
                    "user_msg": user_msg,
                    "assistant_reply": assistant_reply,
                    "extra_meta": extra_meta,
                }
            )

    uma_memory.pipeline = _Pipeline()
    uma_memory._ingestion_ready = True

    with pytest.warns(DeprecationWarning, match="process_turn"):
        await uma_memory.process_turn(
            user_id="u1",
            user_msg="hello",
            assistant_reply="world",
            extra_meta={"session_id": "session-a"},
        )

    assert calls == [
        {
            "user_id": "user:u1",
            "user_msg": "hello",
            "assistant_reply": "world",
            "extra_meta": {"session_id": "session-a"},
        }
    ]


@pytest.mark.asyncio
async def test_ingest_document_shim_warns_and_routes_legacy_owner_shape_to_target_owner(uma_memory, monkeypatch, tmp_path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("hello world", encoding="utf-8")
    seen = {}

    async def fake_ingest(file_path, *, target_owner=None, owner_type=None, owner_id=None, config=None, memory=None):
        seen.update(
            {
                "file_path": file_path,
                "target_owner": target_owner,
                "owner_type": owner_type,
                "owner_id": owner_id,
                "memory": memory,
            }
        )
        return SimpleNamespace(doc_id="doc-1")

    monkeypatch.setattr("uma.core.ingest.ingest_service.ingest_document", fake_ingest)
    uma_memory._ingestion_ready = True

    with pytest.warns(DeprecationWarning) as caught:
        await uma_memory.ingest_document(str(path), owner_type="user", owner_id="u1")

    messages = [str(item.message) for item in caught]
    assert any("ingest_document is deprecated" in message for message in messages)
    assert any("ingest_document(owner_type, owner_id)" in message for message in messages)
    assert seen["file_path"] == str(path)
    assert seen["owner_type"] == "user"
    assert seen["owner_id"] == "u1"
    assert seen["target_owner"] is None
    assert seen["memory"] is uma_memory


@pytest.mark.asyncio
async def test_rebuild_shims_warn_and_route_to_canonical_maintenance_helpers(uma_memory, monkeypatch) -> None:
    seen = []

    async def fake_rebuild_vector_indexes(memory, **kwargs):
        seen.append(("vector", memory, kwargs))
        return {"status": "ok", "report": {}}

    async def fake_rebuild_derived_indexes(memory, **kwargs):
        seen.append(("derived", memory, kwargs))
        return {"status": "ok", "vector": {}, "graph": {}}

    monkeypatch.setattr("uma.core.utils.maintenance.rebuild_vector_indexes", fake_rebuild_vector_indexes)
    monkeypatch.setattr("uma.core.utils.maintenance.rebuild_derived_indexes", fake_rebuild_derived_indexes)

    with pytest.warns(DeprecationWarning, match="rebuild_vector_indexes"):
        vector_result = await uma_memory.rebuild_vector_indexes(owner_type="user", owner_id="user:u1")
    with pytest.warns(DeprecationWarning, match="rebuild_derived_indexes"):
        derived_result = await uma_memory.rebuild_derived_indexes(owner_type="user", owner_id="user:u1")

    assert vector_result["status"] == "ok"
    assert derived_result["status"] == "ok"
    assert seen[0] == ("vector", uma_memory, {"owner_type": "user", "owner_id": "user:u1", "include_episodic": True, "include_semantic": True, "include_procedural": True, "batch_size": 32})
    assert seen[1] == ("derived", uma_memory, {"owner_type": "user", "owner_id": "user:u1", "include_episodic": True, "include_semantic": True, "include_procedural": True, "include_graph": True, "batch_size": 32})
