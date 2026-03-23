from __future__ import annotations

from datetime import datetime, timezone

import pytest

from uma.core.ingest.ingest_service import _coerce_ingest_target_owner
from uma.types import OwnershipRef, Skill, TargetOwner


def test_ingest_target_owner_accepts_agent_user_and_workspace() -> None:
    agent_owner = _coerce_ingest_target_owner("agent", "agent:alpha")
    assert agent_owner.owner_type == "agent"
    assert agent_owner.owner_id == "agent:alpha"

    user_owner = _coerce_ingest_target_owner("user", "u1")
    assert user_owner.owner_type == "user"
    assert user_owner.owner_id == "user:u1"
    assert user_owner.workspace_id is None

    workspace_owner = _coerce_ingest_target_owner("workspace", "workspace:alpha")
    assert workspace_owner.owner_type == "workspace"
    assert workspace_owner.owner_id == "workspace:alpha"
    assert workspace_owner.workspace_id == "workspace:alpha"


@pytest.mark.parametrize("owner_type", ["system"])
def test_ingest_target_owner_rejects_unsupported_owner_types(owner_type: str) -> None:
    with pytest.raises(ValueError, match="owner_type"):
        _coerce_ingest_target_owner(owner_type, f"{owner_type}:alpha")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_owner", "expected_owner_type", "expected_owner_id", "expected_workspace_id"),
    [
        (TargetOwner(tenant_id="default", owner_type="user", owner_id="user:u1"), "user", "user:u1", None),
        (TargetOwner(tenant_id="default", owner_type="agent", owner_id="agent:alpha"), "agent", "agent:alpha", None),
        (
            TargetOwner(
                tenant_id="default",
                owner_type="workspace",
                owner_id="workspace:alpha",
                workspace_id="workspace:alpha",
            ),
            "workspace",
            "workspace:alpha",
            "workspace:alpha",
        ),
    ],
)
async def test_ingest_document_persists_explicit_target_owner(
    uma_memory,
    tmp_path,
    target_owner: TargetOwner,
    expected_owner_type: str,
    expected_owner_id: str,
    expected_workspace_id: str | None,
) -> None:
    memory = uma_memory
    path = tmp_path / f"{expected_owner_type}-target-doc.txt"
    path.write_text(
        "UMA explicit ingest target owner test. "
        "This paragraph is deliberately long enough to survive chunking and fact extraction without "
        "falling below the extractor threshold. It includes additional concrete statements about a "
        "workspace playbook, operational checklist, ownership marker, and validation trail so the "
        "document-derived fact path is exercised reliably during ingestion.\n"
    )

    report = await memory.ingest_document(str(path), target_owner=target_owner)
    assert report.doc_id

    conn = memory.document_store._conn()
    try:
        rows = memory.document_store._query_all(
            conn,
            """
            SELECT owner_type, owner_id, tenant_id, workspace_id
            FROM documents
            WHERE doc_id=?
            """,
            params=[report.doc_id],
            log_context="test_target_owner_ingest_target_owner",
        )
        assert rows
        assert rows[0]["tenant_id"] == "default"
        assert rows[0]["owner_type"] == expected_owner_type
        assert rows[0]["owner_id"] == expected_owner_id
        assert rows[0]["workspace_id"] == expected_workspace_id
    finally:
        conn.close()

    if expected_owner_type == "workspace":
        conn = memory.semantic_core.store._conn()
        try:
            fact_rows = memory.semantic_core.store._query_all(
                conn,
                """
                SELECT owner_type, owner_id, workspace_id
                FROM facts
                WHERE owner_type=? AND owner_id=? AND meta LIKE ?
                """,
                params=["workspace", "workspace:alpha", f"%{report.doc_id}%"],
                log_context="test_target_owner_ingest_workspace_facts",
            )
            assert fact_rows
            assert all(row["owner_type"] == "workspace" for row in fact_rows)
            assert all(row["owner_id"] == "workspace:alpha" for row in fact_rows)
            assert all(row["workspace_id"] == "workspace:alpha" for row in fact_rows)
        finally:
            conn.close()


@pytest.mark.asyncio
async def test_legacy_document_ingest_boundary_adapter_still_works(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "owner-contract-doc.txt"
    path.write_text(
        "UMA document ingestion contract test. "
        "This passage is intentionally long enough to produce a valid chunk and fact extraction path.\n"
    )

    report = await memory.ingest_document(str(path), owner_type="user", owner_id="u1")
    assert report.doc_id

    # Query directly to avoid changing the public read surface in this PR.
    conn = memory.document_store._conn()
    try:
        rows = memory.document_store._query_all(
            conn,
            "SELECT owner_type, owner_id, tenant_id FROM documents WHERE owner_type=? AND owner_id=? ORDER BY ingested_at DESC LIMIT 1",
            params=["user", "user:u1"],
            log_context="test_target_owner_ingest_manifest",
        )
        assert rows
        assert rows[0]["owner_type"] == "user"
        assert rows[0]["owner_id"] == "user:u1"
        assert rows[0]["tenant_id"] == "default"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_document_ingest_rejects_system_target_owner(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "system-target-doc.txt"
    path.write_text(
        "UMA unsupported system ingest target test. "
        "This paragraph is intentionally long enough to exercise the validation path.\n"
    )

    with pytest.raises(ValueError, match="owner_type"):
        await memory.ingest_document(
            str(path),
            target_owner=TargetOwner(
                tenant_id="default",
                owner_type="system",
                owner_id="system:ops",
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_owner", "expected_owner_type", "expected_owner_id"),
    [
        (TargetOwner(tenant_id="tenant-1", owner_type="user", owner_id="user:u1"), "user", "user:u1"),
        (TargetOwner(tenant_id="tenant-1", owner_type="agent", owner_id="agent:alpha"), "agent", "agent:alpha"),
        (
            TargetOwner(
                tenant_id="tenant-1",
                owner_type="workspace",
                owner_id="workspace:alpha",
                workspace_id="workspace:alpha",
            ),
            "workspace",
            "workspace:alpha",
        ),
    ],
)
async def test_procedural_add_skill_for_owner_persists_explicit_target_owner(
    uma_memory,
    target_owner: TargetOwner,
    expected_owner_type: str,
    expected_owner_id: str,
) -> None:
    memory = uma_memory
    now = datetime.now(timezone.utc)
    skill = Skill(
        id=f"skill_{expected_owner_type}",
        name="Target-owner skill",
        description="Verifies explicit write-target persistence.",
        created_at=now,
        updated_at=now,
        trigger_phrases=["owner"],
        trigger_patterns=[],
        plan={"steps": ["verify"]},
        tools=["shell"],
        example="check owner",
        meta={},
    )
    embedding = (await memory.embedder.embed([skill.description]))[0]

    persisted = await memory.procedural_core.add_skill_for_owner(
        skill,
        embedding,
        target_owner=target_owner,
    )
    assert persisted is not None

    conn = memory.procedural_core.store._conn()
    try:
        rows = memory.procedural_core.store._query_all(
            conn,
            """
            SELECT tenant_id, owner_type, owner_id, workspace_id
            FROM skills WHERE id=?
            """,
            params=[skill.id],
            log_context="test_target_owner_skill_write",
        )
        assert rows
        assert rows[0]["tenant_id"] == "tenant-1"
        assert rows[0]["owner_type"] == expected_owner_type
        assert rows[0]["owner_id"] == expected_owner_id
        assert rows[0]["workspace_id"] == getattr(target_owner, "workspace_id", None)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_procedural_add_skill_for_owner_rejects_system_target(uma_memory) -> None:
    memory = uma_memory
    now = datetime.now(timezone.utc)
    skill = Skill(
        id="skill_system_reject",
        name="Unsupported target",
        description="Should not persist under system owner in PR4.",
        created_at=now,
        updated_at=now,
        trigger_phrases=["reject"],
        trigger_patterns=[],
        plan={"steps": ["reject"]},
        tools=["shell"],
        example="reject",
        meta={},
    )
    embedding = (await memory.embedder.embed([skill.description]))[0]

    persisted = await memory.procedural_core.add_skill_for_owner(
        skill,
        embedding,
        target_owner=TargetOwner(
            tenant_id="tenant-1",
            owner_type="system",
            owner_id="system:ops",
        ),
    )
    assert persisted is None

    conn = memory.procedural_core.store._conn()
    try:
        rows = memory.procedural_core.store._query_all(
            conn,
            "SELECT id FROM skills WHERE id=?",
            params=[skill.id],
            log_context="test_target_owner_system_reject",
        )
        assert rows == []
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_legacy_procedural_add_skill_boundary_adapter_still_works(uma_memory) -> None:
    memory = uma_memory
    now = datetime.now(timezone.utc)
    skill = Skill(
        id="skill_legacy_adapter",
        name="Legacy adapter",
        description="Still writes through add_skill(skill, embedding).",
        created_at=now,
        updated_at=now,
        owner_type="agent",
        owner_id="agent:alpha",
        trigger_phrases=["legacy"],
        trigger_patterns=[],
        plan={"steps": ["persist"]},
        tools=["shell"],
        example="legacy",
        meta={},
    )
    embedding = (await memory.embedder.embed([skill.description]))[0]

    persisted = await memory.procedural_core.add_skill(skill, embedding)
    assert persisted is not None

    conn = memory.procedural_core.store._conn()
    try:
        rows = memory.procedural_core.store._query_all(
            conn,
            "SELECT owner_type, owner_id FROM skills WHERE id=?",
            params=[skill.id],
            log_context="test_target_owner_legacy_adapter",
        )
        assert rows
        assert rows[0]["owner_type"] == "agent"
        assert rows[0]["owner_id"] == "agent:alpha"
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_owner", "lookup_owner"),
    [
        (
            TargetOwner(tenant_id="tenant-1", owner_type="user", owner_id="user:u1"),
            OwnershipRef(tenant_id="tenant-1", owner_type="user", owner_id="user:u1"),
        ),
        (
            TargetOwner(tenant_id="tenant-1", owner_type="agent", owner_id="agent:alpha"),
            OwnershipRef(tenant_id="tenant-1", owner_type="agent", owner_id="agent:alpha"),
        ),
        (
            TargetOwner(
                tenant_id="tenant-1",
                owner_type="workspace",
                owner_id="workspace:alpha",
                workspace_id="workspace:alpha",
            ),
            OwnershipRef(tenant_id="tenant-1", owner_type="workspace", owner_id="workspace:alpha"),
        ),
    ],
)
async def test_procedural_reads_require_explicit_owner_scope(
    uma_memory,
    target_owner: TargetOwner,
    lookup_owner: OwnershipRef,
) -> None:
    memory = uma_memory
    now = datetime.now(timezone.utc)
    suffix = target_owner.owner_type
    skill = Skill(
        id=f"skill_read_{suffix}",
        name="Scoped read skill",
        description="Verifies explicit procedural read scoping.",
        created_at=now,
        updated_at=now,
        trigger_phrases=["scoped read"],
        trigger_patterns=[],
        plan={"steps": ["read"]},
        tools=["shell"],
        example="read scope",
        meta={},
    )
    embedding = (await memory.embedder.embed([skill.description]))[0]
    persisted = await memory.procedural_core.add_skill_for_owner(
        skill,
        embedding,
        target_owner=target_owner,
    )
    assert persisted is not None
    assert not hasattr(memory, "user_id")

    query_embedding = (await memory.embedder.embed(["scoped read"]))[0]
    found = await memory.procedural_core.search(
        query_embedding=query_embedding,
        owner=lookup_owner,
        k=5,
    )
    assert [item.id for item in found] == [skill.id]

    loaded = await memory.procedural_core.get_skill(skill.id, owner=lookup_owner)
    assert loaded is not None
    assert loaded.id == skill.id

    listed = await memory.procedural_core.list_skills(owner=lookup_owner, limit=5)
    assert [item.id for item in listed] == [skill.id]


@pytest.mark.asyncio
async def test_procedural_reads_reject_system_scope(uma_memory) -> None:
    query_embedding = [0.0] * int(uma_memory.embedding_cfg.dimension)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.search(
            query_embedding=query_embedding,
            owner=OwnershipRef(tenant_id="default", owner_type="system", owner_id="system:ops"),
            k=5,
        )


@pytest.mark.asyncio
async def test_procedural_core_read_apis_fail_clearly_for_unsupported_scope(uma_memory) -> None:
    owner = OwnershipRef(tenant_id="default", owner_type="system", owner_id="system:ops")
    query_embedding = [0.0] * int(uma_memory.embedding_cfg.dimension)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.get_skill("skill-missing", owner=owner)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.list_skills(owner=owner, limit=5)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.fetch_by_ids(["skill-missing"], owner=owner)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.delete_skill("skill-missing", owner=owner)
    with pytest.raises(ValueError, match="owner_type must be one of: agent, user, workspace"):
        await uma_memory.procedural_core.search(
            query_embedding=query_embedding,
            owner=owner,
            k=5,
        )
