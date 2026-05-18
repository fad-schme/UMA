from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from uma.common.ownership import validate_explicit_owner
from uma.common.types import Fact, OwnershipRef, Skill, SCOPE_MODEL_VERSION
from uma.memory.promotion import PromotionPolicy


def test_explicit_write_owner_accepts_agent_user_and_workspace() -> None:
    agent_owner = validate_explicit_owner(
        owner_type="agent",
        owner_id="agent:alpha",
    )
    assert agent_owner == {
        "tenant_id": "default",
        "owner_type": "agent",
        "owner_id": "agent:alpha",
        "workspace_id": None,
    }

    user_owner = validate_explicit_owner(
        owner_type="user",
        owner_id="u1",
    )
    assert user_owner == {
        "tenant_id": "default",
        "owner_type": "user",
        "owner_id": "user:u1",
        "workspace_id": None,
    }

    workspace_owner = validate_explicit_owner(
        owner_type="workspace",
        owner_id="workspace:alpha",
    )
    assert workspace_owner == {
        "tenant_id": "default",
        "owner_type": "workspace",
        "owner_id": "workspace:alpha",
        "workspace_id": "workspace:alpha",
    }


def test_explicit_write_owner_accepts_system_scope_when_requested() -> None:
    owner = validate_explicit_owner(owner_type="system", owner_id="system:alpha")
    assert owner == {
        "tenant_id": "default",
        "owner_type": "system",
        "owner_id": "system:alpha",
        "workspace_id": None,
    }


def test_explicit_write_owner_preserves_tenant_and_workspace() -> None:
    owner = validate_explicit_owner(
        tenant_id="tenant-1",
        owner_type="workspace",
        owner_id="workspace:alpha",
        workspace_id="workspace:alpha",
    )
    assert owner == {
        "tenant_id": "tenant-1",
        "owner_type": "workspace",
        "owner_id": "workspace:alpha",
        "workspace_id": "workspace:alpha",
    }


@pytest.mark.asyncio
async def test_document_ingest_rejects_missing_owner_type(uma_memory, tmp_path) -> None:
    path = tmp_path / "missing-owner-type.txt"
    path.write_text("Explicit owner validation should reject missing owner_type.\n")
    with pytest.raises(ValueError, match="owner_type and owner_id are required"):
        await uma_memory.ingest_document(str(path), owner_id="user:u1")


@pytest.mark.asyncio
async def test_document_ingest_rejects_missing_owner_id(uma_memory, tmp_path) -> None:
    path = tmp_path / "missing-owner-id.txt"
    path.write_text("Explicit owner validation should reject missing owner_id.\n")
    with pytest.raises(ValueError, match="owner_type and owner_id are required"):
        await uma_memory.ingest_document(str(path), owner_type="user")


@pytest.mark.asyncio
async def test_promotion_rejects_missing_owner_type(uma_memory) -> None:
    policy = PromotionPolicy(agent_id=uma_memory.agent_id)
    now = datetime.now(timezone.utc)
    fact = Fact(
        id="fact_missing_owner_type",
        subject="team",
        predicate="USES",
        object="kubernetes cluster orchestration for production workloads",
        created_at=now,
        updated_at=now,
        source_ids=["chunk-source-1"],
        confidence=0.95,
        salience=0.92,
        meta={"source_type": "text"},
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        scope_model_version=SCOPE_MODEL_VERSION,
    )
    with pytest.raises(ValueError, match="owner_type and owner_id are required"):
        policy.promote(fact, owner_id="user:u1")


@pytest.mark.asyncio
async def test_promotion_rejects_missing_owner_id(uma_memory) -> None:
    policy = PromotionPolicy(agent_id=uma_memory.agent_id)
    now = datetime.now(timezone.utc)
    fact = Fact(
        id="fact_missing_owner_id",
        subject="team",
        predicate="USES",
        object="kubernetes cluster orchestration for production workloads",
        created_at=now,
        updated_at=now,
        source_ids=["chunk-source-1"],
        confidence=0.95,
        salience=0.92,
        meta={"source_type": "text"},
        owner_type="user",
        owner_id="user:u1",
        tenant_id="default",
        scope_model_version=SCOPE_MODEL_VERSION,
    )
    with pytest.raises(ValueError, match="owner_type and owner_id are required"):
        policy.promote(fact, owner_type="user")


def test_no_legacy_or_duplicate_ownership_resolvers() -> None:
    forbidden = [
        "TargetOwner",
        "target_owner",
        "make_target_owner",
        "resolve_target_owner",
        "select_target_owner",
        "_resolve_owner",
        "_resolve_ownership_ref",
        "_read_owner_ref",
        "_write_owner_from_skill",
        "_select_owner",
        "_select_ownership",
        "resolve_ownership_ref",
        "resolve_explicit_owner",
    ]
    root = Path(__file__).resolve().parents[1]
    checked_files = list((root / "uma").rglob("*.py"))
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{marker} remains in {path}"

    ownership_text = (root / "uma/common/ownership.py").read_text(encoding="utf-8")
    assert "def validate_explicit_owner(" in ownership_text

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_type", "owner_id", "expected_owner_type", "expected_owner_id", "expected_workspace_id"),
    [
        ("user", "user:u1", "user", "user:u1", None),
        ("agent", "agent:alpha", "agent", "agent:alpha", None),
        ("workspace", "workspace:alpha", "workspace", "workspace:alpha", "workspace:alpha"),
    ],
)
async def test_ingest_document_persists_explicit_owner_fields(
    uma_memory,
    tmp_path,
    owner_type: str,
    owner_id: str,
    expected_owner_type: str,
    expected_owner_id: str,
    expected_workspace_id: str | None,
) -> None:
    memory = uma_memory
    path = tmp_path / f"{expected_owner_type}-target-doc.txt"
    path.write_text(
        "UMA explicit ingest owner field test. "
        "This paragraph is deliberately long enough to survive chunking and fact extraction without "
        "falling below the extractor threshold. It includes additional concrete statements about a "
        "workspace playbook, operational checklist, ownership marker, and validation trail so the "
        "document-derived fact path is exercised reliably during ingestion.\n"
    )

    report = await memory.ingest_document(str(path), owner_type=owner_type, owner_id=owner_id)
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
            log_context="test_write_owner_ingest_document",
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
                log_context="test_write_owner_ingest_workspace_facts",
            )
            assert fact_rows
            assert all(row["owner_type"] == "workspace" for row in fact_rows)
            assert all(row["owner_id"] == "workspace:alpha" for row in fact_rows)
            assert all(row["workspace_id"] == "workspace:alpha" for row in fact_rows)
        finally:
            conn.close()


@pytest.mark.asyncio
async def test_document_ingest_requires_explicit_owner_fields(uma_memory, tmp_path) -> None:
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
            log_context="test_write_owner_ingest_manifest",
        )
        assert rows
        assert rows[0]["owner_type"] == "user"
        assert rows[0]["owner_id"] == "user:u1"
        assert rows[0]["tenant_id"] == "default"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_document_ingest_rejects_system_owner_scope(uma_memory, tmp_path) -> None:
    memory = uma_memory
    path = tmp_path / "system-target-doc.txt"
    path.write_text(
        "UMA unsupported system ingest target test. "
        "This paragraph is intentionally long enough to exercise the validation path.\n"
    )

    with pytest.raises(ValueError, match="owner_type"):
        await memory.ingest_document(
            str(path),
            owner_type="system",
            owner_id="system:ops",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_type", "owner_id", "expected_owner_type", "expected_owner_id", "expected_workspace_id"),
    [
        ("user", "user:u1", "user", "user:u1", None),
        ("agent", "agent:alpha", "agent", "agent:alpha", None),
        ("workspace", "workspace:alpha", "workspace", "workspace:alpha", "workspace:alpha"),
    ],
)
async def test_procedural_add_skill_for_owner_persists_explicit_owner_fields(
    uma_memory,
    owner_type: str,
    owner_id: str,
    expected_owner_type: str,
    expected_owner_id: str,
    expected_workspace_id: str | None,
) -> None:
    memory = uma_memory
    now = datetime.now(timezone.utc)
    skill = Skill(
        id=f"skill_{expected_owner_type}",
        name="Explicit owner skill",
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
        tenant_id="tenant-1",
        owner_type=owner_type,
        owner_id=owner_id,
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
            log_context="test_write_owner_skill_write",
        )
        assert rows
        assert rows[0]["tenant_id"] == "tenant-1"
        assert rows[0]["owner_type"] == expected_owner_type
        assert rows[0]["owner_id"] == expected_owner_id
        assert rows[0]["workspace_id"] == expected_workspace_id
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_procedural_add_skill_for_owner_rejects_system_scope(uma_memory) -> None:
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
        tenant_id="tenant-1",
        owner_type="system",
        owner_id="system:ops",
    )
    assert persisted is None

    conn = memory.procedural_core.store._conn()
    try:
        rows = memory.procedural_core.store._query_all(
            conn,
            "SELECT id FROM skills WHERE id=?",
            params=[skill.id],
            log_context="test_write_owner_system_reject",
        )
        assert rows == []
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_procedural_add_skill_uses_skill_owner_fields_when_no_override_is_given(uma_memory) -> None:
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
            log_context="test_write_owner_add_skill_default",
        )
        assert rows
        assert rows[0]["owner_type"] == "agent"
        assert rows[0]["owner_id"] == "agent:alpha"
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("write_owner_type", "write_owner_id", "lookup_owner"),
    [
        (
            "user",
            "user:u1",
            OwnershipRef(tenant_id="tenant-1", owner_type="user", owner_id="user:u1"),
        ),
        (
            "agent",
            "agent:alpha",
            OwnershipRef(tenant_id="tenant-1", owner_type="agent", owner_id="agent:alpha"),
        ),
        (
            "workspace",
            "workspace:alpha",
            OwnershipRef(tenant_id="tenant-1", owner_type="workspace", owner_id="workspace:alpha"),
        ),
    ],
)
async def test_procedural_reads_require_explicit_owner_scope(
    uma_memory,
    write_owner_type: str,
    write_owner_id: str,
    lookup_owner: OwnershipRef,
) -> None:
    memory = uma_memory
    now = datetime.now(timezone.utc)
    suffix = write_owner_type
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
        tenant_id="tenant-1",
        owner_type=write_owner_type,
        owner_id=write_owner_id,
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
