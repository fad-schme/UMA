from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from tests.helpers.runtime import build_test_config
from uma.cli import main
from uma.ingest.types import IngestReport


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "uma.yaml"
    path.write_text(
        yaml.safe_dump(build_test_config(db_root=tmp_path / "db")),
        encoding="utf-8",
    )
    return path


class _FakeMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.shutdown_called = False
        self.failure: Exception | None = None

    async def retrieve_context(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("retrieve_context", kwargs))
        if self.failure:
            raise self.failure
        return {"product": "context", "query": kwargs["query_text"]}

    async def retrieve_memory(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("retrieve_memory", kwargs))
        if self.failure:
            raise self.failure
        return {"product": "memory", "query": kwargs["query_text"]}

    async def ingest_document(self, file_path: str, **kwargs: Any) -> IngestReport:
        self.calls.append(
            (
                "ingest_document",
                {"file_path": file_path, **kwargs},
            )
        )
        return IngestReport(
            doc_id="doc-1",
            chunks_created=2,
            facts_created=1,
            graph_edges_created=0,
            warnings=[],
        )

    async def process_turn(self, **kwargs: Any) -> None:
        self.calls.append(("process_turn", kwargs))

    async def load_memory_bootstrap(
        self,
        file_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "load_memory_bootstrap",
                {"file_path": file_path, **kwargs},
            )
        )
        return {"status": "ingested", "facts_created": 1}

    async def load_daily_diary_bootstrap(
        self,
        file_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "load_daily_diary_bootstrap",
                {"file_path": file_path, **kwargs},
            )
        )
        return {"status": "ingested", "episodes_created": 1}

    def shutdown(self) -> None:
        self.shutdown_called = True


def _install_memory(monkeypatch, memory: _FakeMemory) -> None:
    monkeypatch.setattr(
        "uma.cli.operations.UMAMemory",
        SimpleNamespace(from_yaml=lambda path: memory),
    )


def test_retrieve_context_uses_request_scope_and_reports_audit_effect(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    _install_memory(monkeypatch, memory)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "retrieve",
                "context",
                "Where is the runbook?",
                "--tenant",
                "tenant-a",
                "--agent",
                "agent-a",
                "--user",
                "user-a",
                "--session",
                "session-a",
                "--workspace",
                "workspace-a",
                "--request-id",
                "request-a",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["command"] == "retrieve.context"
    assert result["data"]["effects"] == ["retrieval_audit_write"]
    assert result["data"]["scope"] == {
        "tenant_id": "tenant-a",
        "agent_id": "agent-a",
        "user_id": "user-a",
        "session_id": "session-a",
        "workspace_id": "workspace-a",
        "request_id": "request-a",
    }
    assert memory.calls == [
        (
            "retrieve_context",
            {
                "query_text": "Where is the runbook?",
                "agent_id": "agent-a",
                "user_id": "user-a",
                "tenant_id": "tenant-a",
                "request_id": "request-a",
                "workspace_id": "workspace-a",
                "session_id": "session-a",
            },
        )
    ]
    assert memory.shutdown_called is True


def test_retrieve_memory_calls_public_memory_api(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    _install_memory(monkeypatch, memory)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "retrieve",
                "memory",
                "What changed?",
                "--agent",
                "agent-a",
                "--user",
                "user-a",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["effects"] == ["retrieval_audit_write"]
    assert memory.calls[0][0] == "retrieve_memory"


def test_retrieval_requires_agent_and_user_before_runtime_initialization(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    monkeypatch.setattr(
        "uma.cli.operations.UMAMemory",
        SimpleNamespace(
            from_yaml=lambda path: (_ for _ in ()).throw(
                AssertionError("must not initialize")
            )
        ),
    )

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "retrieve",
                "context",
                "query",
            ]
        )
        == 2
    )

    result = json.loads(capsys.readouterr().out)
    assert "--agent" in result["errors"][0]["message"]
    assert "--user" in result["errors"][0]["message"]


def test_document_ingestion_is_scoped_by_owner_tuple_alone(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    document = tmp_path / "document.txt"
    document.write_text("operational notes", encoding="utf-8")
    memory = _FakeMemory()
    _install_memory(monkeypatch, memory)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "ingest",
                "document",
                str(document),
                "--tenant",
                "tenant-a",
                "--owner-type",
                "workspace",
                "--owner-id",
                "workspace-a",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["effects"] == [
        "memory_write",
        "vector_index_write",
    ]
    assert result["data"]["scope"] == {
        "tenant_id": "tenant-a",
        "owner_type": "workspace",
        "owner_id": "workspace-a",
    }
    assert result["data"]["result"]["chunks_created"] == 2
    assert memory.calls[0][0] == "ingest_document"
    # Uploading a file is a user action; the owner tuple alone decides who
    # reads the document back, so no agent identity travels with the ingest.
    assert "agent_id" not in memory.calls[0][1]


def test_ingest_missing_file_fails_before_runtime_initialization(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    initialized = False

    def from_yaml(config_path: str) -> Any:
        nonlocal initialized
        initialized = True
        raise AssertionError("must not initialize")

    monkeypatch.setattr(
        "uma.cli.operations.UMAMemory",
        SimpleNamespace(from_yaml=from_yaml),
    )

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "ingest",
                "document",
                str(tmp_path / "missing.txt"),
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
            ]
        )
        == 2
    )

    result = json.loads(capsys.readouterr().out)
    assert "input file not found" in result["errors"][0]["message"]
    assert initialized is False


def test_turn_ingestion_requires_scope_and_never_passes_skip_scan(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    _install_memory(monkeypatch, memory)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "ingest",
                "turn",
                "--user-message",
                "What is the runbook?",
                "--assistant-reply",
                "The runbook is in the operations repository.",
                "--agent",
                "agent-a",
                "--user",
                "user-a",
                "--session",
                "session-a",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["effects"] == [
        "memory_write",
        "vector_index_write",
    ]
    name, kwargs = memory.calls[0]
    assert name == "process_turn"
    assert "skip_scan" not in kwargs
    assert kwargs["session_id"] == "session-a"


def test_turn_ingestion_requires_session(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    _install_memory(monkeypatch, memory)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "ingest",
                "turn",
                "--user-message",
                "hello",
                "--assistant-reply",
                "hi",
                "--agent",
                "agent-a",
                "--user",
                "user-a",
            ]
        )
        == 2
    )

    result = json.loads(capsys.readouterr().out)
    assert "--session" in result["errors"][0]["message"]
    assert memory.calls == []


def test_bootstrap_commands_use_public_scoped_loaders(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)

    for operation, method_name in (
        ("memory-bootstrap", "load_memory_bootstrap"),
        ("diary-bootstrap", "load_daily_diary_bootstrap"),
    ):
        source = tmp_path / f"{operation}.md"
        source.write_text("- durable entry\n", encoding="utf-8")
        memory = _FakeMemory()
        _install_memory(monkeypatch, memory)

        assert (
            main(
                [
                    "--config",
                    str(path),
                    "--format",
                    "json",
                    "ingest",
                    operation,
                    str(source),
                    "--agent",
                    "agent-a",
                    "--user",
                    "user-a",
                ]
            )
            == 0
        )
        result = json.loads(capsys.readouterr().out)
        assert result["data"]["effects"] == [
            "memory_write",
            "vector_index_write",
        ]
        assert memory.calls[0][0] == method_name


def test_audit_list_always_filters_one_explicit_tenant(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    _install_memory(monkeypatch, memory)
    calls: list[dict[str, Any]] = []

    async def list_audit(memory_arg: Any, **kwargs: Any) -> list[dict[str, Any]]:
        assert memory_arg is memory
        calls.append(kwargs)
        return [{"request_id": "request-a", "tenant_id": kwargs["tenant_id"]}]

    monkeypatch.setattr(
        "uma.cli.operations.management_api.list_retrieval_audit",
        list_audit,
    )
    monkeypatch.setenv("UMA_TENANT_ID", "tenant-from-env")

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "audit",
                "list",
                "--user",
                "user-a",
                "--limit",
                "25",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["effects"] == []
    assert calls == [
        {
            "tenant_id": "tenant-from-env",
            "user_id": "user-a",
            "severity_min": None,
            "limit": 25,
        }
    ]
    assert calls[0]["tenant_id"] is not None


def test_quarantine_list_requires_independent_owner_scope(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    _install_memory(monkeypatch, memory)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "quarantine",
                "list",
            ]
        )
        == 2
    )

    result = json.loads(capsys.readouterr().out)
    message = result["errors"][0]["message"]
    assert "--owner-type" in message
    assert "--owner-id" in message


def test_quarantine_list_uses_public_owner_scoped_enumerator(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    _install_memory(monkeypatch, memory)
    calls: list[dict[str, Any]] = []

    async def list_records(
        memory_arg: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        assert memory_arg is memory
        calls.append(kwargs)
        return [
            {
                "id": "fact-a",
                "lane": "semantic",
                "quarantined_at": datetime(
                    2026,
                    7,
                    30,
                    tzinfo=timezone.utc,
                ),
            }
        ]

    monkeypatch.setattr(
        "uma.cli.operations.management_api.list_quarantined",
        list_records,
    )

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "quarantine",
                "list",
                "--tenant",
                "tenant-a",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "semantic",
                "--limit",
                "10",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["effects"] == []
    assert result["data"]["result"][0]["quarantined_at"].startswith(
        "2026-07-30"
    )
    assert calls == [
        {
            "tenant_id": "tenant-a",
            "owner_type": "user",
            "owner_id": "user-a",
            "lane": "semantic",
            "limit": 10,
        }
    ]


def test_quarantine_list_failure_is_not_reported_as_empty_success(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    _install_memory(monkeypatch, memory)

    async def fail(*args: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError("quarantine store unavailable")

    monkeypatch.setattr(
        "uma.cli.operations.management_api.list_quarantined",
        fail,
    )

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "quarantine",
                "list",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
            ]
        )
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "store unavailable" in result["data"]["error"]
    assert memory.shutdown_called is True


def test_runtime_operation_failure_is_exit_one_and_still_shuts_down(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    memory.failure = RuntimeError("retrieval exploded")
    _install_memory(monkeypatch, memory)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "retrieve",
                "context",
                "query",
                "--agent",
                "agent-a",
                "--user",
                "user-a",
            ]
        )
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert result["data"]["effects"] == ["retrieval_audit_write"]
    assert "retrieval exploded" in result["data"]["error"]
    assert memory.shutdown_called is True
