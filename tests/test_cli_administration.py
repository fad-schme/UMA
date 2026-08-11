from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tests.helpers.cli import uma_entry_point
from tests.helpers.runtime import build_test_config
from uma.api.management import (
    IntegrityVerificationResult,
    list_quarantined,
)
from uma.api.memory import UMAMemory
from uma.cli import main
from uma.cli.confirmation import (
    ConfirmationDeclined,
    ConfirmationRequired,
    require_confirmation,
)
from uma.common.results import (
    DerivedRebuildReport,
    GraphRebuildReport,
    LaneRebuildStatus,
    VectorRebuildReport,
)
from uma.common.types import Fact


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "uma.yaml"
    path.write_text(
        yaml.safe_dump(build_test_config(db_root=tmp_path / "db")),
        encoding="utf-8",
    )
    return path


class _FakeAdminMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.shutdown_called = False

    async def rebuild_vector_indexes(self, **kwargs: Any) -> "VectorRebuildReport":
        self.calls.append(("rebuild_vector_indexes", kwargs))
        selected = next(
            lane
            for lane in ("episodic", "semantic", "procedural")
            if kwargs[f"include_{lane}"]
        )
        return VectorRebuildReport(
            status="degraded",
            report={
                lane: LaneRebuildStatus(
                    status="ok" if lane == selected else "skipped",
                    count=2 if lane == selected else 0,
                )
                for lane in ("episodic", "semantic", "procedural")
            },
        )

    async def rebuild_derived_indexes(self, **kwargs: Any) -> "DerivedRebuildReport":
        self.calls.append(("rebuild_derived_indexes", kwargs))
        selected = next(
            (
                lane
                for lane in ("episodic", "semantic", "procedural")
                if kwargs[f"include_{lane}"]
            ),
            None,
        )
        return DerivedRebuildReport(
            status="degraded",
            vector=VectorRebuildReport(
                status="degraded",
                report={
                    lane: LaneRebuildStatus(
                        status="ok" if lane == selected else "skipped",
                        count=1 if lane == selected else 0,
                    )
                    for lane in ("episodic", "semantic", "procedural")
                },
            ),
            graph=GraphRebuildReport(
                status="ok" if kwargs["include_graph"] else "skipped",
                episodes=0,
                facts=0,
                episode_fact_links=0,
                temporal_links=0,
            ),
        )

    def shutdown(self) -> None:
        self.shutdown_called = True


def _install_memory(monkeypatch, memory: _FakeAdminMemory) -> None:
    monkeypatch.setattr(
        "uma.cli.operations.UMAMemory",
        SimpleNamespace(from_yaml=lambda path: memory),
    )


class _TTYInput(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_require_confirmation_accepts_assume_yes(capsys) -> None:
    require_confirmation(
        message="exact target: tenant-a/user-a/fact-a",
        assume_yes=True,
        stdin_is_tty=False,
    )

    captured = capsys.readouterr()
    assert "tenant-a/user-a/fact-a" in captured.err
    assert captured.out == ""


def test_require_confirmation_rejects_noninteractive_without_yes(
    capsys,
) -> None:
    with pytest.raises(ConfirmationRequired, match="requires --yes"):
        require_confirmation(
            message="exact target: record-a",
            assume_yes=False,
            stdin_is_tty=False,
        )

    assert "record-a" in capsys.readouterr().err


def test_require_confirmation_decline(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", _TTYInput("no\n"))

    with pytest.raises(ConfirmationDeclined, match="declined"):
        require_confirmation(
            message="exact target: record-a",
            assume_yes=False,
            stdin_is_tty=True,
        )

    assert "Continue? [y/N]" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "help_fragment"),
    [
        (["quarantine", "reinstate", "--help"], "--record-id"),
        (["quarantine", "purge", "--help"], "--reason"),
        (["index", "rebuild-vectors", "--help"], "--batch-size"),
        (["index", "rebuild-derived", "--help"], "graph"),
        (["integrity", "enforce", "--help"], "--record-id"),
    ],
)
def test_guarded_command_help_recognizes_options(
    command: list[str],
    help_fragment: str,
    capsys,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(command)

    assert exc_info.value.code == 0
    assert help_fragment in capsys.readouterr().out


def test_noninteractive_mutation_requires_yes_before_sdk_call(
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
                "quarantine",
                "reinstate",
                "--tenant",
                "tenant-a",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "semantic",
                "--record-id",
                "fact-a",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "requires --yes" in result["data"]["error"]
    assert "tenant_id='tenant-a'" in captured.err
    assert "record_id='fact-a'" in captured.err
    assert initialized is False


def test_declined_confirmation_makes_no_sdk_call(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    monkeypatch.setattr("sys.stdin", _TTYInput("n\n"))
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
                "integrity",
                "enforce",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "semantic",
                "--record-id",
                "fact-a",
            ]
        )
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["error"] == "operation declined"


def test_purge_requires_nonempty_reason_before_confirmation_or_sdk(
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
                "quarantine",
                "purge",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "semantic",
                "--record-id",
                "fact-a",
                "--yes",
            ]
        )
        == 2
    )

    result = json.loads(capsys.readouterr().out)
    assert "--reason" in result["errors"][0]["message"]


def test_wildcard_owner_is_rejected_by_shared_scope_parser(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "quarantine",
                "purge",
                "--owner-type",
                "user",
                "--owner-id",
                "user:*",
                "--lane",
                "semantic",
                "--record-id",
                "fact-a",
                "--reason",
                "expired",
                "--yes",
            ]
        )

    assert exc_info.value.code == 2
    assert "wildcards" in capsys.readouterr().err


def test_quarantine_reinstate_calls_exact_public_api_target(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeAdminMemory()
    _install_memory(monkeypatch, memory)
    calls: list[dict[str, Any]] = []

    async def reinstate(memory_arg: Any, **kwargs: Any) -> bool:
        assert memory_arg is memory
        calls.append(kwargs)
        return True

    monkeypatch.setattr(
        "uma.api.management.reinstate_quarantined",
        reinstate,
    )

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "quarantine",
                "reinstate",
                "--tenant",
                "tenant-a",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "semantic",
                "--record-id",
                "fact-a",
                "--reason",
                "false positive",
                "--yes",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["data"]["effects"] == [
        "quarantine_state_write",
        "security_audit_write",
    ]
    assert result["data"]["target"]["record_id"] == "fact-a"
    assert "exact resolved target" in captured.err
    assert calls == [
        {
            "record_id": "fact-a",
            "lane": "semantic",
            "owner_type": "user",
            "owner_id": "user-a",
            "tenant_id": "tenant-a",
            "reason": "false positive",
        }
    ]
    assert memory.shutdown_called is True


def test_quarantine_purge_calls_one_record_only(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeAdminMemory()
    _install_memory(monkeypatch, memory)
    calls: list[dict[str, Any]] = []

    async def purge(memory_arg: Any, **kwargs: Any) -> bool:
        assert memory_arg is memory
        calls.append(kwargs)
        return True

    monkeypatch.setattr("uma.api.management.purge_quarantined", purge)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "quarantine",
                "purge",
                "--tenant",
                "tenant-a",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "semantic",
                "--record-id",
                "fact-a",
                "--reason",
                "retention expired",
                "--yes",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["effects"] == [
        "memory_delete",
        "vector_index_delete",
    ]
    assert calls[0]["record_id"] == "fact-a"
    assert isinstance(calls[0]["record_id"], str)


def test_rebuild_vectors_selects_exactly_one_lane(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeAdminMemory()
    _install_memory(monkeypatch, memory)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "index",
                "rebuild-vectors",
                "--tenant",
                "tenant-a",
                "--owner-type",
                "workspace",
                "--owner-id",
                "workspace-a",
                "--lane",
                "semantic",
                "--batch-size",
                "17",
                "--yes",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "ok"
    assert result["data"]["target"]["record_scope"].startswith("all records")
    assert "lane='semantic'" in captured.err
    assert memory.calls == [
        (
            "rebuild_vector_indexes",
            {
                "tenant_id": "tenant-a",
                "owner_type": "workspace",
                "owner_id": "workspace-a",
                "include_episodic": False,
                "include_semantic": True,
                "include_procedural": False,
                "batch_size": 17,
            },
        )
    ]


def test_rebuild_derived_graph_disables_all_vector_lanes(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeAdminMemory()
    _install_memory(monkeypatch, memory)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "index",
                "rebuild-derived",
                "--tenant",
                "tenant-a",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "graph",
                "--yes",
            ]
        )
        == 0
    )

    json.loads(capsys.readouterr().out)
    assert memory.calls == [
        (
            "rebuild_derived_indexes",
            {
                "tenant_id": "tenant-a",
                "owner_type": "user",
                "owner_id": "user-a",
                "include_episodic": False,
                "include_semantic": False,
                "include_procedural": False,
                "batch_size": 32,
                "include_graph": True,
            },
        )
    ]


def test_integrity_enforce_calls_exact_record_and_reports_findings(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeAdminMemory()
    _install_memory(monkeypatch, memory)
    calls: list[dict[str, Any]] = []

    async def enforce(memory_arg: Any, **kwargs: Any) -> Any:
        assert memory_arg is memory
        calls.append(kwargs)
        return IntegrityVerificationResult(
            record_id=kwargs["record_id"],
            lane=kwargs["lane"],
            status="failed",
            expected_hash="expected",
            actual_hash="actual",
            quarantined=True,
        )

    monkeypatch.setattr("uma.api.management.verify_integrity", enforce)

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "integrity",
                "enforce",
                "--tenant",
                "tenant-a",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "semantic",
                "--record-id",
                "fact-a",
                "--yes",
            ]
        )
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "findings"
    assert result["data"]["result"]["quarantined"] is True
    assert calls == [
        {
            "record_id": "fact-a",
            "lane": "semantic",
            "owner_type": "user",
            "owner_id": "user-a",
            "tenant_id": "tenant-a",
        }
    ]


async def _seed_quarantined_facts(config_path: Path) -> None:
    memory = UMAMemory.from_yaml(str(config_path))
    memory._ensure_ingestion_ready()
    assert memory.semantic_core is not None
    now = datetime.now(timezone.utc)
    try:
        for record_id, tenant_id, owner_id in (
            ("fact_reinstate", "tenant-a", "user-a"),
            ("fact_purge", "tenant-a", "user-a"),
            ("fact_other_scope", "tenant-b", "user-b"),
        ):
            await memory.semantic_core.upsert_fact(
                Fact(
                    id=record_id,
                    subject=owner_id,
                    predicate="contains",
                    object=record_id,
                    created_at=now,
                    updated_at=now,
                    tenant_id=tenant_id,
                    owner_type="user",
                    owner_id=owner_id,
                    trust_score=0.0,
                    quarantined_at=now,
                ),
                [0.1] * 64,
            )
    finally:
        memory.shutdown()


async def _quarantined_ids(
    config_path: Path,
    *,
    tenant_id: str,
    owner_id: str,
) -> set[str]:
    memory = UMAMemory.from_yaml(str(config_path))
    memory._ensure_ingestion_ready()
    try:
        records = await list_quarantined(
            memory,
            tenant_id=tenant_id,
            owner_type="user",
            owner_id=owner_id,
            lane="semantic",
        )
        return {record.id for record in records}
    finally:
        memory.shutdown()


def test_quarantine_commands_use_real_sqlite_and_preserve_scope_isolation(
    tmp_path: Path,
    capsys,
) -> None:
    path = _config_path(tmp_path)
    asyncio.run(_seed_quarantined_facts(path))

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "quarantine",
                "reinstate",
                "--tenant",
                "tenant-a",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "semantic",
                "--record-id",
                "fact_reinstate",
                "--reason",
                "reviewed false positive",
                "--yes",
            ]
        )
        == 0
    )
    reinstate_result = json.loads(capsys.readouterr().out)
    assert reinstate_result["data"]["result"] is True

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "quarantine",
                "purge",
                "--tenant",
                "tenant-a",
                "--owner-type",
                "user",
                "--owner-id",
                "user-a",
                "--lane",
                "semantic",
                "--record-id",
                "fact_purge",
                "--reason",
                "confirmed malicious content",
                "--yes",
            ]
        )
        == 0
    )
    purge_result = json.loads(capsys.readouterr().out)
    assert purge_result["data"]["result"] is True

    assert (
        asyncio.run(
            _quarantined_ids(
                path,
                tenant_id="tenant-a",
                owner_id="user-a",
            )
        )
        == set()
    )
    assert asyncio.run(
        _quarantined_ids(
            path,
            tenant_id="tenant-b",
            owner_id="user-b",
        )
    ) == {"fact_other_scope"}


def test_installed_cli_rejects_noninteractive_purge_without_yes(
    tmp_path: Path,
) -> None:
    path = _config_path(tmp_path)
    executable = uma_entry_point()
    assert executable.is_file(), "the test environment must install the uma entry point"

    completed = subprocess.run(
        [
            str(executable),
            "--config",
            str(path),
            "--format",
            "json",
            "quarantine",
            "purge",
            "--tenant",
            "tenant-a",
            "--owner-type",
            "user",
            "--owner-id",
            "user-a",
            "--lane",
            "semantic",
            "--record-id",
            "fact-a",
            "--reason",
            "confirmed malicious content",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        # `capture_output` redirects stdout/stderr only; without an explicit
        # stdin the child inherits the parent's terminal and `isatty()` is True
        # whenever the suite runs from a shell, sending the CLI down the
        # interactive prompt path instead of the non-interactive one under test.
        # `input` makes stdin a pipe, which reports False on POSIX and Windows
        # alike — `DEVNULL` does not, because Windows `isatty()` returns True
        # for the NUL character device.
        input="",
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert result["command"] == "quarantine.purge"
    assert "requires --yes" in result["data"]["error"]
    assert "tenant_id='tenant-a'" in completed.stderr
    assert "record_id='fact-a'" in completed.stderr
    assert not (tmp_path / "db").exists()
