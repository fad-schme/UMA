from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from tests.helpers.runtime import build_test_config
from uma.api.memory import UMAMemory
from uma.cli import main
from uma.common.health import HealthCheck
from uma.common.results import HealthStatus


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "uma.yaml"
    path.write_text(
        yaml.safe_dump(build_test_config(db_root=tmp_path / "db")),
        encoding="utf-8",
    )
    return path


def _health(
    status: str = "ok",
    *,
    check_name: str = "runtime",
    check_status: str | None = None,
    detail: str = "ready",
) -> HealthStatus:
    resolved_check_status = check_status or status
    return HealthStatus(
        status=status,
        checks={
            check_name: HealthCheck(
                name=check_name,
                status=resolved_check_status,
                detail=detail,
            )
        },
    )


class _FakeMemory:
    def __init__(
        self,
        health: HealthStatus | None = None,
        *,
        health_error: Exception | None = None,
        shutdown_error: Exception | None = None,
    ) -> None:
        self.health = health or _health()
        self.health_error = health_error
        self.shutdown_error = shutdown_error
        self.shutdown_called = False
        self.context_agents: list[str] = []

    def set_context(self, *, agent_id: str) -> "_FakeMemory":
        self.context_agents.append(agent_id)
        return self

    def health_check(self) -> HealthStatus:
        if self.health_error:
            raise self.health_error
        return self.health

    def shutdown(self) -> None:
        self.shutdown_called = True
        if self.shutdown_error:
            raise self.shutdown_error


class _HealthyGraphAdapter:
    def verify_connectivity(self) -> bool:
        return True


class _HealthyGraphCore:
    def __init__(self) -> None:
        self.adapter = _HealthyGraphAdapter()

    def close(self) -> None:
        return None


def _factory(memory: Any) -> SimpleNamespace:
    return SimpleNamespace(from_yaml=lambda path: memory)


def test_health_healthy_inmemory_runtime(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    real_from_yaml = UMAMemory.from_yaml
    shutdown_called = False

    def from_yaml(config_path: str) -> UMAMemory:
        nonlocal shutdown_called
        memory = real_from_yaml(config_path)
        memory.graph_core = _HealthyGraphCore()
        real_shutdown = memory.shutdown

        def shutdown() -> None:
            nonlocal shutdown_called
            shutdown_called = True
            real_shutdown()

        memory.shutdown = shutdown
        return memory

    monkeypatch.setattr(
        "uma.cli.diagnostics.UMAMemory",
        SimpleNamespace(from_yaml=from_yaml),
    )

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["data"]["llm_probe"] == "initialization_only"
    assert all(
        check["status"] == "ok"
        for check in result["data"]["checks"].values()
    )
    assert shutdown_called is True


def test_health_missing_config_is_a_usage_error(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UMA_CONFIG", raising=False)

    assert main(["--format", "json", "health"]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "Use --config or UMA_CONFIG" in result["errors"][0]["message"]


def test_health_invalid_provider_is_runtime_error(
    tmp_path: Path,
    capsys,
) -> None:
    path = _config_path(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["llms"]["uma"]["provider"] = "invalid-provider"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "Unsupported provider" in result["data"]["checks"][
        "initialization"
    ]["detail"]


def test_health_embedder_dimension_mismatch(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory(
        _health(
            "error",
            check_name="embedding",
            detail="dimension mismatch: embedder=32 config=64",
        )
    )
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    embedding = result["data"]["checks"]["embedding"]
    assert embedding["status"] == "error"
    assert "dimension mismatch" in embedding["detail"]
    assert memory.shutdown_called is True


def test_health_missing_vector_index_is_degraded(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory(
        _health(
            "degraded",
            check_name="vector:semantic",
            check_status="skipped",
            detail="index missing",
        )
    )
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 4
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "degraded"
    assert (
        result["data"]["checks"]["vector:semantic"]["detail"]
        == "index missing"
    )


def test_health_degraded_graph_state(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory(
        _health(
            "degraded",
            check_name="graph",
            check_status="skipped",
            detail="graph disabled",
        )
    )
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 4
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "degraded"
    assert result["data"]["checks"]["graph"]["status"] == "skipped"


def test_health_initialization_exception_is_reported(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)

    def fail_initialization(config_path: str) -> Any:
        raise RuntimeError("initialization exploded")

    monkeypatch.setattr(
        "uma.cli.diagnostics.UMAMemory",
        SimpleNamespace(from_yaml=fail_initialization),
    )

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "initialization exploded" in result["data"]["checks"][
        "initialization"
    ]["detail"]


def test_health_always_shuts_down_after_success(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert memory.shutdown_called is True
    assert memory.context_agents == []


def test_health_always_shuts_down_after_health_failure(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory(health_error=RuntimeError("probe exploded"))
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert "probe exploded" in result["data"]["checks"]["health_check"][
        "detail"
    ]
    assert memory.shutdown_called is True


def test_health_shutdown_failure_changes_result_to_error(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory(shutdown_error=RuntimeError("close exploded"))
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "close exploded" in result["data"]["checks"]["shutdown"]["detail"]
    assert memory.shutdown_called is True


def test_health_applies_agent_context_only_when_requested(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "health",
                "--agent",
                "agent-7",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["agent_id"] == "agent-7"
    assert memory.context_agents == ["agent-7"]


def test_health_timeout_is_error_and_still_shuts_down(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory()

    def slow_health() -> HealthStatus:
        time.sleep(0.1)
        return _health()

    memory.health_check = slow_health
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(
            [
                "--config",
                str(path),
                "--format",
                "json",
                "health",
                "--timeout",
                "0.01",
            ]
        )
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "exceeded" in result["data"]["checks"]["timeout"]["detail"]
    assert memory.shutdown_called is True


def test_default_doctor_combines_offline_and_runtime_health(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory(
        _health(
            "degraded",
            check_name="graph",
            check_status="skipped",
            detail="graph disabled",
        )
    )
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))
    monkeypatch.setattr("uma.cli.diagnostics._module_available", lambda _: True)

    assert (
        main(["--config", str(path), "--format", "json", "doctor"])
        == 4
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "degraded"
    assert result["data"]["mode"] == "runtime"
    assert result["data"]["offline"]["mode"] == "offline"
    assert result["data"]["runtime"]["checks"]["graph"]["status"] == "skipped"


def test_doctor_offline_never_initializes_runtime(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    monkeypatch.setattr("uma.cli.diagnostics._module_available", lambda _: True)
    monkeypatch.setattr(
        "uma.cli.diagnostics.UMAMemory",
        SimpleNamespace(
            from_yaml=lambda path: (_ for _ in ()).throw(
                AssertionError("must stay offline")
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
                "doctor",
                "--offline",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["data"]["mode"] == "offline"
