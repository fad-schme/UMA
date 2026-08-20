from __future__ import annotations

import json
import pytest
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


def test_health_missing_vector_index_is_error(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory(
        _health(
            "error",
            check_name="vector:semantic",
            check_status="error",
            detail="index missing",
        )
    )
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert (
        result["data"]["checks"]["vector:semantic"]["detail"]
        == "index missing"
    )


def test_health_disabled_graph_is_ok_and_exits_zero(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    """Graph is opt-in; disabling it must not fail a container healthcheck."""
    path = _config_path(tmp_path)
    memory = _FakeMemory(
        _health(
            "ok",
            check_name="graph",
            check_status="disabled",
            detail="graph disabled",
        )
    )
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["data"]["checks"]["graph"]["status"] == "disabled"


def test_health_degraded_status_maps_to_exit_four(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    path = _config_path(tmp_path)
    memory = _FakeMemory(
        _health(
            "degraded",
            check_name="vector:semantic",
            check_status="degraded",
            detail="reachable but rebuilding",
        )
    )
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert (
        main(["--config", str(path), "--format", "json", "health"])
        == 4
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "degraded"


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


def test_health_has_no_agent_scope(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    """health_check probes dependency readiness, which every agent on the
    runtime shares. The command takes no agent and reports none."""
    path = _config_path(tmp_path)
    memory = _FakeMemory()
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))

    assert main(["--config", str(path), "--format", "json", "health"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert "agent_id" not in result["data"]

    with pytest.raises(SystemExit):
        main(["--config", str(path), "health", "--agent", "agent-7"])


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
            "ok",
            check_name="graph",
            check_status="disabled",
            detail="graph disabled",
        )
    )
    monkeypatch.setattr("uma.cli.diagnostics.UMAMemory", _factory(memory))
    monkeypatch.setattr("uma.cli.diagnostics._module_available", lambda _: True)

    assert (
        main(["--config", str(path), "--format", "json", "doctor"])
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["data"]["mode"] == "runtime"
    assert result["data"]["offline"]["mode"] == "offline"
    assert result["data"]["runtime"]["checks"]["graph"]["status"] == "disabled"


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


# ---------------------------------------------------------------------------
# run_health_checks aggregation
#
# The aggregation itself had no direct coverage: every test above feeds a
# pre-built HealthStatus into the CLI. That gap is why a correct lite install
# reported "degraded" and exited 4 for its whole 0.2.0 life.
# ---------------------------------------------------------------------------


class _StubAdapter:
    def __init__(self, rows: Any = (1,)) -> None:
        self._rows = rows

    def get_connection(self) -> Any:
        return self

    def cursor(self) -> Any:
        return self

    def execute(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def fetchone(self) -> Any:
        return self._rows

    def close(self) -> None:
        return None


class _StubVectorIndex:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def query(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []


class _StubLLM:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def generate(self, *_args: Any, **_kwargs: Any) -> str:
        if self.error:
            raise self.error
        return "ok"


class _StubEmbedder:
    def __init__(self, dim: int = 8, error: Exception | None = None) -> None:
        self.dimension = dim
        self.error = error

    async def embed(self, texts: Any) -> list[list[float]]:
        if self.error:
            raise self.error
        return [[0.0] * self.dimension for _ in texts]


def _health_memory(
    *,
    graph_backend: str = "disabled",
    graph_core: Any = None,
    dim: int = 8,
    llm: Any = None,
    embedder: Any = None,
) -> Any:
    """Minimal duck-typed stand-in matching what run_health_checks reads."""
    store = SimpleNamespace(_db_adapter=_StubAdapter(), vector_index=_StubVectorIndex(dim))
    cfg = SimpleNamespace(
        provider="ollama",
        model="m",
        dimension=dim,
        config={"host": "http://localhost:11434", "timeout": 5.0},
    )
    return SimpleNamespace(
        episodic_core=SimpleNamespace(store=store),
        semantic_core=SimpleNamespace(ingestor=SimpleNamespace(semantic_store=store)),
        procedural_core=SimpleNamespace(store=store),
        graph_core=graph_core,
        raw_config=SimpleNamespace(storage=SimpleNamespace(graph_backend=graph_backend)),
        embedding_cfg=cfg,
        llm_cfg=cfg,
        agent_llm_cfg=cfg,
        llm=llm if llm is not None else _StubLLM(),
        embedder=embedder if embedder is not None else _StubEmbedder(dim),
    )


def test_disabled_graph_does_not_degrade_overall_health() -> None:
    from uma.common.health import run_health_checks

    health = run_health_checks(_health_memory(graph_backend="disabled"))

    assert health.checks["graph"].status == "disabled"
    assert health.status == "ok"


def test_configured_graph_without_adapter_is_an_error() -> None:
    from uma.common.health import run_health_checks

    health = run_health_checks(_health_memory(graph_backend="pkg.mod:Factory"))

    assert health.checks["graph"].status == "error"
    assert health.status == "error"


def test_missing_store_adapter_is_error_not_skipped() -> None:
    from uma.common.health import run_health_checks

    memory = _health_memory()
    memory.episodic_core.store._db_adapter = None

    health = run_health_checks(memory)

    assert health.checks["db:episodic"].status == "error"
    assert health.status == "error"


# ---------------------------------------------------------------------------
# Provider reachability
#
# UMA is RLM-first: retrieval and ingestion both require an LLM, and every
# lane requires an embedder. A provider that is configured but not running is
# a hard failure, so health must actually call them rather than assert the
# adapter object exists.
# ---------------------------------------------------------------------------


def test_unreachable_llm_is_an_error() -> None:
    from uma.common.health import run_health_checks

    memory = _health_memory(llm=_StubLLM(error=ConnectionError("connection refused")))
    health = run_health_checks(memory)

    assert health.checks["llm"].status == "error"
    assert "connection refused" in health.checks["llm"].detail
    assert "ollama:m" in health.checks["llm"].detail
    assert health.status == "error"


def test_unreachable_embedder_is_an_error() -> None:
    from uma.common.health import run_health_checks

    memory = _health_memory(
        embedder=_StubEmbedder(error=ConnectionError("connection refused"))
    )
    health = run_health_checks(memory)

    assert health.checks["embedding"].status == "error"
    assert "connection refused" in health.checks["embedding"].detail
    assert health.status == "error"


def test_provider_timeout_is_an_error_naming_the_deadline() -> None:
    import asyncio as _asyncio

    from uma.common.health import run_health_checks

    class _SlowEmbedder:
        dimension = 8

        async def embed(self, _texts: Any) -> list[list[float]]:
            await _asyncio.sleep(5)
            return [[0.0] * 8]

    memory = _health_memory(embedder=_SlowEmbedder())
    memory.embedding_cfg.config = {"host": "http://localhost:11434", "timeout": 0.05}

    health = run_health_checks(memory)

    assert health.checks["embedding"].status == "error"
    assert "did not respond" in health.checks["embedding"].detail
    assert health.status == "error"


def test_embedder_returning_wrong_dimension_is_an_error() -> None:
    from uma.common.health import run_health_checks

    class _WrongDim:
        dimension = 8

        async def embed(self, _texts: Any) -> list[list[float]]:
            return [[0.0] * 3]

    health = run_health_checks(_health_memory(embedder=_WrongDim()))

    assert health.checks["embedding"].status == "error"
    assert "returned dim=3" in health.checks["embedding"].detail


def test_reachable_providers_report_ok() -> None:
    from uma.common.health import run_health_checks

    health = run_health_checks(_health_memory())

    assert health.checks["llm"].status == "ok"
    assert health.checks["embedding"].status == "ok"
    assert health.status == "ok"


@pytest.mark.asyncio
async def test_probe_works_when_called_from_a_running_event_loop() -> None:
    """health_check() is sync but may be called from async application code."""
    from uma.common.health import run_health_checks

    health = run_health_checks(_health_memory())

    assert health.checks["embedding"].status == "ok"
    assert health.status == "ok"
