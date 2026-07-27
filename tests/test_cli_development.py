from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from uma.cli import main


def _repo(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / "uma").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_security_example.py").touch()
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_dev_check_lists_predefined_checks(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "uma.cli.development.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert main(["--format", "json", "dev", "check", "--list"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert {check["name"] for check in result["data"]["checks"]} == {
        "ruff",
        "bandit",
        "security-tests",
        "contract-tests",
        "pip-check",
        "pytest",
        "pip-audit",
        "build",
        "twine",
    }


def test_dev_check_quick_profile_runs_predefined_commands(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _repo(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr("uma.cli.development._find_executable", lambda name: name)
    monkeypatch.setattr(
        "uma.cli.development.importlib.util.find_spec",
        lambda name: SimpleNamespace(loader=object()),
    )

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("uma.cli.development.subprocess.run", run)

    assert main(["--format", "json", "dev", "check"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert [check["name"] for check in result["data"]["checks"]] == [
        "ruff",
        "bandit",
        "security-tests",
        "contract-tests",
    ]
    assert len(commands) == 4


def test_dev_check_only_and_fail_fast(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _repo(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr("uma.cli.development._find_executable", lambda name: name)

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="lint failed")

    monkeypatch.setattr("uma.cli.development.subprocess.run", run)

    assert (
        main(
            [
                "--format",
                "json",
                "dev",
                "check",
                "--only",
                "ruff,bandit",
                "--fail-fast",
            ]
        )
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert [check["name"] for check in result["data"]["checks"]] == ["ruff"]
    assert len(commands) == 1


def test_dev_check_missing_tool_returns_dependency_error(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _repo(tmp_path, monkeypatch)
    monkeypatch.setattr("uma.cli.development._find_executable", lambda name: None)

    assert main(["--format", "json", "dev", "check", "--only", "ruff"]) == 3

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["data"]["checks"][0]["status"] == "missing"


def test_dev_check_rejects_local_directory_as_python_tool(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "uma.cli.development.importlib.util.find_spec",
        lambda name: SimpleNamespace(loader=None),
    )

    assert main(["--format", "json", "dev", "check", "--only", "build"]) == 3

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["checks"][0]["status"] == "missing"


def test_dev_check_rejects_unknown_check(capsys) -> None:
    assert main(["--format", "json", "dev", "check", "--only", "unknown"]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "unknown development checks" in result["errors"][0]["message"]
