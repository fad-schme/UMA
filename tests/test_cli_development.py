from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from uma.cli import main
from uma.cli.development import CheckDefinition


def _repo(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / "uma").mkdir()
    (tmp_path / "uma" / "__init__.py").touch()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_security_example.py").touch()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "uma"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _norm(commands: list[list[str]]) -> list[list[str]]:
    """Normalize path separators so argv assertions are platform-independent.

    `dev check` derives a few arguments through `pathlib` (the security test
    module, the built wheel), so they carry `\\` on Windows while the literal
    arguments in the check definitions are written with `/`. Applying this to
    both sides keeps the assertions about *which* commands run rather than
    about `os.sep`; `sys.executable` is normalized identically on both sides.
    """
    return [[part.replace("\\", "/") for part in command] for command in commands]


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
    assert _norm(commands) == _norm([
        [
            "ruff",
            "check",
            "uma/",
            "tests/",
            "--output-format=github",
        ],
        [
            "bandit",
            "--recursive",
            "uma/",
            "--severity-level",
            "medium",
            "--confidence-level",
            "medium",
            "--format",
            "txt",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_security_example.py",
            "-q",
            "--tb=short",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_configuration.py",
            "tests/test_vector_contract.py",
            "-q",
            "--tb=short",
        ],
    ])


def test_check_definition_has_required_immutable_schema() -> None:
    assert CheckDefinition.__dataclass_params__.frozen is True
    assert [field.name for field in fields(CheckDefinition)] == [
        "name",
        "argv",
        "profiles",
        "required_executable",
    ]


def test_dev_check_full_profile_matches_local_ci_order_and_commands(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    repo_root = _repo(tmp_path, monkeypatch)
    (repo_root / "dist").mkdir()
    (repo_root / "dist" / "uma.whl").touch()
    commands = []
    monkeypatch.setattr("uma.cli.development._find_executable", lambda name: name)
    monkeypatch.setattr(
        "uma.cli.development.importlib.util.find_spec",
        lambda name: SimpleNamespace(loader=object()),
    )

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["cwd"] == repo_root
        assert "shell" not in kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("uma.cli.development.subprocess.run", run)

    assert main(["--format", "json", "dev", "check", "--profile", "full"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert [check["name"] for check in result["data"]["checks"]] == [
        "pip-check",
        "pytest",
        "ruff",
        "bandit",
        "pip-audit",
        "build",
        "twine",
    ]
    assert _norm(commands) == _norm([
        [sys.executable, "-m", "pip", "check"],
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--tb=short",
            "--cov=uma",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ],
        [
            "ruff",
            "check",
            "uma/",
            "tests/",
            "--output-format=github",
        ],
        [
            "bandit",
            "--recursive",
            "uma/",
            "--severity-level",
            "medium",
            "--confidence-level",
            "medium",
            "--format",
            "txt",
        ],
        [
            "pip-audit",
            "--strict",
            "--vulnerability-service",
            "pypi",
            "--format",
            "columns",
        ],
        [sys.executable, "-m", "build"],
        [
            sys.executable,
            "-m",
            "twine",
            "check",
            "dist/uma.whl",
        ],
    ])


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


def test_dev_check_only_preserves_selected_profile_order(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "uma.cli.development._find_executable",
        lambda name: name,
    )
    monkeypatch.setattr(
        "uma.cli.development.importlib.util.find_spec",
        lambda name: SimpleNamespace(loader=object()),
    )
    monkeypatch.setattr(
        "uma.cli.development.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    assert (
        main(
            [
                "--format",
                "json",
                "dev",
                "check",
                "--profile",
                "full",
                "--only",
                "ruff,pytest,pip-check",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert [check["name"] for check in result["data"]["checks"]] == [
        "pip-check",
        "pytest",
        "ruff",
    ]


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
    assert "pip install ruff" in result["data"]["checks"][0]["stderr"]


def test_dev_check_non_missing_os_error_is_check_failure(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "uma.cli.development._find_executable",
        lambda name: name,
    )
    monkeypatch.setattr(
        "uma.cli.development.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("execution denied")
        ),
    )

    assert (
        main(["--format", "json", "dev", "check", "--only", "ruff"])
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert result["data"]["checks"][0]["status"] == "failed"
    assert "execution denied" in result["data"]["checks"][0]["stderr"]


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


def test_dev_check_continues_after_ordinary_failure(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _repo(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr("uma.cli.development._find_executable", lambda name: name)

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=1 if command[0] == "ruff" else 0,
            stdout="ordinary output",
            stderr="",
        )

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
            ]
        )
        == 1
    )

    result = json.loads(capsys.readouterr().out)
    assert [check["status"] for check in result["data"]["checks"]] == [
        "failed",
        "passed",
    ]
    assert len(commands) == 2


def test_dev_check_json_contains_arbitrary_captured_output(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _repo(tmp_path, monkeypatch)
    monkeypatch.setattr("uma.cli.development._find_executable", lambda name: name)
    arbitrary_stdout = 'quotes: "\' backslash: \\\\ newline:\nnull:\x00 snowman: ☃'
    arbitrary_stderr = "line one\nline two\r\n"
    monkeypatch.setattr(
        "uma.cli.development.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout=arbitrary_stdout,
            stderr=arbitrary_stderr,
        ),
    )

    assert main(["--format", "json", "dev", "check", "--only", "ruff"]) == 1

    result = json.loads(capsys.readouterr().out)
    check = result["data"]["checks"][0]
    assert check["stdout"] == arbitrary_stdout
    assert check["stderr"] == arbitrary_stderr


def test_dev_check_discovers_root_from_a_subdirectory(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    repo_root = _repo(tmp_path, monkeypatch)
    monkeypatch.chdir(repo_root / "uma")
    monkeypatch.setattr("uma.cli.development._find_executable", lambda name: name)

    def run(command, **kwargs):
        assert kwargs["cwd"] == repo_root
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("uma.cli.development.subprocess.run", run)

    assert main(["--format", "json", "dev", "check", "--only", "ruff"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data"]["repo_root"] == str(repo_root)


def test_dev_check_refuses_to_run_outside_source_checkout(
    capsys,
    monkeypatch,
) -> None:
    # Deliberately a system temp dir, not `tmp_path`: pytest's basetemp lives
    # inside the repo, so root discovery would walk up and find the real
    # checkout instead of failing.
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.chdir(directory)
        monkeypatch.setattr(
            "uma.cli.development._find_executable",
            lambda name: name,
        )
        monkeypatch.setattr(
            "uma.cli.development.subprocess.run",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("must not run")
            ),
        )

        try:
            assert (
                main(["--format", "json", "dev", "check", "--only", "ruff"])
                == 2
            )
            result = json.loads(capsys.readouterr().out)
            assert result["status"] == "error"
            assert "UMA source checkout" in result["errors"][0]["message"]
        finally:
            # Windows refuses to remove a directory that is a process's cwd,
            # so leave before TemporaryDirectory cleans up. No-op elsewhere.
            monkeypatch.chdir(original_cwd)


def test_dev_check_interruption_returns_130_and_valid_json(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _repo(tmp_path, monkeypatch)
    monkeypatch.setattr("uma.cli.development._find_executable", lambda name: name)
    monkeypatch.setattr(
        "uma.cli.development.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert main(["--format", "json", "dev", "check", "--only", "ruff"]) == 130
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert result["errors"] == [{"message": "interrupted"}]
