"""Predefined development checks for UMA."""

from __future__ import annotations

import glob
import importlib.util
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckDefinition:
    """A development check whose command is controlled by UMA."""

    name: str
    argv: tuple[str, ...]
    profiles: frozenset[str]
    required_executable: str | None


# Keep command arguments in one place so local checks cannot drift from CI.
CHECK_DEFINITIONS = (
    CheckDefinition(
        name="ruff",
        argv=(
            "ruff",
            "check",
            "uma/",
            "tests/",
            "--output-format=github",
        ),
        profiles=frozenset({"quick", "full"}),
        required_executable="ruff",
    ),
    CheckDefinition(
        name="bandit",
        argv=(
            "bandit",
            "--recursive",
            "uma/",
            "--severity-level",
            "medium",
            "--confidence-level",
            "medium",
            "--format",
            "txt",
        ),
        profiles=frozenset({"quick", "full"}),
        required_executable="bandit",
    ),
    CheckDefinition(
        name="security-tests",
        argv=(
            sys.executable,
            "-m",
            "pytest",
            "tests/test_security_*.py",
            "-q",
            "--tb=short",
        ),
        profiles=frozenset({"quick"}),
        required_executable=None,
    ),
    CheckDefinition(
        name="contract-tests",
        argv=(
            sys.executable,
            "-m",
            "pytest",
            "tests/test_configuration.py",
            "tests/test_vector_contract.py",
            "-q",
            "--tb=short",
        ),
        profiles=frozenset({"quick"}),
        required_executable=None,
    ),
    CheckDefinition(
        name="pip-check",
        argv=(sys.executable, "-m", "pip", "check"),
        profiles=frozenset({"full"}),
        required_executable=None,
    ),
    CheckDefinition(
        name="pytest",
        argv=(
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--tb=short",
            "--cov=uma",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ),
        profiles=frozenset({"full"}),
        required_executable=None,
    ),
    CheckDefinition(
        name="pip-audit",
        argv=(
            "pip-audit",
            "--strict",
            "--vulnerability-service",
            "pypi",
            "--format",
            "columns",
        ),
        profiles=frozenset({"full"}),
        required_executable="pip-audit",
    ),
    CheckDefinition(
        name="build",
        argv=(sys.executable, "-m", "build"),
        profiles=frozenset({"full"}),
        required_executable=None,
    ),
    CheckDefinition(
        name="twine",
        argv=(sys.executable, "-m", "twine", "check", "dist/*"),
        profiles=frozenset({"full"}),
        required_executable=None,
    ),
)

_PROFILE_ORDER = {
    "quick": (
        "ruff",
        "bandit",
        "security-tests",
        "contract-tests",
    ),
    "full": (
        "pip-check",
        "pytest",
        "ruff",
        "bandit",
        "pip-audit",
        "build",
        "twine",
    ),
}

_INSTALL_REQUIREMENTS = {
    "bandit": "bandit[toml]",
    "build": "build",
    "pip-audit": "pip-audit",
    "pytest": "pytest",
    "ruff": "ruff",
    "twine": "twine",
}


def _find_repo_root() -> Path:
    current = Path.cwd().resolve()
    for path in (current, *current.parents):
        if (
            (path / ".git").exists()
            and (path / "pyproject.toml").is_file()
            and (path / "uma" / "__init__.py").is_file()
            and _pyproject_declares_uma(path / "pyproject.toml")
        ):
            return path
    raise ValueError("dev check must run from an UMA source checkout")


def _pyproject_declares_uma(pyproject_path: Path) -> bool:
    """Confirm the checkout identity without requiring a TOML backport."""

    section = ""
    try:
        lines = pyproject_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if section != "[project]" or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key == "name":
            return value.strip("\"'") == "uma"
    return False


def _find_executable(name: str) -> str | None:
    environment_executable = Path(sys.executable).with_name(name)
    if environment_executable.is_file():
        return str(environment_executable)
    return shutil.which(name)


def _python_module(check: CheckDefinition) -> str | None:
    if len(check.argv) >= 3 and check.argv[1] == "-m":
        return check.argv[2]
    return None


def _missing_tool(check: CheckDefinition) -> str | None:
    if check.required_executable:
        if _find_executable(check.required_executable) is None:
            return check.required_executable
        return None

    module = _python_module(check)
    if module is None:
        return None
    module_spec = importlib.util.find_spec(module)
    if module_spec is None or module_spec.loader is None:
        return module
    return None


def _installation_hint(tool: str) -> str:
    if tool == "pip":
        return (
            "Repair this Python environment so "
            f"{sys.executable} -m pip is available."
        )
    requirement = _INSTALL_REQUIREMENTS.get(tool, tool)
    return (
        "Install it explicitly with: "
        f"{sys.executable} -m pip install {requirement}"
    )


def _parse_names(value: str | None) -> set[str]:
    if not value:
        return set()
    return {name.strip() for name in value.split(",") if name.strip()}


def _expanded_argv(
    check: CheckDefinition,
    repo_root: Path,
) -> list[str]:
    argv: list[str] = []
    for argument in check.argv:
        if not glob.has_magic(argument):
            argv.append(argument)
            continue
        matches = sorted(repo_root.glob(argument))
        if not matches:
            raise ValueError(f"{check.name}: no files match {argument!r}")
        argv.extend(str(path.relative_to(repo_root)) for path in matches)

    if (
        check.required_executable
        and argv[0] == check.required_executable
    ):
        executable = _find_executable(check.required_executable)
        if executable is not None:
            argv[0] = executable
    return argv


def _select_checks(
    profile: str,
    only_names: set[str],
    skip_names: set[str],
) -> list[CheckDefinition]:
    definitions = {check.name: check for check in CHECK_DEFINITIONS}
    if only_names:
        ordered_names = [
            name
            for name in _PROFILE_ORDER[profile]
            if name in only_names
        ]
        ordered_names.extend(
            check.name
            for check in CHECK_DEFINITIONS
            if check.name in only_names
            and check.name not in ordered_names
        )
    else:
        ordered_names = list(_PROFILE_ORDER[profile])
    return [
        definitions[name]
        for name in ordered_names
        if name not in skip_names
    ]


def _missing_result(
    check: CheckDefinition,
    tool: str,
    duration_ms: int = 0,
    detail: str | None = None,
) -> dict[str, Any]:
    message = f"Required development tool {tool!r} is not installed."
    if detail:
        message = f"{message} {detail}"
    return {
        "name": check.name,
        "status": "missing",
        "command": list(check.argv),
        "return_code": None,
        "duration_ms": duration_ms,
        "stdout": "",
        "stderr": f"{message} {_installation_hint(tool)}",
    }


def _failed_setup_result(
    check: CheckDefinition,
    message: str,
) -> dict[str, Any]:
    return {
        "name": check.name,
        "status": "failed",
        "command": list(check.argv),
        "return_code": None,
        "duration_ms": 0,
        "stdout": "",
        "stderr": message,
    }


def run_checks(
    *,
    profile: str,
    only: str | None,
    skip: str | None,
    list_only: bool,
    fail_fast: bool,
) -> tuple[dict[str, Any], str, str, int]:
    available = {check.name for check in CHECK_DEFINITIONS}
    only_names = _parse_names(only)
    skip_names = _parse_names(skip)
    unknown = (only_names | skip_names) - available
    if unknown:
        raise ValueError(
            f"unknown development checks: {', '.join(sorted(unknown))}"
        )

    if list_only:
        checks = [
            {
                "name": check.name,
                "profiles": [
                    profile_name
                    for profile_name in _PROFILE_ORDER
                    if profile_name in check.profiles
                ],
            }
            for check in CHECK_DEFINITIONS
        ]
        return (
            {"checks": checks},
            "\n".join(
                f"{check['name']}: {', '.join(check['profiles'])}"
                for check in checks
            ),
            "ok",
            0,
        )

    selected = _select_checks(profile, only_names, skip_names)
    if not selected:
        raise ValueError("no development checks selected")

    repo_root = _find_repo_root()
    results: list[dict[str, Any]] = []
    for check in selected:
        missing_tool = _missing_tool(check)
        if missing_tool:
            results.append(_missing_result(check, missing_tool))
            if fail_fast:
                break
            continue

        try:
            argv = _expanded_argv(check, repo_root)
        except ValueError as exc:
            results.append(_failed_setup_result(check, str(exc)))
            if fail_fast:
                break
            continue

        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=repo_root,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            duration_ms = round((time.monotonic() - started) * 1000)
            tool = check.required_executable or _python_module(check) or argv[0]
            results.append(
                _missing_result(
                    check,
                    tool,
                    duration_ms=duration_ms,
                    detail=str(exc),
                )
            )
            if fail_fast:
                break
            continue
        except OSError as exc:
            duration_ms = round((time.monotonic() - started) * 1000)
            results.append(
                {
                    **_failed_setup_result(check, str(exc)),
                    "command": argv,
                    "duration_ms": duration_ms,
                }
            )
            if fail_fast:
                break
            continue

        duration_ms = round((time.monotonic() - started) * 1000)
        passed = completed.returncode == 0
        results.append(
            {
                "name": check.name,
                "status": "passed" if passed else "failed",
                "command": argv,
                "return_code": completed.returncode,
                "duration_ms": duration_ms,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if fail_fast and not passed:
            break

    missing_tool = any(result["status"] == "missing" for result in results)
    failed = any(result["status"] == "failed" for result in results)
    status = "failed" if missing_tool or failed else "ok"
    lines = [f"UMA development checks ({profile})"]
    for result in results:
        lines.append(
            f"[{result['status']}] {result['name']} "
            f"({result['duration_ms']} ms)"
        )
        if result["status"] != "passed":
            output = "\n".join(
                part
                for part in (result["stdout"], result["stderr"])
                if part
            ).strip()
            if output:
                lines.append(output)
    lines.append(f"Overall: {status}")
    return (
        {
            "profile": profile,
            "repo_root": str(repo_root),
            "checks": results,
        },
        "\n".join(lines),
        status,
        3 if missing_tool else 1 if failed else 0,
    )
