"""Predefined local development checks for UMA."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Check:
    name: str
    profiles: tuple[str, ...]
    command: tuple[str, ...]
    executable: Optional[str] = None
    module: Optional[str] = None


_CHECKS = (
    Check(
        "ruff",
        ("quick", "full"),
        ("ruff", "check", "uma/", "tests/"),
        executable="ruff",
    ),
    Check(
        "bandit",
        ("quick", "full"),
        (
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
        executable="bandit",
    ),
    Check(
        "security-tests",
        ("quick",),
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/test_security_*.py",
            "-q",
            "--tb=short",
        ),
        module="pytest",
    ),
    Check(
        "contract-tests",
        ("quick",),
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/test_configuration.py",
            "tests/test_vector_contract.py",
            "-q",
            "--tb=short",
        ),
        module="pytest",
    ),
    Check(
        "pip-check",
        ("full",),
        (sys.executable, "-m", "pip", "check"),
        module="pip",
    ),
    Check(
        "pytest",
        ("full",),
        (sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"),
        module="pytest",
    ),
    Check(
        "pip-audit",
        ("full",),
        (
            "pip-audit",
            "--strict",
            "--vulnerability-service",
            "pypi",
            "--format",
            "columns",
        ),
        executable="pip-audit",
    ),
    Check(
        "build",
        ("full",),
        (sys.executable, "-m", "build"),
        module="build",
    ),
    Check(
        "twine",
        ("full",),
        (sys.executable, "-m", "twine", "check", "dist/*"),
        module="twine",
    ),
)


def _find_repo_root() -> Path:
    current = Path.cwd().resolve()
    for path in (current, *current.parents):
        if (
            (path / ".git").exists()
            and (path / "pyproject.toml").is_file()
            and (path / "uma").is_dir()
        ):
            return path
    raise ValueError("dev check must run from an UMA source checkout")


def _find_executable(name: str) -> Optional[str]:
    environment_executable = Path(sys.executable).with_name(name)
    if environment_executable.is_file():
        return str(environment_executable)
    return shutil.which(name)


def _parse_names(value: Optional[str]) -> set[str]:
    if not value:
        return set()
    return {name.strip() for name in value.split(",") if name.strip()}


def _expanded_command(check: Check, repo_root: Path) -> list[str]:
    command: list[str] = []
    for argument in check.command:
        if "*" not in argument:
            command.append(argument)
            continue
        matches = sorted(repo_root.glob(argument))
        if not matches:
            raise ValueError(f"{check.name}: no files match {argument!r}")
        command.extend(str(path.relative_to(repo_root)) for path in matches)
    return command


def run_checks(
    *,
    profile: str,
    only: Optional[str],
    skip: Optional[str],
    list_only: bool,
    fail_fast: bool,
) -> tuple[dict[str, Any], str, str, int]:
    available = {check.name for check in _CHECKS}
    only_names = _parse_names(only)
    skip_names = _parse_names(skip)
    unknown = (only_names | skip_names) - available
    if unknown:
        raise ValueError(f"unknown development checks: {', '.join(sorted(unknown))}")

    if list_only:
        checks = [
            {"name": check.name, "profiles": list(check.profiles)}
            for check in _CHECKS
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

    selected = [
        check
        for check in _CHECKS
        if (check.name in only_names if only_names else profile in check.profiles)
        and check.name not in skip_names
    ]
    if not selected:
        raise ValueError("no development checks selected")

    repo_root = _find_repo_root()
    results: list[dict[str, Any]] = []
    for check in selected:
        module_spec = (
            importlib.util.find_spec(check.module)
            if check.module
            else None
        )
        executable = (
            _find_executable(check.executable)
            if check.executable
            else None
        )
        missing = (check.executable and executable is None) or (
            check.module
            and (module_spec is None or module_spec.loader is None)
        )
        if missing:
            results.append(
                {
                    "name": check.name,
                    "status": "missing",
                    "command": list(check.command),
                    "return_code": None,
                    "duration_ms": 0,
                    "stdout": "",
                    "stderr": "Required development tool is not installed. Install UMA's dev dependencies.",
                }
            )
            if fail_fast:
                break
            continue

        try:
            command = _expanded_command(check, repo_root)
            if executable:
                command[0] = executable
        except ValueError as exc:
            results.append(
                {
                    "name": check.name,
                    "status": "failed",
                    "command": list(check.command),
                    "return_code": None,
                    "duration_ms": 0,
                    "stdout": "",
                    "stderr": str(exc),
                }
            )
            if fail_fast:
                break
            continue

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            results.append(
                {
                    "name": check.name,
                    "status": "missing",
                    "command": command,
                    "return_code": None,
                    "duration_ms": round(
                        (time.monotonic() - started) * 1000
                    ),
                    "stdout": "",
                    "stderr": str(exc),
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
                "command": command,
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
            f"[{result['status']}] {result['name']} ({result['duration_ms']} ms)"
        )
        if result["status"] != "passed":
            output = (result["stdout"] + result["stderr"]).strip()
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
