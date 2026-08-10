"""Every shipped subpackage must import on its own in a fresh interpreter.

Import cycles are invisible to the rest of the suite: conftest imports
``uma.api`` first, which populates ``uma.retrieve`` and ``uma.memory`` in a
working order, so a module that cannot be imported *first* still passes
everywhere else. Only a clean interpreter per module exposes that, hence the
subprocess per case.

Regression origin: ``import uma.memory`` raised ImportError from a cycle
through ``uma.retrieve.__init__`` -> RLM controller -> ``uma.memory.chunk.core``.
"""
from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import uma

# Discovered rather than hardcoded so a new subpackage is covered on arrival.
# ``__main__`` runs the CLI on import by design and would exit non-zero here.
# ``iter_modules`` skips implicit namespace packages (``uma.stores`` has no
# ``__init__.py``), so directories holding modules are collected separately.
_ROOT = Path(uma.__path__[0])
_TOP_LEVEL = sorted(
    {
        f"uma.{info.name}"
        for info in pkgutil.iter_modules(uma.__path__)
        if info.name != "__main__"
    }
    | {
        f"uma.{d.name}"
        for d in _ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and any(d.glob("*.py"))
    }
)


def test_discovery_found_the_known_subpackages() -> None:
    """Guard the parametrization itself: silent discovery of nothing would
    turn every case below into a vacuous pass."""
    assert set(_TOP_LEVEL) >= {
        "uma.adapters",
        "uma.api",
        "uma.cli",
        "uma.common",
        "uma.ingest",
        "uma.memory",
        "uma.retrieve",
        "uma.stores",
        "uma.version",
    }


@pytest.mark.parametrize("module", _TOP_LEVEL)
def test_module_imports_as_the_first_uma_import(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"`import {module}` failed as the first UMA import — most likely a new "
        f"import cycle between subsystems.\n{result.stderr.strip()}"
    )
