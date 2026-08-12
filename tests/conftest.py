from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest_asyncio

from tests.helpers.runtime import init_uma_for_tests

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLEANDIR = _REPO_ROOT / "scripts" / "cleandir.py"


def pytest_sessionfinish(session, exitstatus) -> None:
    """Sweep regenerable caches after every test run.

    Runs `scripts/cleandir.py --caches-only`, so build outputs and coverage
    artifacts survive — a `pytest --cov` run must not delete the report it
    just produced.

    `scripts/` is gitignored, so the script is absent in CI and in fresh
    clones. That is not an error: skip silently rather than failing a green
    test run over a local convenience. Cleanup failures are reported but never
    change the session's exit status.
    """
    if not _CLEANDIR.is_file():
        return
    try:
        subprocess.run(
            [sys.executable, str(_CLEANDIR), "--caches-only", "--root", str(_REPO_ROOT)],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"cleandir sweep skipped: {exc}", file=sys.stderr)


@pytest_asyncio.fixture
async def uma_memory(tmp_path):
    mem = await init_uma_for_tests(
        tmp_path,
        graph_backend="tests.helpers.graph_adapter:RecordingGraphAdapter",
        graph_config={},
    )
    try:
        yield mem
    finally:
        # Phase 5: promotion is fire-and-forget. Drain any lingering
        # background tasks before shutdown so a test that scheduled a
        # promotion but did not explicitly await it cannot leave a task
        # bleeding into the next test's fixture teardown.
        pipeline = getattr(mem, "pipeline", None)
        if pipeline is not None and hasattr(pipeline, "await_pending_background"):
            try:
                await pipeline.await_pending_background()
            except Exception:
                # Best-effort drain; do not let teardown mask a test
                # failure with a fixture-level exception.
                pass
        try:
            mem.shutdown()
        except Exception:
            # Shutdown is best-effort; tests use disabled graph by default.
            pass
