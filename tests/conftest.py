from __future__ import annotations

import pytest_asyncio

from tests.helpers.runtime import init_uma_for_tests


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
