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
        try:
            mem.shutdown()
        except Exception:
            # Shutdown is best-effort; tests use disabled graph by default.
            pass
