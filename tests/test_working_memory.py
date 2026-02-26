from __future__ import annotations

import asyncio

from uma.adapters.llm.callable_adapter import CallableLLMAdapter
from uma.core.utils.config_types import WorkingMemorySettings
from uma.core.working_memory.core import WorkingMemoryCore

from tests.helpers.providers import fake_llm


class _MemoryClient:
    def __init__(self, wm_cfg: WorkingMemorySettings) -> None:
        self.working_memory_cfg = wm_cfg


def test_working_memory_chunked_compaction_produces_summary():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    wm_cfg = WorkingMemorySettings(
        max_tokens=120,
        warning_ratio=0.2,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = _MemoryClient(wm_cfg)
    wm = WorkingMemoryCore(llm=llm, memory_client=mem)

    user_id = "user:u1"
    for i in range(6):
        wm.append(user_id=user_id, role="user", content=f"msg {i} " + "word " * 8)

    asyncio.run(wm.compact(user_id=user_id))

    ctx = wm.get_context(user_id)
    assert ctx
    assert ctx[0].role == "summary"


def test_working_memory_emergency_prune_keeps_recent_messages():
    llm = CallableLLMAdapter(callable_fn=fake_llm, name="tests.fake_llm")
    wm_cfg = WorkingMemorySettings(
        max_tokens=20,
        warning_ratio=0.1,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = _MemoryClient(wm_cfg)
    wm = WorkingMemoryCore(llm=llm, memory_client=mem)

    user_id = "user:u2"
    for _i in range(12):
        wm.append(user_id=user_id, role="user", content="word " * 10)

    asyncio.run(wm.compact(user_id=user_id))

    ctx = wm.get_context(user_id)
    assert len(ctx) == 10
    assert all(msg.role != "summary" for msg in ctx)
