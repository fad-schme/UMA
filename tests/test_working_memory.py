import asyncio

from uma.core.working_memory.core import WorkingMemoryCore


class DummyLLM:
    def __init__(self):
        self.call_count = 0
        self.last_messages = None

    async def generate(self, messages, max_tokens=256, temperature=0.0, **kwargs):
        self.call_count += 1
        self.last_messages = messages
        return "summary"

class DummyWMConfig:
    def __init__(
        self,
        max_tokens,
        warning_ratio,
        hard_limit_ratio,
        chunk_size,
        keep_recent_messages,
        keep_recent_token_fraction,
    ):
        self.max_tokens = max_tokens
        self.warning_ratio = warning_ratio
        self.hard_limit_ratio = hard_limit_ratio
        self.chunk_size = chunk_size
        self.keep_recent_messages = keep_recent_messages
        self.keep_recent_token_fraction = keep_recent_token_fraction


class DummyMemoryClient:
    def __init__(self, wm_cfg):
        self.working_memory_cfg = wm_cfg


def test_working_memory_chunked_compaction():
    llm = DummyLLM()
    wm_cfg = DummyWMConfig(
        max_tokens=120,
        warning_ratio=0.2,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = DummyMemoryClient(wm_cfg)
    wm = WorkingMemoryCore(
        llm=llm,
        memory_client=mem,
    )

    user_id = "u1"
    # Add enough content to trigger compaction and chunking
    for i in range(6):
        wm.append(user_id=user_id, role="user", content=f"msg {i} " + "word " * 8)

    asyncio.run(wm.compact(user_id=user_id))

    ctx = wm.get_context(user_id)
    assert ctx
    assert ctx[0].role == "summary"
    # With chunking, summarizer should be called more than once
    assert llm.call_count >= 2


def test_working_memory_emergency_prune():
    llm = DummyLLM()
    wm_cfg = DummyWMConfig(
        max_tokens=20,
        warning_ratio=0.1,
        hard_limit_ratio=0.9,
        chunk_size=2,
        keep_recent_messages=1,
        keep_recent_token_fraction=0.0,
    )
    mem = DummyMemoryClient(wm_cfg)
    wm = WorkingMemoryCore(
        llm=llm,
        memory_client=mem,
    )

    user_id = "u2"
    # Force emergency prune path (> 2x max_tokens).
    for i in range(12):
        wm.append(user_id=user_id, role="user", content="word " * 10)

    asyncio.run(wm.compact(user_id=user_id))

    ctx = wm.get_context(user_id)
    assert len(ctx) == 10
    assert all(msg.role != "summary" for msg in ctx)
