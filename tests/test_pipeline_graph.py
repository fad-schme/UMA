import asyncio

from uma3.core.pipeline import MemoryPipeline


class DummyHooks:
    async def run_before_turn(self, **kwargs):
        return None

    async def run_after_turn(self, **kwargs):
        return None


class DummyWM:
    def __init__(self):
        self.messages = []

    def append(self, **kwargs):
        self.messages.append(kwargs)

    async def compact(self, user_id):
        return None

    def get_context(self, user_id):
        return []


class DummyEpisode:
    def __init__(self, eid, user_id):
        self.id = eid
        self.user_id = user_id
        self.timestamp = _DummyTimestamp()
        self.summary = "summary"


class _DummyTimestamp:
    def isoformat(self):
        return "2024-01-01T00:00:00Z"


class DummyFact:
    def __init__(self, fid, subject, predicate, obj):
        self.id = fid
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.updated_at = _DummyTimestamp()
        self.confidence = 0.9


class DummyEpisodicCore:
    async def store_episode(self, **kwargs):
        return DummyEpisode("ep_current", kwargs.get("user_id"))


class DummySemanticCore:
    async def ingest(self, subject, text):
        return [DummyFact("f1", subject, "likes", "sushi")]


class DummyEpisodicStore:
    def __init__(self, current, prev):
        self._current = current
        self._prev = prev

    async def list_recent(self, user_id, n=2):
        return [self._current, self._prev]


class DummyGraphCore:
    def __init__(self):
        self.calls = []

    def add_episode(self, episode):
        self.calls.append(("add_episode", episode.id))

    def add_facts(self, facts):
        self.calls.append(("add_facts", [f.id for f in facts]))

    def link_episode_to_facts(self, episode, facts):
        self.calls.append(("link_episode_to_facts", episode.id, [f.id for f in facts]))

    def link_temporal(self, prev_ep, next_ep):
        self.calls.append(("link_temporal", prev_ep.id, next_ep.id))


class DummyMemory:
    def __init__(self):
        self.working_memory = DummyWM()
        self.episodic_core = DummyEpisodicCore()
        self.semantic_core = DummySemanticCore()
        self.graph_core = DummyGraphCore()
        self.hooks = DummyHooks()

        prev = DummyEpisode("ep_prev", "u1")
        current = DummyEpisode("ep_current", "u1")
        self.episodic_store = DummyEpisodicStore(current=current, prev=prev)


def test_pipeline_updates_graph_with_facts_and_temporal_links():
    mem = DummyMemory()
    pipeline = MemoryPipeline(memory_client=mem, hooks=mem.hooks)

    asyncio.run(
        pipeline.process_turn(
            user_id="u1",
            user_msg="hello",
            assistant_reply="hi",
        )
    )

    calls = mem.graph_core.calls
    assert ("add_episode", "ep_current") in calls
    assert any(c[0] == "add_facts" for c in calls)
    assert any(c[0] == "link_episode_to_facts" for c in calls)
    assert ("link_temporal", "ep_prev", "ep_current") in calls
