import asyncio
import yaml
from tempfile import TemporaryDirectory
from pathlib import Path

from uma.core.uma_memory import UMAMemory


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
    def __init__(self, eid, owner_id):
        self.id = eid
        self.user_id = owner_id
        self.owner_type = "user"
        self.owner_id = owner_id
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
        return DummyEpisode("ep_current", kwargs.get("owner_id"))

    async def list_recent(self, owner_type, owner_id, n=2):
        return [DummyEpisode("ep_current", owner_id), DummyEpisode("ep_prev", owner_id)]


class DummySemanticCore:
    async def ingest(self, subject, text, *, extra_meta=None):
        return [DummyFact("f1", subject, "likes", "sushi")]



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


def test_pipeline_updates_graph_with_facts_and_temporal_links():
    with TemporaryDirectory() as tmp:
        cfg = {
            "storage": {
                "db_root": str(Path(tmp) / "db") + "/",
                "sql_backend": "sqlite",
                "vector_backend": "inmemory",
                "graph_backend": "disabled",
            },
            "working_memory": {
                "max_tokens": 256,
                "warning_ratio": 0.7,
                "hard_limit_ratio": 0.95,
                "chunk_size": 10,
            },
            "embedding": {
                "provider": "tests.test_rebuild_indexes:fake_embed",
                "model": "x",
                "dimension": 3,
            },
            "llm": {
                "provider": "tests.test_rebuild_indexes:fake_llm",
                "model": "x",
            },
            "retrieval": {
                "max_episodes": 5,
                "max_facts": 5,
                "max_skills": 5,
                "max_graph_items": 5,
            },
            "consolidation": {
                "enabled": True,
                "cluster_similarity": 0.75,
                "max_episodes_per_cycle": 50,
                "prune_min_fact_salience": 0.2,
            },
            "features": {
                "load": [],
            },
        }
        cfg_path = Path(tmp) / "uma_test.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        mem = UMAMemory.from_yaml(str(cfg_path))
        mem.initialize()

        mem.working_memory = DummyWM()
        mem.episodic_core = DummyEpisodicCore()
        mem.semantic_core = DummySemanticCore()
        mem.graph_core = DummyGraphCore()
        mem.hooks = DummyHooks()

        mem.pipeline.hooks = mem.hooks

        asyncio.run(
            mem.process_turn(
                user_id="user:u1",
                user_msg="hello",
                assistant_reply="hi",
            )
        )

        calls = mem.graph_core.calls
        assert ("add_episode", "ep_current") in calls
        assert any(c[0] == "add_facts" for c in calls)
        assert any(c[0] == "link_episode_to_facts" for c in calls)
        assert ("link_temporal", "ep_prev", "ep_current") in calls
