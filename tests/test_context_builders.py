from uma.retrieve.context_pack_builder import ContextPackBuilder
from uma.retrieve.cot_memory_builder import CoTMemoryBuilder


class DummyMsg:
    def __init__(self):
        self.role = "user"
        self.content = "hello"
        self.metadata = {}
        self.token_estimate = 3


def test_context_pack_builder_accepts_dicts_and_objects():
    ctx = {
        "working_memory": [DummyMsg(), {"role": "assistant", "text": "ok", "tokens": 2}],
        "episodic": [{"id": "e1", "summary": "summary"}],
        "facts": [{"subject": "user:1", "predicate": "likes", "object": "tea"}],
        "skills": [{"name": "skill", "plan": {"steps": ["a"]}}],
        "graph": [{"labels": ["Entity"], "properties": {"name": "x"}}],
    }

    pack = ContextPackBuilder.build("q", ctx)
    assert pack["working_memory"]
    assert pack["episodic"][0]["summary"] == "summary"
    assert pack["facts"][0]["predicate"] == "likes"
    assert pack["skills"][0]["name"] == "skill"
    assert "id" in pack["facts"][0]
    assert "id" in pack["skills"][0]


def test_context_pack_builder_dedupes_by_id():
    ctx = {
        "episodic": [{"id": "e1", "summary": "a"}, {"id": "e1", "summary": "b"}],
        "facts": [{"id": "f1", "subject": "s", "predicate": "p", "object": "o"}, {"id": "f1", "subject": "s2", "predicate": "p2", "object": "o2"}],
        "chunks": [{"id": "c1", "doc_id": "d", "text": "x"}, {"id": "c1", "doc_id": "d", "text": "y"}],
        "skills": [{"id": "k1", "name": "s1"}, {"id": "k1", "name": "s2"}],
        "graph": [{"id": "g1", "node": "x"}, {"id": "g1", "node": "y"}],
    }
    pack = ContextPackBuilder.build("q", ctx)
    assert len(pack["episodic"]) == 1
    assert len(pack["facts"]) == 1
    assert len(pack["chunks"]) == 1
    assert len(pack["skills"]) == 1
    assert len(pack["graph"]) == 1


def test_context_pack_builder_preserves_episode_provenance_in_serialized_output():
    ctx = {
        "episodic": [
            {
                "id": "ep-import-1",
                "timestamp": "2026-05-01T10:00:00+00:00",
                "summary": "Imported daily diary entry",
                "tags": ["daily_diary"],
                "owner_type": "user",
                "owner_id": "user:u1",
                "meta": {
                    "source_type": "daily_diary",
                    "source_file": "/tmp/diary.md",
                    "diary_date": "2026-05-01",
                    "import_mode": "bootstrap",
                },
            }
        ]
    }

    pack = ContextPackBuilder.build("q", ctx)
    episode = pack["episodic"][0]

    assert episode["kind"] == "episodic_event"
    assert episode["kb_lane"] == "episodic"
    assert episode["meta"]["provenance"]["episode_id"] == "ep-import-1"
    assert episode["meta"]["provenance"]["source_file"] == "/tmp/diary.md"
    assert episode["meta"]["provenance"]["diary_date"] == "2026-05-01"
    assert episode["meta"]["provenance"]["import_mode"] == "bootstrap"


def test_cot_memory_builder_accepts_dicts_and_objects():
    ctx = {
        "facts": [{"subject": "user:1", "predicate": "likes", "object": "tea"}],
        "episodic": [{"summary": "event"}],
        "skills": [{"name": "skill", "plan": {"steps": ["s1"]}}],
        "graph": [],
    }

    cot = CoTMemoryBuilder.build(ctx)
    assert "user:1 likes tea" in cot["known_facts"]
    assert "event" in cot["relevant_events"]
    assert cot["available_skills"][0]["skill_name"] == "skill"
