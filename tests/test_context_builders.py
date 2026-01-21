from uma.core.utils.context_pack_builder import ContextPackBuilder
from uma.core.utils.cot_memory_builder import CoTMemoryBuilder


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
        "semantic": [{"subject": "user:1", "predicate": "likes", "object": "tea"}],
        "procedural": [{"name": "skill", "plan": {"steps": ["a"]}}],
        "graph": [{"labels": ["Entity"], "properties": {"name": "x"}}],
    }

    pack = ContextPackBuilder.build("q", ctx)
    assert pack["working_memory"]
    assert pack["episodic"][0]["summary"] == "summary"
    assert pack["semantic"][0]["predicate"] == "likes"
    assert pack["procedural"][0]["name"] == "skill"


def test_cot_memory_builder_accepts_dicts_and_objects():
    ctx = {
        "semantic": [{"subject": "user:1", "predicate": "likes", "object": "tea"}],
        "episodic": [{"summary": "event"}],
        "procedural": [{"name": "skill", "plan": {"steps": ["s1"]}}],
        "graph": [],
    }

    cot = CoTMemoryBuilder.build(ctx)
    assert "user:1 likes tea" in cot["known_facts"]
    assert "event" in cot["relevant_events"]
    assert cot["available_skills"][0]["skill_name"] == "skill"
