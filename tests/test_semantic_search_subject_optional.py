import pytest

from uma.core.semantic.core import SemanticCore


class DummyStore:
    def __init__(self, facts):
        self._facts = facts

    async def search(self, **kwargs):
        subject = kwargs.get("subject")
        if subject is None:
            return list(self._facts)
        return [f for f in self._facts if f.get("subject") == subject]

    async def search_text(self, **kwargs):
        return []


@pytest.mark.asyncio
async def test_semantic_search_subject_optional():
    facts = [
        {"id": "f1", "subject": "entity:zero_trust", "predicate": "principle", "object": "least privilege"},
        {"id": "f2", "subject": "entity:cloud_security", "predicate": "principle", "object": "segmentation"},
        {"id": "f3", "subject": "user:local", "predicate": "remembered", "object": "note"},
    ]
    core = SemanticCore(llm=None, embedder=None, semantic_store=DummyStore(facts))
    core.ingestor.semantic_store = DummyStore(facts)

    # Subject=None => corpus-wide within owner scope.
    all_facts = await core.search(
        subject=None,
        query_embedding=[0.0],
        owner_type="agent",
        owner_id="agent-default",
        k=10,
    )
    assert len(all_facts) == 3

    # Subject=user:<id> => filtered.
    user_facts = await core.search(
        subject="user:local",
        query_embedding=[0.0],
        owner_type="agent",
        owner_id="agent-default",
        k=10,
    )
    assert len(user_facts) == 1
    assert user_facts[0]["subject"] == "user:local"
