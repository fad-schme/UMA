import pytest

from uma.core.semantic.core import SemanticCore


class DummyStore:
    def __init__(self, facts):
        self._facts = facts
        self.last_search_kwargs = None

    async def search(self, **kwargs):
        self.last_search_kwargs = dict(kwargs)
        # SemanticCore no longer passes/accepts subject-gating; store should treat this as corpus-wide.
        return list(self._facts)

    async def lexical_search(self, **kwargs):
        return []


@pytest.mark.asyncio
async def test_semantic_search_subject_optional():
    facts = [
        {"id": "f1", "subject": "entity:zero_trust", "predicate": "principle", "object": "least privilege"},
        {"id": "f2", "subject": "entity:cloud_security", "predicate": "principle", "object": "segmentation"},
        {"id": "f3", "subject": "user:local", "predicate": "remembered", "object": "note"},
    ]
    store = DummyStore(facts)
    core = SemanticCore(llm=None, embedder=None, semantic_store=store)
    core.store = store

    all_facts = await core.search(query_embedding=[0.0], owner_type="agent", owner_id="agent-default", k=10)
    assert len(all_facts) == 3
    assert store.last_search_kwargs is not None
    assert "subject" not in store.last_search_kwargs

    # Subject filters are ignored (ownership-only retrieval).
    filtered = await core.search(
        query_embedding=[0.0],
        owner_type="agent",
        owner_id="agent-default",
        k=10,
        filters={"subject": "user:local"},
    )
    assert len(filtered) == 3
