from datetime import datetime

import pytest

from uma.core.semantic.core import SemanticCore
from uma.types import Fact


class DummyStore:
    def __init__(self, facts):
        self._facts = facts

    async def list_facts_for_subject(self, subject, limit=None, owner_type=None, owner_id=None):
        return list(self._facts)


class DummyIngestor:
    def __init__(self, store):
        self.semantic_store = store


@pytest.mark.asyncio
async def test_fetch_more_facts_pages_deterministically_by_offset():
    now = datetime.utcnow()
    facts = [
        Fact(id="1", subject="user:u1", predicate="P", object="a", created_at=now, updated_at=now, source_ids=[], confidence=None, meta={}, salience=0.0, owner_type="user", owner_id="user:u1"),
        Fact(id="2", subject="user:u1", predicate="P", object="b", created_at=now, updated_at=now, source_ids=[], confidence=None, meta={}, salience=0.0, owner_type="user", owner_id="user:u1"),
        Fact(id="3", subject="user:u1", predicate="Q", object="c", created_at=now, updated_at=now, source_ids=[], confidence=None, meta={}, salience=0.0, owner_type="user", owner_id="user:u1"),
        Fact(id="4", subject="user:u1", predicate="P", object="d", created_at=now, updated_at=now, source_ids=[], confidence=None, meta={}, salience=0.0, owner_type="user", owner_id="user:u1"),
    ]

    core = SemanticCore(llm=None, embedder=None, semantic_store=DummyStore(facts))
    core.ingestor = DummyIngestor(DummyStore(facts))

    page1 = await core.fetch_more_facts("u1", "P", owner_type="user", owner_id="user:u1", k=2, offset=0)
    page2 = await core.fetch_more_facts("u1", "P", owner_type="user", owner_id="user:u1", k=2, offset=2)

    assert [f.id for f in page1] == ["1", "2"]
    assert [f.id for f in page2] == ["4"]

