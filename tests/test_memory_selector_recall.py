from uma.core.retrieval.policy import RetrievalPolicy
from uma.core.retrieval.selector import MemorySelector


class MockFact:
    def __init__(self, id_, salience, confidence, owner_type):
        self.id = id_
        self.meta = {"salience": salience}
        self.confidence = confidence
        self.owner_type = owner_type


def test_memory_selector_recall_does_not_reorder_facts():
    # query contains "remember" → recall_score > 0
    policy = RetrievalPolicy("do you remember when we talked about X?")
    selector = MemorySelector(
        max_episodes=3,
        max_facts=5,
        max_skills=2,
        max_graph_items=2,
    )

    # Create user fact (lower base score) and agent fact (higher base)
    user_fact = MockFact("u1", salience=0.6, confidence=0.6, owner_type="user")
    agent_fact = MockFact("a1", salience=0.7, confidence=0.7, owner_type="agent")

    results = selector.select({"facts": [agent_fact, user_fact]}, policy=policy)
    facts = results["facts"]

    # Ordering is driven by base score only (no scope bias)
    assert facts[0].id == "a1"


def test_memory_selector_prefers_agent_without_recall():
    # no recall keywords
    policy = RetrievalPolicy("how do I configure X for performance?")
    selector = MemorySelector(
        max_episodes=3,
        max_facts=5,
        max_skills=2,
        max_graph_items=2,
    )

    user_fact = MockFact("u1", salience=0.1, confidence=0.5, owner_type="user")
    agent_fact = MockFact("a1", salience=0.9, confidence=0.9, owner_type="agent")

    results = selector.select({"facts": [user_fact, agent_fact]}, policy=policy)
    facts = results["facts"]

    # Without recall intent, agent_fact should be first
    assert facts[0].id == "a1"
