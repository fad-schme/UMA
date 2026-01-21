from uma.adapters.graph.core import TemporalGraphCore
from uma.adapters.graph.base import GraphAdapter


class _StubAdapter(GraphAdapter):
    def run_query(self, cypher, params=None):
        return []

    def close(self):
        return None


def test_graph_core_rejects_invalid_labels_and_relations():
    core = TemporalGraphCore(_StubAdapter())
    core.add_entity("user:1", labels=["Good", "Bad-Label"])
    core.add_relationship("a", "LIKES", "b")
    core.add_relationship("a", "BAD-REL", "b")
