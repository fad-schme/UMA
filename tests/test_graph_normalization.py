from uma.adapters.graph.neo4j_adapter import Neo4jAdapter


def test_graph_normalization_handles_nested_lists():
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    data = {"items": [(1, 2), {"x": (3, 4)}]}
    out = adapter._normalize_record(data)
    assert out["items"] == [[1, 2], {"x": [3, 4]}]
