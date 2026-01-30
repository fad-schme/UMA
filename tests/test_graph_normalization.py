import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXTENSIONS = os.path.join(ROOT, "extensions")
if EXTENSIONS not in sys.path:
    sys.path.insert(0, EXTENSIONS)

from graph.neo4j_adapter import Neo4jAdapter


def test_graph_normalization_handles_nested_lists():
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    data = {"items": [(1, 2), {"x": (3, 4)}]}
    out = adapter._normalize_record(data)
    assert out["items"] == [[1, 2], {"x": [3, 4]}]
