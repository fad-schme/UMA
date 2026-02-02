from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from uma.adapters.graph.base import GraphAdapter
from uma.core.graph.core import TemporalGraphCore
from uma.core.graph.updater import GraphUpdater
from uma.types_episode import Episode


class RecordingAdapter(GraphAdapter):
    """
    Records Cypher queries and parameters for assertions.
    This adapter does NOT execute against a real graph backend.
    """

    def __init__(self):
        self.queries: List[tuple[str, Dict[str, Any]]] = []

    def run_query(self, cypher: str, params: Dict[str, Any] | None = None):
        self.queries.append((cypher, params or {}))
        return []

    def close(self):
        pass


def test_insert_fact_triplet_sanitizes_predicate_and_stamps_owner():
    adapter = RecordingAdapter()
    core = TemporalGraphCore(adapter)

    core.insert_fact_triplet(
        fact_id="f1",
        subject="user:u1",
        predicate="Bad-REL!!",  # must be sanitized
        object="tea",
        owner_type="user",
        owner_id="user:u1",
        source_chunk_id="c1",
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
    )

    assert adapter.queries, "Expected Cypher execution"

    cypher, params = adapter.queries[0]

    # Predicate must be sanitized to a safe relationship type
    assert "BAD_REL" in cypher or "RELATES_TO" in cypher

    # Ownership and provenance are stamped as top-level parameters
    assert params.get("owner_type") == "user"
    assert params.get("owner_id") == "user:u1"
    assert params.get("fact_id") == "f1"
    assert params.get("source_chunk_id") == "c1"


def test_episode_edges_have_ownership():
    adapter = RecordingAdapter()
    graph_core = TemporalGraphCore(adapter)
    updater = GraphUpdater(graph_core)

    ep = Episode(
        id="ep1",
        timestamp=datetime.utcnow(),
        summary="Test episode",
        user_id="u1",
        owner_type="user",
        owner_id="user:u1",
    )

    updater.add_episode_node(ep)

    assert adapter.queries, "No Cypher executed for episode insertion"

    # Find the HAS_EPISODE query and validate ownership stamping
    matched = False
    for cypher, params in adapter.queries:
        if "HAS_EPISODE" in cypher:
            matched = True
            assert params.get("owner_type") == "user"
            assert params.get("owner_id") == "user:u1"
            break

    assert matched, "Expected HAS_EPISODE relationship write"