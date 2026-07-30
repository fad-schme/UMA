from __future__ import annotations

from typing import Any, Optional

from uma.adapters.graph.base import GraphAdapter


class RecordingGraphAdapter(GraphAdapter):
    """
    Test-only graph adapter that records Cypher queries and returns empty results.

    This uses UMA's normal graph adapter interface but avoids any external DB.
    """

    def __init__(self, **_kwargs: Any) -> None:
        self.queries: list[tuple[str, Optional[dict[str, Any]]]] = []
        self.next_results: list[list[dict[str, Any]]] = []

    def run_query(
        self, cypher: str, params: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        self.queries.append((str(cypher), dict(params) if isinstance(params, dict) else params))
        if self.next_results:
            return self.next_results.pop(0)
        return []

    def close(self) -> None:
        return
