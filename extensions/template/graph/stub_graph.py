"""
Template graph adapter.

Implement a GraphAdapter-compatible class.
"""

from __future__ import annotations

from typing import Any, Dict, List

from uma.adapters.graph.base import GraphAdapter


class ExampleGraphAdapter(GraphAdapter):
    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError("Implement GraphAdapter for your backend")

    def run_query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None
