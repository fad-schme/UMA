"""
uma3.adapters.graph.base
========================

GraphAdapter — Abstract graph backend interface for UMA-3.

This module defines the common interface that all graph adapters must
implement (e.g., Neo4jAdapter, MemgraphAdapter).

Design goals
------------
- Keep UMA-3 core logic independent of any specific graph database.
- Allow plug-and-play replacement of graph backends.
- Provide a minimal, expressive API (`run_query`, `close`).

Coding Agent Instructions
-------------------------
- Do NOT put application or domain logic here; this is infrastructure.
- Concrete adapters (Neo4j, Memgraph, etc.) MUST subclass GraphAdapter.
- If you add async support later, consider a separate AsyncGraphAdapter.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GraphAdapter(abc.ABC):
    """
    Abstract graph backend interface.

    All UMA-3 graph drivers (Neo4j, Memgraph, etc.) must implement this.

    Required Methods
    ----------------
    - run_query(cypher: str, params: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]
    - close() -> None
    """

    @abc.abstractmethod
    def run_query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a graph query and return a list of result rows as dicts.

        Concrete implementations MUST:
        - Catch and log any driver-specific exceptions.
        - Return [] on errors, NOT raise to the caller.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """
        Cleanly close all underlying resources (connections, sessions, etc.).

        Concrete implementations MUST:
        - Be idempotent (safe to call multiple times).
        - Catch and log errors internally.
        """
        raise NotImplementedError