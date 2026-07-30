"""
uma.adapters.graph.base
========================

GraphAdapter — Abstract graph backend interface for UMA.

This module defines the common interface that all graph adapters must
implement (e.g., Neo4jAdapter, MemgraphAdapter).

Design goals
------------
- Keep UMA core logic independent of any specific graph database.
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
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GraphAdapter(abc.ABC):
    """
    Abstract graph backend interface.

    All UMA graph drivers (Neo4j, Memgraph, etc.) must implement this.

    Required Methods
    ----------------
    - run_query(cypher: str, params: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]
    - close() -> None
    """

    @abc.abstractmethod
    def run_query(
        self,
        cypher: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
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

    def verify_connectivity(self) -> bool:
        """
        Best-effort connectivity check for health probes.

        Adapters may override this to use native driver checks for accuracy.
        """
        try:
            self.run_query("RETURN 1 AS ok")
            return True
        except Exception:
            logger.exception("GraphAdapter.verify_connectivity failed.")
            return False
