"""
uma.adapters.graph.neo4j_adapter
=================================

Neo4jAdapter — unified Neo4j graph backend for UMA.

This adapter implements the GraphAdapter interface and is intended to be
the single Neo4j integration point for UMA. It wraps the official
`neo4j` Python driver with:

- Connection pooling
- Safe query execution
- Structured logging
- Clear error handling

Design goals
------------
- Keep UMA core independent from Neo4j driver details.
- Provide a simple, consistent API for running Cypher queries.
- Make it easy to add retries, tracing, or async support later without
  changing core logic.

Coding Agent Instructions
-------------------------
- Do NOT put application or temporal-graph logic here; this is pure I/O.
- If you add retries, make them configurable (max attempts, backoff).
- If you later introduce async, consider a separate AsyncNeo4jAdapter
  implementing the same GraphAdapter interface but with awaitables.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, basic_auth, Driver, Session
from neo4j.graph import Node, Relationship, Path

from uma.adapters.graph.base import GraphAdapter
from uma.core.utils.retry import retry_sync

logger = logging.getLogger(__name__)


class Neo4jAdapter(GraphAdapter):
    """
    Unified Neo4j backend implementation for UMA.

    Parameters
    ----------
    uri : str
        Neo4j connection URI (e.g. 'neo4j://localhost:7687').
    user : str
        Username for Neo4j authentication.
    password : str
        Password for Neo4j authentication.
    max_pool_size : int, default=50
        Maximum number of connections in the driver pool.

    Notes
    -----
    - This adapter is synchronous and uses neo4j.GraphDatabase.driver.
    - All queries should go through `run_query()`.
    - Use `close()` during shutdown to free resources.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        max_pool_size: int = 50,
        database: Optional[str] = None,
    ) -> None:
        if not uri:
            raise ValueError("Neo4jAdapter: uri must not be empty")
        if not user:
            raise ValueError("Neo4jAdapter: user must not be empty")

        self._database = database
        try:
            self._driver: Driver = GraphDatabase.driver(
                uri,
                auth=basic_auth(user, password),
                max_connection_pool_size=max_pool_size,
            )
            # Optionally test connectivity:
            self._driver.verify_connectivity()

            logger.info(
                "Neo4jAdapter connected to %s with max_pool_size=%d",
                uri,
                max_pool_size,
            )
        except Exception:
            logger.exception("Neo4jAdapter: failed to create driver for %s", uri)
            raise

    # ------------------------------------------------------------------ #
    # GraphAdapter implementation
    # ------------------------------------------------------------------ #

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, Node):
            return {
                "id": value.id,
                "labels": list(value.labels),
                "properties": dict(value),
            }
        if isinstance(value, Relationship):
            return {
                "id": value.id,
                "type": value.type,
                "start_id": value.start_node.id,
                "end_id": value.end_node.id,
                "properties": dict(value),
            }
        if isinstance(value, Path):
            return {
                "nodes": [self._normalize_value(n) for n in value.nodes],
                "relationships": [self._normalize_value(r) for r in value.relationships],
            }
        if isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        if isinstance(value, tuple):
            return [self._normalize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in value.items()}
        return value

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self._normalize_value(v) for k, v in record.items()}

    def run_query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return all records as dictionaries.

        Parameters
        ----------
        cypher : str
            Cypher query text.
        params : Optional[Dict[str, Any]], default=None
            Query parameters (if any).

        Returns
        -------
        List[Dict[str, Any]]
            Result rows as dictionaries.
        """
        if not cypher:
            raise ValueError("Neo4jAdapter.run_query called with empty Cypher string.")

        def _call() -> List[Dict[str, Any]]:
            with self._driver.session(database=self._database) as session:
                result = session.run(cypher, params or {})
                return [self._normalize_record(dict(record)) for record in result]

        records = retry_sync(_call)
        logger.debug(
            "Neo4jAdapter.run_query: executed Cypher with %d row(s) returned.",
            len(records),
        )
        return records

    def close(self) -> None:
        """
        Close the underlying Neo4j driver and release resources.

        This method should be called during clean shutdown of UMA.
        """
        try:
            self._driver.close()
            logger.info("Neo4jAdapter: driver closed.")
        except Exception:
            logger.exception("Neo4jAdapter: failed to close driver.")

    def verify_connectivity(self) -> bool:
        """Verify backend connectivity. Returns True if reachable."""
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            logger.exception("Neo4jAdapter: connectivity check failed.")
            return False
