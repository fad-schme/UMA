"""
uma.adapters.graph.neo4j_adapter
=================================

Neo4jAdapter — GraphAdapter backed by the official `neo4j` Bolt driver.

Referenced via a plugin spec in config, e.g.:

    storage:
      graph_backend: "uma.adapters.graph.neo4j_adapter:Neo4jAdapter"
      graph_config:
        uri: "bolt://localhost:7687"
        user: "neo4j"
        password: "..."
        database: "neo4j"

`user`/`password` are optional — omit both for a container running with
auth disabled (`NEO4J_AUTH=none`).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from neo4j import GraphDatabase

from .base import GraphAdapter

logger = logging.getLogger(__name__)


class Neo4jAdapter(GraphAdapter):
    def __init__(
        self,
        uri: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        **driver_config: Any,
    ) -> None:
        auth = (user, password) if user and password else None
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=auth, **driver_config)
        logger.info("Neo4jAdapter initialized (uri=%s, database=%s).", uri, database)

    def run_query(
        self,
        cypher: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]
        except Exception:
            logger.exception("Neo4jAdapter.run_query failed.")
            return []

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            logger.exception("Neo4jAdapter.close failed.")

    def verify_connectivity(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            logger.exception("Neo4jAdapter.verify_connectivity failed.")
            return False
