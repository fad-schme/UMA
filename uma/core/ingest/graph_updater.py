from __future__ import annotations

import logging
from typing import List, Any

from ...types_fact import Fact

logger = logging.getLogger(__name__)


async def update_graph(
    facts: List[Fact],
    *,
    graph_core: Any,
) -> int:
    """
    Update graph with new facts.

    Returns number of fact edges attempted.
    """
    if not facts:
        return 0
    if graph_core is None:
        logger.warning("update_graph: graph_core missing; skipping")
        return 0

    try:
        graph_core.add_facts(facts)
        return len(facts)
    except Exception:
        logger.exception("update_graph: graph update failed")
        return 0
