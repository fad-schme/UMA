from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def maybe_trigger_consolidation(
    *,
    memory: Any,
    user_id: str | None,
    enabled: bool = False,
) -> None:
    """
    Lightweight hook to trigger consolidation.

    This does NOT run consolidation logic; it only calls the feature method
    if present and enabled.
    """
    if not enabled:
        return
    if memory is None or not hasattr(memory, "consolidation_run"):
        logger.debug("maybe_trigger_consolidation: consolidation feature unavailable")
        return
    if not user_id:
        logger.debug("maybe_trigger_consolidation: missing user_id")
        return

    try:
        await memory.consolidation_run(user_id)
        logger.info("Consolidation triggered for user_id=%s", user_id)
    except Exception:
        logger.exception("maybe_trigger_consolidation: consolidation_run failed")
