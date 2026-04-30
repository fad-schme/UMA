"""
episodic/mapper.py
==================

EpisodeMapper
-------------

Maps working memory entries to Episode-friendly structures.

Coding Agent Instructions
-------------------------
- This mapper is intentionally simple—it only transforms WMEntry models
  into dictionaries consumed by EpisodeIndexer.
- Extend here if you add metadata extraction rules, speaker tracking,
  or utterance normalization.
"""

from __future__ import annotations
import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)


class EpisodeMapper:
    """
    Maps WM entries → indexer input.
    """

    def __init__(self):
        logger.debug("EpisodeMapper initialized.")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def map_entries(self, entries: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert WMEntry objects into dicts expected by EpisodeIndexer.
        """
        mapped = []
        for ent in entries:
            try:
                mapped.append(
                    {
                        "role": getattr(ent, "role"),
                        "content": getattr(ent, "content"),
                        "metadata": getattr(ent, "metadata", None),
                    }
                )
            except Exception:
                logger.exception("Failed to map WMEntry=%s", ent)
        return mapped
