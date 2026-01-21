"""
Episodic Memory Subsystem (UMA Core)
======================================

This package contains the full episodic memory implementation for UMA:

    core.py        – EpisodicCore orchestrator
    indexer.py     – Episode creation, summarization, embedding
    archive.py     – Archival, lifecycle management, forgetting
    mapper.py      – Working memory → Episode transformations
    policies.py    – Retention policies, TTL, salience thresholds

Coding Agent Instructions
-------------------------
- Maintain clear separation between "episodic creation", "episodic storage",
  and "episodic lifecycle".
- Do not embed business rules directly in EpisodicCore; this module should
  remain generic and data-driven.
"""

from .core import EpisodicCore
from .indexer import EpisodeIndexer
from .archive import EpisodicArchive
from .mapper import EpisodeMapper
from .policies import EpisodicRetentionPolicy

__all__ = [
    "EpisodicCore",
    "EpisodeIndexer",
    "EpisodicArchive",
    "EpisodeMapper",
    "EpisodicRetentionPolicy",
]