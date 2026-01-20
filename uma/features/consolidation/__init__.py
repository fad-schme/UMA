"""
UMA-3 Consolidation Subsystem (Optional Feature)
================================================

This package implements optional consolidation ("sleep cycle") logic
for UMA-3. Consolidation merges episodic memory into higher-level
semantic knowledge and prunes outdated content.

Modules:
    - feature.py       → Integration with UMAMemory via UMAFeature
    - consolidator.py  → Full consolidation workflow
    - clusterer.py     → Episode similarity clustering
    - summarizer.py    → LLM-based cluster summarization
    - pruner.py        → Forgetting logic
    - utils.py         → Simple helpers

This subsystem is OPTIONAL. UMA-3 runs without it, but consolidation 
improves long-term quality and reduces data bloat.
"""

from .feature import ConsolidationFeature
from .consolidator import Consolidator
from .clusterer import EpisodeClusterer
from .summarizer import ConsolidationSummarizer
from .pruner import Pruner

__all__ = [
    "ConsolidationFeature",
    "Consolidator",
    "EpisodeClusterer",
    "ConsolidationSummarizer",
    "Pruner",
]