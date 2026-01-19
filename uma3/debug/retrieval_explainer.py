"""
Hybrid Retrieval Explainer

Shows:
- What stores were queried
- Which candidates were retrieved
- Why items were selected (selector rules)
- Reranker output

This is critical for debugging UMA-3 hybrid retrieval behavior.

Coding agent instructions:
--------------------------
- Use with UMA3Memory.retrieve_context.
- DO NOT break agent logic — this is read-only visualization.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class RetrievalExplainer:

    @staticmethod
    def explain(raw: Dict[str, List[Any]], selected: Dict[str, List[Any]]) -> str:
        """
        Build a readable explanation comparing:
        - raw retrieved candidates
        - final selected items
        """
        out = ["HYBRID RETRIEVAL EXPLAINER\n"]

        def fmt_list(name, lst):
            out.append(f"{name.upper()} (raw={len(lst)}):")
            for item in lst:
                out.append(f"  - {getattr(item, 'id', item)}")
            out.append("")

        fmt_list("episodes", raw.get("episodes", []))
        fmt_list("facts", raw.get("facts", []))
        fmt_list("skills", raw.get("skills", []))
        fmt_list("graph", raw.get("graph", []))

        out.append("SELECTED:")
        for key, lst in selected.items():
            out.append(f"  {key}: {[getattr(x, 'id', x) for x in lst]}")

        return "\n".join(out)