"""
Temporal Knowledge Graph → GraphViz exporter.

This visualization creates a .dot file that can be rendered via:

    dot -Tpng tkg.dot -o tkg.png

Shows:
- Entity nodes
- Fact nodes
- Episode nodes
- Relationship edges

Coding agent instructions:
--------------------------
- Use this with Neo4jBackendAdvanced.query() outputs.
- NEVER modify graph DB from this module.
"""

from __future__ import annotations

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TKGGraphVizExporter:
    """
    Converts TKG records into GraphViz DOT.

    Usage:
        exporter = TKGGraphVizExporter()
        dot = exporter.to_dot(records)
        with open("tkg.dot", "w") as f: f.write(dot)
    """

    def to_dot(self, records: List[Dict]) -> str:
        """
        Convert graph query records into DOT format.

        Expects each record to be like:
            {"ent": {...}, "x": {...}}

        Returns
        -------
        str: DOT graph
        """
        nodes = set()
        edges = set()

        for rec in records:
            keys = list(rec.keys())
            if len(keys) < 2:
                continue

            a, b = rec[keys[0]], rec[keys[1]]

            if not (isinstance(a, dict) and isinstance(b, dict)):
                continue

            if "id" not in a or "id" not in b:
                continue

            nodes.add(a["id"])
            nodes.add(b["id"])
            edges.add((a["id"], b["id"]))

        out = ["digraph TKG {", '  rankdir="LR";']

        for n in nodes:
            out.append(f'  "{n}" [shape=ellipse];')

        for (a, b) in edges:
            out.append(f'  "{a}" -> "{b}";')

        out.append("}")
        return "\n".join(out)