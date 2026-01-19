"""
Memory HTML Reporter

Generates a standalone HTML report visualizing:
- Semantic facts
- Episodes
- Working memory snapshot
- Skills
- Hybrid retrieval explanation (optional)

Coding agent instructions:
--------------------------
- Use this for administrative dashboards.
- HTML deliberately uses minimal CSS to avoid dependencies.
"""

from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime
import html as _html
from ..types_fact import Fact
from ..types_episode import Episode
from ..types_skill import Skill


class HTMLReport:

    @staticmethod
    def generate(
        facts: List[Fact],
        episodes: List[Episode],
        skills: List[Skill],
        retrieval_expl: str = "",
    ) -> str:
        """
        Construct a full HTML report.
        """
        def esc(s: str) -> str:
            return _html.escape(str(s))

        facts_html = "<br>".join(
            f"<b>{esc(f.subject)}</b>: {esc(f.predicate)} = {esc(f.object)} (salience={esc(f.meta.get('salience'))})"
            for f in facts
        )

        episodes_html = "<br>".join(
            f"<b>{esc(ep.timestamp)}</b> — {esc(ep.summary)}" for ep in episodes
        )

        skills_html = "<br>".join(
            f"<b>{esc(s.name)}</b> — triggers: {esc(s.trigger_phrases)}" for s in skills
        )

        return f"""
        <html>
        <head>
            <title>UMA-3 Memory Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>UMA-3 Memory Debug Report</h1>

            <div class="section">
                <h2>Semantic Facts</h2>
                {facts_html}
            </div>

            <div class="section">
                <h2>Episodic Timeline</h2>
                {episodes_html}
            </div>

            <div class="section">
                <h2>Skills</h2>
                {skills_html}
            </div>

            <div class="section">
                <h2>Hybrid Retrieval Explanation</h2>
                <pre>{_html.escape(retrieval_expl)}</pre>
            </div>

        </body>
        </html>
        """