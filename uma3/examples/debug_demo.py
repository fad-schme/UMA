"""
UMA-3 Debugging Demo

Shows how to:
- Inspect working memory
- Print episodic timeline
- Display semantic facts
- Export a TKG snapshot to GraphViz
- Explain hybrid retrieval

Coding agent instructions:
--------------------------
- Plug this script after a conversation loop.
"""

import asyncio

from ...debug.working_memory_viewer import WorkingMemoryViewer
from ...debug.episodic_timeline import EpisodicTimeline
from ...debug.semantic_graph_viewer import SemanticGraphViewer
from ...debug.tkg_graphviz_exporter import TKGGraphVizExporter
from ...debug.retrieval_explainer import RetrievalExplainer


async def main(memory, user_id, last_query):
    """
    memory: UMA3Memory instance
    user_id: user we debug
    last_query: last user query text
    """

    # 1. Working memory inspection
    if hasattr(memory, "get_working_context"):
        ctx = memory.get_working_context(user_id)
        print("\n--- Working Memory ---")
        print(WorkingMemoryViewer.render_context(ctx))

    # 2. Episodic timeline
    if memory.episodic_store:
        dummy = [0.0] * 16
        eps = await memory.episodic_store.search(dummy, user_id=user_id, k=50)
        print("\n--- Episodic Timeline ---")
        print(EpisodicTimeline.chronological_view(eps))

    # 3. Semantic facts
    if memory.semantic_store:
        dummy = [0.0] * 16
        facts = await memory.semantic_store.search(dummy, subject=user_id, k=50)
        print("\n--- Semantic Facts ---")
        print(SemanticGraphViewer.render_subject_facts(facts))

    # 4. TKG GraphViz export
    if memory.graph:
        recs = memory.graph.get_context_subgraph(user_id)
        dot = TKGGraphVizExporter().to_dot(recs)
        with open("tkg.dot", "w") as f:
            f.write(dot)
        print("\nWrote Temporal Graph to tkg.dot")

    # 5. Hybrid retrieval explainer
    raw = await memory.retriever.retrieve([0.0]*16, user_id, memory)
    selected = memory.selector.select(raw)
    print("\n--- Hybrid Retrieval Explanation ---")
    print(RetrievalExplainer.explain(raw, selected))


# Note: This demo does not run itself.
# The coding agent must call main() from a real UMA-3 agent context.