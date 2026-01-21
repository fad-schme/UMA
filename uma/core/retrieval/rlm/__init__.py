# uma/core/retrieval/rlm/__init__.py
from .controller import RLMController
from .environment import MemoryEnvironment, UMAMemoryEnvironment
from .context_pack import ContextPack

"""
uma.core.retrieval.rlm
======================

Recursive Language Model (RLM) retrieval components for UMA.

This package implements **bounded, recursive memory retrieval**, allowing
UMA to explore large memory stores iteratively instead of relying on a
single retrieval pass.

IMPORTANT:
----------
These components:
• Do NOT perform agent reasoning
• Do NOT generate answers
• Do NOT construct prompts

They exist solely to improve **memory recall quality** by deciding
*what memory to retrieve next*, not *what to say*.

All reasoning and response generation remains outside UMA.


uma.core.retrieval.rlm_controller
=================================

RLMController — Recursive (bounded) retrieval controller for UMA.

Why this exists
---------------
UMA already has:
- Working Memory (WM)
- Multi-store retrieval (episodic/semantic/procedural/graph)
- Deterministic ranking/truncation (MemorySelector)

But UMA retrieval is currently "single-shot":
    embed(query) -> retrieve top-k -> rank -> return.

Recursive Language Models (RLM) style retrieval adds:
    "Do we have enough context? If not, what should we fetch next?"

This module implements a production-safe, bounded retrieval controller that:
- Uses an LLM *only* to choose next retrieval actions (not to answer user queries).
- Calls a safe MemoryEnvironment to retrieve additional items in small steps.
- Stops deterministically based on budgets (steps, token budget, time).
- Returns a developer-friendly ContextPack (RAG-ready), not prompt glue.

Design principles
----------------
- Safety: no arbitrary code execution; environment exposes only whitelisted ops.
- Robustness: strict JSON parsing, schema validation, fallbacks.
- Observability: structured logging for each step/action.
- Boundedness: max_steps, max_actions, max_items, timeouts.

Integration points
------------------
- UMAMemory.get_user_context(user_id, query_text) may use:
      controller = RLMController(...)
      pack = await controller.retrieve_context(user_id, query_text)

- Controller assumes a MemoryEnvironment implementation is provided.
  A default UMAMemoryEnvironment is provided here which wraps RetrievalService
  and graph adapter in a safe way.

Python compatibility
--------------------
- Uses Python 3.9 compatible typing (no `|` unions).

"""