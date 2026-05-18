# UMA — Overview

## What UMA Is

UMA (Universal Memory Architecture — Retrieval Language Model) is a **memory and context runtime SDK** for developers building AI agents. It ingests data (documents, conversation turns), stores it across multiple typed memory lanes (SQL + vector + optional graph), and exposes two thin retrieval products:

- `retrieve_context(...)` — curated, evidence-oriented context for RAG-style LLM prompting
- `retrieve_memory(...)` — compiled, evidence-backed knowledge for continuity-oriented memory

UMA manages memory only. Your application owns prompts, tool use, reasoning, and final responses.

## What UMA Is NOT

- Not a chat application or autonomous agent
- Not a "big prompt builder"
- Not a knowledge-graph-first system (graph is a supporting lane, not the primary truth)
- Not a framework that maintains legacy/backward compatibility — obsolete paths are removed

## Core Product Principle

**RLM is always enabled.** Retrieval is LLM-controlled search for context, not answering. The "long context" lives in the environment (memory), not in the prompt.

## Design Philosophy

Every component must be:
- **Understandable** — explainable in one sentence, traceable through one canonical flow
- **Lean** — no thin wrappers, no pass-through helpers, no duplicate logic
- **Evidence-backed** — provenance is carried from raw chunks through compiled artifacts
- **Owner-scoped** — every stored and retrieved artifact has an explicit owner

If a feature makes the system harder to explain, install, or trace, simplify before extending.

## DAT Invariants (Non-Negotiable)

Every stored and retrieved artifact must be **owner-scoped**:

| Field | Rule |
|---|---|
| `owner_type` | One of: `agent`, `user`, `workspace`, `system` |
| `owner_id` | Required; consistent with the lane |
| `tenant_id` | Required for durable artifacts; preserved end-to-end |
| `session_id` + `agent_id` | Required for session-local artifacts |

Cross-tenant access is impossible by construction. Cross-agent sharing is denied by default.

**Practical rule:** If you cannot answer "which user/agent/project is allowed to see this row?" the design is wrong.

## Runtime Scope Invariants

- No shared mutable object stores current request scope (`memory.user_id`, `controller.current_scope`, etc. are forbidden patterns)
- Every API entry point operates from an **explicit immutable `RuntimeContext`**
- Working memory and episodic turn memory are **session-local by default**
- Semantic facts extracted from turns are session-local by default and must be explicitly promoted to become durable

## One-Sentence Product Test

A user must be able to understand what UMA is in one sentence and install it in one path.

## Public API Surface (intentionally small)

```python
from uma import UMAMemory
from uma.api.management import explain_result, lint_memory_drift

memory = UMAMemory.from_yaml("config/uma.yaml")
memory = memory.set_context(agent_id="agent-default")

context = await memory.retrieve_context(query_text=..., user_id=..., ...)
result  = await memory.retrieve_memory(query_text=..., user_id=..., ...)
await memory.process_turn(user_id=..., user_msg=..., assistant_reply=..., session_id=...)
```

See `uma-api.md` for full signatures and contracts.

## Storage Architecture

| Layer | Role |
|---|---|
| SQLite | Authoritative source of truth for chunk text and metadata |
| LanceDB / Qdrant / FAISS | Retrieval accelerator — rebuildable from SQL |
| Graph (optional) | Relationship routing; disabled in public profiles |

Vector store is never the source of truth. It stores only ids and filterable metadata, not full chunk text.

## Runtime Profile

UMA uses a single embedded profile: SQLite (authoritative) + LanceDB (vector index). No external services required.

| Config | Use |
|---|---|
| `config/uma.yaml` | Default runnable config |
| `config/uma_lite.yaml` | Reference embedded profile (same storage settings) |

LLM and embedding values are user-customizable baselines. See `uma-configure.md` for full configuration details.
