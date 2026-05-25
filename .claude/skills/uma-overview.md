---
name: uma-overview
description: Explains what UMA is and isn't, its design philosophy, DAT (data access tenancy) invariants, ownership and tenancy model, runtime scope rules, and the security primitives that ship with every artifact. Use this skill when answering questions like "what is UMA", "is UMA a chat app", "what are the DAT invariants", "what does UMA NOT do", or any orientation question about UMA's scope, philosophy, or core guarantees.
---

# UMA — Overview

## What UMA Is

UMA (Universal Memory Architecture) is a **memory and context runtime SDK** for developers building AI agents. It ingests data (documents, conversation turns), stores it across multiple typed memory lanes (SQL + vector + optional graph), and exposes two thin retrieval products:

- `retrieve_context(...)` — curated, evidence-oriented context for RAG-style LLM prompting
- `retrieve_memory(...)` — compiled, evidence-backed knowledge for continuity-oriented memory

UMA manages memory only. The calling application owns prompts, tool use, reasoning, and final responses.

## What UMA Is NOT

- Not a chat application or autonomous agent
- Not a "big prompt builder"
- Not a knowledge-graph-first system (graph is a supporting lane, not the primary truth)
- Not a framework that maintains legacy/backward compatibility — obsolete paths are removed
- Not a security boundary against malicious developers running it locally (UMA defends artifacts in motion through the pipeline; it does not sandbox the operator)

## Core Product Principle

**RLM is always enabled.** Retrieval is LLM-controlled search for context, not for answering. The "long context" lives in the environment (memory), not in the prompt.

## Design Philosophy

Every component must be:

- **Understandable** — explainable in one sentence, traceable through one canonical flow
- **Lean** — no thin wrappers, no pass-through helpers, no duplicate logic
- **Evidence-backed** — provenance is carried from raw chunks through compiled artifacts
- **Owner-scoped** — every stored and retrieved artifact has an explicit owner
- **Secure by construction** — isolation, scanning, and quarantine are enforced at the storage layer, not by application-layer convention

If a feature makes the system harder to explain, install, or trace, simplify before extending.

## DAT Invariants (Non-Negotiable)

Every stored and retrieved artifact must be **owner-scoped**:

| Field | Rule |
|---|---|
| `owner_type` | One of: `agent`, `user`, `workspace`, `system` |
| `owner_id` | Required; non-empty; consistent with the lane |
| `tenant_id` | Required for durable artifacts; preserved end-to-end |
| `session_id` + `agent_id` | Required for session-local artifacts |

Cross-tenant access is impossible **by construction**:

- The vector index (LanceDB) promotes `tenant_id` / `owner_type` / `owner_id` to first-class indexed columns and pushes them into every query's `WHERE` clause before the candidate cap is applied.
- SQL stores filter by isolation in every read path.
- Vector adapters refuse empty isolation values at write time.

**Practical rule:** If you cannot answer "which user/agent/project is allowed to see this row?" the design is wrong.

## Runtime Scope Invariants

- No shared mutable object stores current request scope (`memory.user_id`, `controller.current_scope`, etc. are forbidden patterns)
- Every API entry point operates from an **explicit immutable `RuntimeContext`**
- Working memory and episodic turn memory are **session-local by default**
- Semantic facts extracted from turns are session-local by default and must be explicitly promoted to become durable

## Security Primitives (OWASP LLM01–LLM08 Baseline)

UMA enforces these on every write boundary:

- **Prompt-injection scanning** — every user/assistant message and every ingested document chunk is scanned against a YAML pattern catalog. High-severity hits trip `quarantined_at`; the artifact stays in the database but is excluded from retrieval.
- **Trust scoring** — every artifact carries a `trust_score` in `[0, 1]`. Retrieval ranks by `(1 - trust_weight) * similarity + trust_weight * trust_score` and drops anything below `min_trust_score` (default 0.5).
- **Content hashing** — every Fact, Episode, Skill, and Chunk carries a SHA-256 `content_hash`. `verify_integrity` re-derives and compares; mismatch → quarantine.
- **MIME consistency + size limits** — ingest rejects executable types, extension/content mismatches, files over `max_file_bytes` (default 50MB), and PDFs over `pdf_max_pages` (default 5000).
- **Vector isolation push-down** — LanceDB filters by tenant/owner before the k-nearest cap is applied. No cross-tenant leakage under load.
- **Retrieval audit log** — every retrieval call records a hashed query preview, scope, severity, and result counts (default on; toggle via `security.retrieval_audit_enabled`).
- **Rate-limit hook** — operators register a single callable that runs at the top of `retrieve_context`, `retrieve_memory`, `process_turn`, and `ingest_document`. The hook raises to refuse.

See `uma-security.md` for the full security model.

## One-Sentence Product Test

A user must be able to understand what UMA is in one sentence and install it in one path.

## Public API Surface (intentionally small)

```python
from uma import UMAMemory, InjectionDetectedError
from uma.api.management import (
    explain_result, lint_memory_drift, verify_integrity,
    list_quarantined, reinstate_quarantined, purge_quarantined,
    list_retrieval_audit,
)

memory = UMAMemory.from_yaml("config/uma.yaml").set_context(agent_id="agent-default")

# Pre-LLM injection gate (returns dict, never raises)
scan = memory.scan_user_input(user_msg)

# Retrieval — explicit scope, isolated by construction
context = await memory.retrieve_context(query_text=..., user_id=..., tenant_id=..., session_id=...)
result  = await memory.retrieve_memory(query_text=..., user_id=..., tenant_id=..., session_id=...)

# Ingest — raises InjectionDetectedError on high-severity user_msg
await memory.process_turn(user_id=..., user_msg=..., assistant_reply=..., session_id=...)

# Rate-limit hook (optional)
memory.set_rate_limit_hook(my_hook)
```

See `uma-api.md` for full signatures and contracts.

## Storage Architecture

| Layer | Role |
|---|---|
| SQLite | Authoritative source of truth for chunk text and metadata |
| LanceDB / FAISS / InMemory | Retrieval accelerator — rebuildable from SQL; isolation enforced via C1 contract |
| Graph (optional) | Relationship routing; disabled in public profiles |

Vector store is never the source of truth. It stores only ids, vectors, isolation columns, and filterable metadata — not full chunk text.

## Runtime Profile

UMA Lite uses a single embedded profile: SQLite (authoritative) + LanceDB (vector index). No external services required.

| Config | Use |
|---|---|
| `config/uma.yaml` | Default runnable config |

LLM and embedding values are user-customizable baselines. See `uma-configure.md` for full configuration details.

## Status

UMA is in **beta**. Schema and API may change. No backward-compatibility guarantees are made; the codebase deliberately removes obsolete paths rather than preserving them.
