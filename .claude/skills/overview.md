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

## Two things worth calling out explicitly

- **No legacy/backward-compatibility shims.** Obsolete paths are removed rather than kept for compatibility, so pin a version if you need stability across releases.
- **Security scope stops at the pipeline.** UMA defends artifacts moving through ingest and retrieval (scanning, quarantine, isolation); it is not a sandbox against a malicious developer running it locally.

## Core Product Principle

**RLM is always enabled.** UMA uses an iterative retrieval loop (RLM) that repeatedly evaluates coverage of what has been found and continues searching — targeting different predicates, lanes, or broader searches — until coverage thresholds are satisfied or hard budgets are reached. After the loop completes, the LLM prunes the collected facts for relevance to the query. Navigation decisions are deterministic and coverage-driven; the LLM participates in post-retrieval quality filtering, not in deciding what to search for next. The "long context" lives in the environment (memory), not in the prompt.

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

Cross-tenant access is enforced at the storage layer, not by application convention:

- The vector index (LanceDB) promotes `tenant_id` / `owner_type` / `owner_id` to first-class indexed columns and pushes them into every query's `WHERE` clause before the candidate cap is applied.
- SQL stores filter by isolation in every read path.
- Vector adapters refuse empty isolation values at write time.

See `security.md` for the isolation contract in full, including the boundary-filter bugs found and fixed in past releases.

**Practical rule:** If you cannot answer "which user/agent/project is allowed to see this row?" the design is wrong.

## Runtime Scope Invariants

- No shared mutable object stores current request scope (`memory.user_id`, `controller.current_scope`, etc. are forbidden patterns)
- Every API entry point operates from an **explicit immutable `RuntimeContext`**
- Working memory and episodic turn memory are **session-local by default**
- Semantic facts extracted from turns are session-local by default and must be explicitly promoted to become durable

## Security Primitives (ASI06 / ASI03 / ASI05 + LLM baseline)

Seven primitives — provenance, write-time trust scoring, cryptographic integrity, injection pattern detection, the two-layer injection gate, quarantine, and ingest boundary hardening — enforce security on every write and read boundary. Together they cover ASI06 (Memory Poisoning, primary), ASI03 (Identity & Privilege Abuse), ASI05 (Unexpected Code Execution, ingest path), and 6 of the OWASP LLM Top 10. Full primitive descriptions, the ASI/LLM coverage tables, and the vector isolation contract: see `security.md`.

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

memory = UMAMemory.from_yaml("config/uma.yaml")
# One instance serves every agent and every user. agent_id and user_id are
# passed on every call; tenant_id defaults to "default" when omitted.

# Pre-LLM injection gate (returns dict, never raises)
scan = memory.scan_user_input(user_msg)

# Retrieval — explicit scope, isolated by construction
context = await memory.retrieve_context(query_text=..., agent_id=..., user_id=..., session_id=...)
result  = await memory.retrieve_memory(query_text=..., agent_id=..., user_id=..., session_id=...)

# Ingest — raises InjectionDetectedError on high-severity user_msg
await memory.process_turn(user_id=..., user_msg=..., assistant_reply=..., session_id=...)

# Optional: opt this agent into profile-gated fact promotion
await memory.set_agent_profile(description=..., focus_areas=[...], tenant_id=...)

# Rate-limit hook (optional)
memory.set_rate_limit_hook(my_hook)
```

See `api.md` for full signatures and contracts.

## Storage Architecture

| Layer | Role |
|---|---|
| SQLite | Authoritative source of truth for chunk text and metadata |
| LanceDB / FAISS / InMemory | Retrieval accelerator — rebuildable from SQL; isolation enforced via C1 contract |
| Graph (optional) | Relationship routing; disabled in public profiles |

Vector store is never the source of truth. It stores only ids, vectors, isolation columns, and filterable metadata — not full chunk text.

## Runtime Profile

UMA Lite embeds SQLite (authoritative) and LanceDB (vector index), so it requires no external storage service. An LLM and embedding provider must still be configured; these may run locally or remotely.

| Config | Use |
|---|---|
| `config/uma.yaml` | Default runnable config |

LLM and embedding values are user-customizable baselines. See `configure.md` for full configuration details.

## Status

UMA is in **beta**. Schema and API may change. No backward-compatibility guarantees are made; the codebase deliberately removes obsolete paths rather than preserving them.
