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

## Security Primitives (ASI06 / ASI03 / ASI05 + LLM baseline)

UMA is a memory SDK — the OWASP Agentic Security Initiative (ASI) is the most relevant framework because memory is where ASI threats materialise. The three ASI controls below are first-class architectural properties; the LLM Top 10 coverage follows from building the memory layer correctly.

**Seven primitives enforce security on every write and read boundary:**

- **Provenance** — every artifact carries its lineage: origin, owner, derivation chain, and timestamps. A runtime invariant, not a debugging convenience. Enables every other primitive.
- **Write-time trust scoring** — `uma.common.trust.score_source` assigns `trust_score ∈ [0, 1]` at the moment of write. The score travels with the artifact permanently and is blended into retrieval ranking: `final = (1 - trust_weight) * similarity + trust_weight * trust_score`. Anything below `min_trust_score` (default 0.5) is dropped before results are returned.
- **Cryptographic integrity** — every Fact, Episode, Skill, and Chunk carries a SHA-256 `content_hash` computed at write time. `verify_integrity` re-derives and compares; a mismatch quarantines the record and appends an audit log entry.
- **Injection pattern detection** — every artifact is scanned at its write boundary against bundled English, French, Spanish, German, and Simplified Chinese YAML catalogs covering jailbreak prompts, role impersonation, context switching, data exfiltration probes, encoded payloads, alignment breaking, debug spoofing, config leakage, delimiter smuggling, and embedded LLM protocol artifacts. High-severity hits set `trust_score` to 0.0 and quarantine the artifact. The catalog is YAML-configurable via `security.custom_patterns_path`.
- **Two-layer injection gate** — `scan_user_input` is the pre-LLM advisory gate: synchronous, never raises, caller decides. `process_turn` rescans `user_msg` at the storage boundary (defense-in-depth); on high severity it raises `InjectionDetectedError` and nothing is stored — no working memory, no episode, no chunks, no facts.
- **Quarantine** — suspicious artifacts are stored with `quarantined_at` set and excluded from every retrieval query (`AND quarantined_at IS NULL`). The record stays in the database for review. `list_quarantined`, `reinstate_quarantined`, and `purge_quarantined` manage the lifecycle.
- **Ingest boundary hardening** — MIME consistency check rejects executables and extension/content mismatches before any parser runs. File size (`max_file_bytes`, default 50 MB) and PDF page count (`pdf_max_pages`, default 5000) caps prevent resource abuse. HTML and Markdown are sanitized of scripts, iframes, inline event handlers, `javascript:` and `data:` URLs before chunking. Per-category removal counts are recorded in `meta["security"]["sanitization"]`.

**OWASP ASI coverage (primary):** ASI06 Memory Poisoning — primitives 2–6 compose directly. ASI03 Identity & Privilege Abuse — mandatory `tenant_id` / `owner_type` / `owner_id` on every artifact, pushed into SQL and vector queries before the candidate cap (C1 contract). ASI05 Unexpected Code Execution (ingest path) — `PickleParser` removed, MIME checks, HTML sanitization.

**OWASP LLM Top 10 coverage (as a consequence — 6 of 10):**

| Control | Scope | How |
|---|---|---|
| **LLM01** Prompt Injection | In scope | Two-layer gate: `scan_user_input` (pre-LLM, advisory) + `process_turn` write-time rescan (raises `InjectionDetectedError` on high severity, drops turn entirely) |
| **LLM02** Sensitive Information Disclosure | Partial | Retrieval audit log stores SHA-256-hashed query preview only, never raw text. HTML sanitization strips active URLs from ingested documents. |
| **LLM03** Supply Chain | Out of scope (adjacent) | No model training or plugin registry. `PickleParser` removed; MIME checks reject executables at the document ingest boundary. |
| **LLM04** Data and Model Poisoning | In scope | Quarantined chunks excluded from fact extraction. SHA-256 `content_hash` + `verify_integrity` detect post-hoc tampering across all lanes. |
| **LLM05** Improper Output Handling | Out of scope | UMA returns context, not generated output. Caller owns rendering and escaping. |
| **LLM06** Excessive Agency | Out of scope | No tool use, no function calling, no autonomous action. Pure memory SDK. |
| **LLM07** System Prompt Leakage | Out of scope | System prompts live in the calling application, not UMA. |
| **LLM08** Vector and Embedding Weaknesses | In scope — primary | C1 contract: isolation (`tenant_id` / `owner_type` / `owner_id`) pushed into every vector query *before* the k-nearest cap — a heavy tenant cannot starve others. All three adapters refuse empty isolation at upsert. SQL layer adds the same filter. Write-time scanning addresses the RAG poisoning sub-problem. |
| **LLM09** Misinformation | Partial | Every fact carries provenance back to source chunks. Quarantined facts excluded at SQL retrieval layer (`AND quarantined_at IS NULL`). `provenance_valid` is a top-level field on every `retrieve_memory` result. |
| **LLM10** Unbounded Consumption | Partial | Ingest side (UMA-owned): `max_file_bytes` and `pdf_max_pages` cap resource use. Retrieval side (caller-owned): `set_rate_limit_hook` exposes a single plug-point on every public method — UMA ships no default limiter and owns no throttling policy (accounting, storage, timeouts, and refusal semantics are the caller's). |

The vector isolation contract (C1) and the rate-limit hook are documented separately; see `uma-security.md` for the full model.

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

UMA Lite embeds SQLite (authoritative) and LanceDB (vector index), so it requires no external storage service. An LLM and embedding provider must still be configured; these may run locally or remotely.

| Config | Use |
|---|---|
| `config/uma.yaml` | Default runnable config |

LLM and embedding values are user-customizable baselines. See `uma-configure.md` for full configuration details.

## Status

UMA is in **beta**. Schema and API may change. No backward-compatibility guarantees are made; the codebase deliberately removes obsolete paths rather than preserving them.
