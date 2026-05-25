# UMA

**Universal Memory Architecture** — long-lived, evidence-backed memory for AI agents, with security built into every write and every read.

UMA is a memory and context runtime SDK for developers building AI agents. It ingests data, stores it across six typed memory lanes, and exposes a small retrieval surface. **UMA manages memory only** — your application owns prompts, tool use, reasoning, and final responses.

> **Status:** beta. No backward-compatibility guarantees.

---

## ✨ Why UMA

- 🧠 **Six typed memory lanes** — working memory, semantic facts, raw chunks, episodic, procedural, compiled wiki. You choose what to query.
- 🪶 **One install, zero external services** — embedded SQLite + LanceDB. `pip install -e .` and you're running.
- 🛡️ **Security by design** — every artifact is owner-scoped, injection-scanned, trust-scored, and content-hashed before it touches storage.
- 🔍 **Evidence-backed retrieval** — every fact carries provenance back to source chunks. No silent degradation into "vibes-based" RAG.
- 🏢 **Multi-tenant by construction** — cross-tenant access is impossible at the storage layer, not by application-layer convention.

---

## 🏛️ Architecture

![UMA architecture diagram](assets/uma-architecture.png)

UMA is a thin SDK around three concerns: **ingest** (data flows in, gets scanned, chunked, embedded), **storage** (SQLite is authoritative, LanceDB is a rebuildable accelerator), and **retrieval** (a canonical pipeline through candidate discovery, fusion, trust-aware ranking, and snippet rendering). Every write boundary scans for prompt injection; every read boundary enforces tenant/owner isolation and filters quarantined records.

For the full architectural model — invariants, pipelines, the vector isolation contract, and the OWASP Top 10 mapping — see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 🛡️ Security by Design

Security in UMA isn't a feature — it's the shape of every code path. Five primitives compose:

1. **Two-layer injection scanning** — pre-LLM advisory gate + write-time defense-in-depth
2. **Trust scoring + quarantine** — every artifact carries a trust score and quarantine flag; retrieval excludes quarantined records by construction
3. **Content hashing + integrity verification** — SHA-256 on every typed artifact; on-demand verification quarantines tampered records
4. **Ingest gating** — MIME consistency, file size caps, HTML/Markdown sanitization
5. **Retrieval audit log** — every retrieve call is recorded with a hashed query preview

### Mapping to OWASP Top 10 for LLM Applications 2025

The [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/) is the de-facto reference for AI application security. UMA is a memory SDK — not every category applies. Here's the honest mapping:

| OWASP 2025 Category | Scope | UMA's contribution |
| --- | --- | --- |
| 🟢 **LLM01: Prompt Injection** | In scope | Two-layer scanning: advisory pre-LLM gate (`scan_user_input`) + write-time per-artifact scan. High severity → quarantine; medium/low → trust reduction. |
| 🟢 **LLM02: Sensitive Information Disclosure** | Partial | Audit log stores SHA-256-hashed query previews only. HTML sanitization strips scripts and active URLs at ingest. |
| ⚪ **LLM03: Supply Chain** | Out of scope | No training, no fine-tuning. `PickleParser` was removed as adjacent ingest hardening. |
| 🟢 **LLM04: Data and Model Poisoning** | In scope (RAG path) | Quarantined chunks dropped before fact extraction. SHA-256 `content_hash` + `verify_integrity` detect post-hoc tampering. |
| ⚪ **LLM05: Improper Output Handling** | Out of scope | UMA returns context, not output. Caller owns rendering and escaping. |
| ⚪ **LLM06: Excessive Agency** | Out of scope | No tool use, no function calling, no autonomy. Pure memory. |
| ⚪ **LLM07: System Prompt Leakage** | Out of scope | System prompts live in the calling application, not UMA. |
| 🟢 **LLM08: Vector and Embedding Weaknesses** | In scope — primary | LanceDB promotes `tenant_id` / `owner_type` / `owner_id` to indexed columns and pushes them into `WHERE` before the k-nearest cap. Cross-tenant access impossible by construction. |
| 🟢 **LLM09: Misinformation** | Partial | Every fact carries provenance back to source chunks. `LatestWinsFactResolver` excludes quarantined facts from canonical selection. |
| 🟢 **LLM10: Unbounded Consumption** | In scope | Optional `set_rate_limit_hook` on every public method. `max_file_bytes` and `pdf_max_pages` cap ingest resource use. |
| 🟢 **ASI03: Identity & Privilege Abuse** (Agentic AI) | Partial — memory-layer | Explicit `tenant_id` / `owner_type` / `owner_id` on every artifact, enforced at the storage layer. Agent identity itself is the caller's concern. |
| 🟢 **ASI05: Unexpected Code Execution** (Agentic AI) | Partial — ingest-only | `PickleParser` removed; MIME consistency check rejects executables; HTML/Markdown sanitized before storage. UMA itself executes no code from memory. |
| 🟢 **ASI06: Memory Poisoning** (Agentic AI) | In scope | Write-time scan + quarantine at every storage boundary. Quarantined artifacts never enter retrieval and never seed fact extraction. |

**Six of ten LLM categories apply.** UMA is honest about what it covers and what's out of scope — there's no security theater. The four out-of-scope categories belong to the calling application: output handling, agent design, system prompts, and supply-chain procurement of models. Pair UMA with the controls appropriate to those layers for full coverage.

**Three of ten ASI categories apply.** UMA is a memory SDK, not an agent. The remaining seven belong to the agent layer above UMA — they require tool use, autonomy, or inter-agent communication that UMA doesn't have.

For the full security model — including the injection pattern catalog, severity behavior, quarantine lifecycle, and integrity verification — see [`.claude/skills/uma-security.md`](.claude/skills/uma-security.md) for the deep dive, or [`ARCHITECTURE.md`](ARCHITECTURE.md) for the architectural model.

---

## Quickstart

```bash
pip install -e .
```

```python
from uma import UMAMemory

memory = UMAMemory.from_yaml("config/uma.yaml").set_context(agent_id="my-agent")

context = await memory.retrieve_context(
    query_text=user_message,
    user_id="user-123",
    tenant_id="default",
    session_id="session-1",
)

reply = await your_llm(context, user_message)   # you own this

await memory.process_turn(
    user_id="user-123",
    user_msg=user_message,
    assistant_reply=reply,
    session_id="session-1",
    tenant_id="default",
)
```

That's the whole loop. For the full agent integration pattern — pre-LLM injection scanning, error handling, multi-tenant SaaS, rate limiting — **ask your coding assistant** (see below).

---

## 🤖 Living Docs for AI Assistants

**You shouldn't have to read tons of documentation to use UMA.** Ask your coding agent instead.

UMA ships eight Agent Skills under `.claude/skills/`. They're structured markdown files with YAML frontmatter that Claude Code (and any [Agent Skills](https://docs.claude.com/en/agents-and-tools/agent-skills/overview)-compatible assistant) automatically loads as context when you ask questions about the project. No setup. No `@` mentions. Just ask:

> *"How do I integrate UMA into my chatbot?"*
> → `uma-agent-loop.md` loads — end-to-end pattern with code

> *"What happens when a user sends a prompt injection?"*
> → `uma-security.md` + `uma-quarantine.md` load — full flow from scan to storage

> *"How do I write a custom vector backend?"*
> → `uma-vector-contract.md` loads — the contract, atomicity, score normalization

> *"How do I filter by lane?"*
> → `uma-lanes.md` loads — the six lanes, when to use each

> *"My YAML — can you help me configure Anthropic as the LLM?"*
> → `uma-configure.md` loads — full YAML reference

### The eight skills

| Skill | Covers |
| --- | --- |
| [`uma-overview.md`](.claude/skills/uma-overview.md) | What UMA is, design philosophy, DAT invariants, security primitives at a glance |
| [`uma-api.md`](.claude/skills/uma-api.md) | Full public API — every method, every management function, scope fields |
| [`uma-lanes.md`](.claude/skills/uma-lanes.md) | Six memory lanes, storage contracts, quarantine semantics, retrieval pipeline |
| [`uma-configure.md`](.claude/skills/uma-configure.md) | YAML reference, LLM/embedding providers, security configuration, install surfaces |
| [`uma-security.md`](.claude/skills/uma-security.md) | Two-layer scanning, pattern catalog, severity behavior, integrity verification |
| [`uma-agent-loop.md`](.claude/skills/uma-agent-loop.md) | End-to-end integration: scan → retrieve → LLM → process_turn |
| [`uma-vector-contract.md`](.claude/skills/uma-vector-contract.md) | Vector isolation contract, push-down filters, custom backend authoring |
| [`uma-quarantine.md`](.claude/skills/uma-quarantine.md) | Quarantine lifecycle, management API, composition with trust scoring |

Each skill is under 500 lines, follows the Agent Skills specification (third-person `description` field for discovery), and is verified against the patched codebase — no phantom APIs.

Assistants that don't follow `.claude/skills/` can read the same files directly, or via a symlink at `.agents/skills/` if your tooling uses that path.

---

## License & Status

UMA is Apache-2.0 licensed and currently in beta. Production packaging is not included in this public repo; a future private or commercial package may provide production-specific profiles, managed-service adapters, and deployment tooling.

For the architectural deep dive, see [`ARCHITECTURE.md`](ARCHITECTURE.md). For everything else, ask your assistant.