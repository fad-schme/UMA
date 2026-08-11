```
                                 _   _ __  __   _  
                                | | | |  \/  | /_\ 
                                | |_| | |\/| |/ _ \
                                 \___/|_|  |___/ \_\
```

> ## Live documentation
>
> To help you get up to speed quickly, UMA ships interactive documentation as a set of skills that can answer your questions directly in your workflow.
>
> 🌐 Website: [uma.ai-mem-engineering.com](https://uma.ai-mem-engineering.com)  
> 📄 Full docs: [uma.ai-mem-engineering.com/docs.html](https://uma.ai-mem-engineering.com/docs.html)


## Universal Memory Architecture

UMA is a memory and context runtime SDK for developers building AI agents. It ingests data and exposes six public `lane_filter` lanes. The planner also uses `profile` (a semantic-store projection) and optional `graph`, for eight architectural lane names in total. **UMA manages memory only** — your application owns prompts, tool use, reasoning, and final responses.

UMA does not generate assistant replies and does not perform agent reasoning.
Developers bring their own LLM or agent loop and use UMA strictly for memory
management.

---

## ✨ Why UMA

- 🧠 **Six public filter lanes, eight architectural lane names** — the public `lane_filter` values are working memory, raw, semantic, episodic, procedural, and wiki; the planner also uses profile and optional graph
- 🪶 **Embedded storage by default** — SQLite + LanceDB require no separate storage service. Configure a local or remote LLM and embedding provider for model-backed operations.
- 🛡️ **Security by design** — every artifact is owner-scoped, injection-scanned, trust-scored, and content-hashed before it touches storage.
- 🔍 **Evidence-backed retrieval** — every fact carries provenance back to source chunks. No silent degradation into "vibes-based" RAG.
- 🏢 **Multi-tenant by construction** — cross-tenant access is impossible at the storage layer, not by application-layer convention.

---

## 🏛️ Architecture

![UMA architecture diagram](https://raw.githubusercontent.com/fad-schme/UMA/main/assets/uma_architecture.png)

UMA is a thin SDK around three concerns: **ingest** (data flows in, gets scanned, chunked, embedded), **storage** (SQLite is authoritative, LanceDB is a rebuildable accelerator), and **retrieval** (a canonical pipeline through candidate discovery, fusion, trust-aware ranking, and snippet rendering). Every write boundary scans for prompt injection; every read boundary enforces tenant/owner isolation and filters quarantined records.

For the full architectural model — invariants, pipelines, the vector isolation contract, and the OWASP Top 10 mapping — see [`ARCHITECTURE.md`](https://github.com/fad-schme/UMA/blob/main/ARCHITECTURE.md).

---

## 🛡️ Security by Design

Security in UMA isn't a feature — it's the shape of every code path. Five primitives compose:

1. **Two-layer injection scanning** — pre-LLM advisory gate + write-time defense-in-depth
2. **Trust scoring + quarantine** — every artifact carries a trust score and quarantine flag; retrieval excludes quarantined records by construction
3. **Content hashing + integrity verification** — SHA-256 on every typed artifact; on-demand verification quarantines tampered records
4. **Ingest gating** — MIME consistency, file size caps, HTML/Markdown sanitization
5. **Retrieval audit log** — every retrieve call is recorded with a hashed query preview

The pre-LLM gate is advisory: it reports a signal for the caller's policy. The write-time boundary scan is the enforcing control.

### Defense-in-Depth Against Memory Poisoning (ASI06)

Memory poisoning is a stateful attack — a single injection can permanently corrupt an agent's knowledge base across all future sessions. Single-layer defenses are rarely enough. UMA implements the [four-layer defense-in-depth model](https://vectorize.io/articles/how-to-prevent-ai-memory-poisoning) recommended for production agent memory:

| Layer | What it means | UMA's implementation |
| --- | --- | --- |
| **1. Pre-Write Sanitization** | Block malicious content before it enters memory stores | Two-layer injection scanning: advisory pre-LLM gate (`scan_user_input`) + write-time boundary scan (`scan_artifact_text`) at every storage boundary. Bundled English, French, Spanish, German, and Simplified Chinese YAML catalogs are aligned with [OWASP Agent Memory Guard](https://owasp.org/www-project-agent-memory-guard/). High severity → quarantine + trust=0.0; medium/low → trust reduction. |
| **2. Provenance Tracking** | Tag and trace the origin of every stored artifact | Every Fact, Episode, Skill, and Chunk carries `source_chunk_ids`, `content_hash`, `trust_score`, and a `meta.security.audit_log`. `verify_integrity` re-derives SHA-256 hashes and quarantines on mismatch. `lint_memory_drift` detects compiled artifacts whose raw evidence has drifted. Source trust is classifier-derived: user turn (0.9) > document (0.7) > assistant reply (0.7) > tool output (0.5). |
| **3. Temporal Decay** | Reduce influence of older memories so corrupted data doesn't anchor permanently | **Not implemented in UMA.** Trust scores are set at write time and do not decay automatically. Time-weighted ranking or TTL policies are caller responsibility. |
| **4. Memory Isolation** | Strict per-user isolation so one poisoned interaction can't infect others | `tenant_id` / `owner_type` / `owner_id` enforced at every SQL read and at the vector layer (LanceDB pushes isolation into the `WHERE` clause before the k-nearest cap — cross-tenant leakage is impossible by construction, not by application convention). |

### Mapping to OWASP Top 10 for LLM Applications 2025

The [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/) is the de-facto reference for AI application security. UMA is a memory SDK — not every category applies. Here's the honest mapping:

| OWASP 2025 Category | Scope | UMA's contribution |
| --- | --- | --- |
| 🟢 **ASI03: Identity & Privilege Abuse** (Agentic AI) | Partial — memory-layer | Explicit `tenant_id` / `owner_type` / `owner_id` on every artifact, enforced at the storage layer. Agent identity itself is the caller's concern. |
| 🟢 **ASI05: Unexpected Code Execution** (Agentic AI) | Partial — ingest-only | User input and file injection scanning. MIME consistency check rejects executables; HTML/Markdown sanitized before storage. |
| 🟢 **ASI06: Memory Poisoning** (Agentic AI) | In scope | User input and file injection scanning + quarantine at every storage boundary. Quarantined artifacts never enter retrieval and never seed fact extraction. |
| 🟢 **LLM01: Prompt Injection** | In scope | Two-layer scanning: advisory pre-LLM gate (`scan_user_input`) + write-time per-artifact scan. High severity → quarantine; medium/low → trust reduction. |
| 🟢 **LLM02: Sensitive Information Disclosure** | Partial | Audit log stores SHA-256-hashed query previews only. HTML sanitization strips scripts and active URLs at ingest. |
| 🟢 **LLM04: Data and Model Poisoning** | In scope (RAG path) | Quarantined chunks dropped before fact extraction. SHA-256 `content_hash` + `verify_integrity` detect post-hoc tampering. |
| 🟢 **LLM08: Vector and Embedding Weaknesses** | In scope — primary | The C1 isolation contract: LanceDB pushes `tenant_id` / `owner_type` / `owner_id` as a SQL `WHERE` clause into the engine *before* the k-nearest cap — without this, heavy users in one tenant would occupy top-k globally and starve others. SQL stores add the same filter on every read path. Cross-tenant leakage is impossible by construction. User input and file injection scanning also addresses the RAG poisoning problem. |
| 🟢 **LLM09: Misinformation** | Partial | Every fact carries provenance back to source chunks. `LatestWinsFactResolver` picks the canonical row by most-recent `updated_at`; quarantined facts are excluded from retrieval at the SQL layer (`AND quarantined_at IS NULL`) so they never surface to callers even if chosen as canonical. |
| 🟢 **LLM10: Unbounded Consumption** | Partial | Ingest side: `max_file_bytes` and `pdf_max_pages` cap resource use — UMA-owned. Retrieval side: `set_rate_limit_hook` exposes a single plug-point on every public method for the caller's own rate limiter. UMA ships no default limiter and owns no throttling policy — the caller decides accounting, storage, timeouts, and refusal semantics. |

**Three of ten ASI categories apply.** UMA is a memory SDK, not an agent. The remaining seven belong to the agent layer above UMA — they require tool use, autonomy, or inter-agent communication that UMA doesn't have.


**Six of ten LLM categories apply** (LLM01, LLM02 partial, LLM04, LLM08, LLM09 partial, LLM10 partial). There is no security theater — the four categories marked out of scope genuinely require capabilities UMA does not have: output rendering (LLM05), autonomous tool use (LLM06), system prompt management (LLM07), and model supply-chain procurement (LLM03). UMA makes an adjacent contribution to LLM03 at the document ingest boundary, but the core supply-chain threat — model provenance and dependency integrity — belongs to the layer above UMA.

**Scanner evaluation status.** The injection catalog is currently regression-tested against an internal smoke corpus of 42 known attack strings and 33 benign controls (see `tests/test_security_injection.py`). Public benchmark results against adversarial-injection corpora (LOCOMO, TensorTrust, HackAPrompt) are in progress and will be published with precision / recall / F1 numbers by corpus and UMA version when complete. Do not read the smoke-corpus pass rate as a general-purpose accuracy claim.

For the full security model — including the injection pattern catalog, severity behavior, quarantine lifecycle, and integrity verification — see [`.claude/skills/security.md`](https://github.com/fad-schme/UMA/blob/main/.claude/skills/security.md) for the deep dive, or [`ARCHITECTURE.md`](https://github.com/fad-schme/UMA/blob/main/ARCHITECTURE.md) for the architectural model.

---

## Quickstart

The distribution is `uma-mem`; the import package and CLI are `uma`.

```bash
pip install uma-mem
```

From a source checkout:

```bash
pip install -e .
```

```python
import asyncio

from uma import UMAMemory


async def main():
    # Pass the path to your uma.yaml — any accessible location works.
    memory = UMAMemory.from_yaml("/path/to/your/uma.yaml").set_context(agent_id="my-agent")

    user_message = "..."   # your inbound turn

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


asyncio.run(main())
```

That's the whole loop. For the full agent integration pattern — pre-LLM injection scanning, error handling, multi-tenant SaaS, rate limiting — **ask your coding assistant** (see below).

If a storage adapter needs credentials, `uma.yaml` also accepts an optional `secrets:` block; the reference shape lives in [`.claude/skills/configure.md`](https://github.com/fad-schme/UMA/blob/main/.claude/skills/configure.md).

---

## Command-line interface

Installing UMA provides both the `uma` executable and the equivalent
`python -m uma.cli` entry point. Global options precede the command:

```bash
uma --config /path/to/uma.yaml --format json config validate
```

`--config` falls back to `UMA_CONFIG`, then `./uma.yaml` and
`./config/uma.yaml`. `--format` accepts `text` (default) or `json`.

| Command | Purpose |
| --- | --- |
| `uma version` | Show the installed UMA version without loading a runtime. |
| `uma config validate` / `config show` | Validate configuration or show it with secret values redacted. |
| `uma doctor --offline` | Check configuration and local dependencies without creating databases or initializing providers. |
| `uma doctor` / `uma health` | Initialize UMA and call `health_check()`. This verifies initialization, dimensions, stores, vectors, and graph state; it does not make a paid provider-generation request. |
| `uma security scan TEXT` | Run the injection scanner. Use exactly one of `TEXT`, `--file`, or `--stdin`. |
| `uma dev check` | Run predefined `quick` or `full` development checks without installing tools or applying fixes. |
| `uma retrieve context` / `retrieve memory` | Run agent/user-scoped retrieval and report the retrieval-audit write effect. |
| `uma ingest document` / `turn` / `memory-bootstrap` / `diary-bootstrap` | Run the corresponding scoped public ingestion API. |
| `uma audit list` / `quarantine list` | List records within one resolved tenant or durable-owner scope. |
| `uma quarantine reinstate` / `purge` | Mutate exactly one tenant/owner/lane/record target. Purge requires `--reason`. |
| `uma index rebuild-vectors` / `rebuild-derived` | Rebuild all records in one exact tenant/owner/lane scope. |
| `uma integrity enforce` | Verify one exact record and quarantine it if its stored hash mismatches. |

Request scope uses `--tenant`, `--agent`, `--user`, `--session`,
`--workspace`, and `--request-id`. Durable owner scope is independent and
uses `--tenant`, `--owner-type`, and `--owner-id`. Tenant defaults to
`UMA_TENANT_ID`, then `default`; administrative owner scope is never inferred
from request scope.

Reinstate, purge, index rebuilds, and integrity enforcement print the exact
resolved target to stderr and require confirmation. Non-interactive use must
pass `--yes`. Their JSON results include an `effects` list describing possible
writes.

---

## MCP client support

UMA ships an MCP server so any MCP-compatible AI client — coding agents, chat
clients, and cloud connectors — can talk to your memory layer. The server is a
thin adapter over UMA's public API; every tool call goes through the same
`(tenant_id, user_id)` scope resolution as any other caller.

Install with the `mcp` optional extra:

```bash
pip install 'uma-mem[mcp]'                # stdio + HTTP with bearer tokens
pip install 'uma-mem[mcp,oauth]'          # + OAuth 2.1 for ChatGPT
```

The `uma-mcp` binary is now on PATH. Point it at your `uma.yaml`:

```bash
uma-mcp                                    # stdio (Claude Code, Codex, ...)
uma-mcp --http --port 3131                 # HTTP + opaque bearer tokens
uma-mcp --http --port 3131 \
        --oauth-issuer https://your-idp/   # HTTP + OAuth 2.1 JWT (ChatGPT)
        --oauth-audience https://your-brain/mcp
```

Tools exposed: `retrieve_context`, `retrieve_memory`, `process_turn`,
`ingest_document`, `health_check`. Every response is well-formed JSON matching
the corresponding `uma.common.results` model.

### Supported clients

| Client                              | Transport             | Doc |
| ----------------------------------- | --------------------- | --- |
| Claude Code                         | stdio                 | [`STDIO_CLIENTS.md`](docs/mcp/STDIO_CLIENTS.md) |
| Codex                               | stdio                 | [`STDIO_CLIENTS.md`](docs/mcp/STDIO_CLIENTS.md) |
| Cursor                              | stdio                 | [`STDIO_CLIENTS.md`](docs/mcp/STDIO_CLIENTS.md) |
| Windsurf                            | stdio                 | [`STDIO_CLIENTS.md`](docs/mcp/STDIO_CLIENTS.md) |
| Claude Desktop (local bridge)       | stdio                 | [`STDIO_CLIENTS.md`](docs/mcp/STDIO_CLIENTS.md) |
| Claude Desktop (remote connector)   | HTTP + bearer         | [`DEPLOY.md`](docs/mcp/DEPLOY.md) |
| Claude Cowork                       | HTTP + bearer         | [`DEPLOY.md`](docs/mcp/DEPLOY.md) |
| Perplexity                          | HTTP + bearer / OAuth | [`DEPLOY.md`](docs/mcp/DEPLOY.md) / [`CHATGPT.md`](docs/mcp/CHATGPT.md) |
| ChatGPT                             | HTTP + OAuth 2.1      | [`CHATGPT.md`](docs/mcp/CHATGPT.md) |

Bearer tokens are opaque, SHA-256-hashed in a local SQLite store, and issued
via `uma auth create <label> --user USER [--tenant TENANT]`. OAuth 2.1 mode
points UMA at any RFC 8414-compliant IdP (Auth0, Microsoft Entra ID, Google,
Okta, Keycloak, Authentik) — UMA acts as a pure resource server per the
current MCP spec direction and never issues tokens itself.

### Security posture

Every HTTP request resolves to an explicit `(tenant_id, user_id)` before it
touches the memory layer — the same DAT invariant every other UMA call
enforces. Tokens are never logged; only their short `token_id` handle appears
in server logs. Bearer plaintext is shown exactly once at issue time by
`uma auth create` and is not recoverable from the store. JWT verification
uses PyJWT with an explicit RS256/ES256 algorithm allowlist — HS256 is
rejected to close the algorithm-confusion attack.

Full model in [`docs/mcp/DEPLOY.md`](docs/mcp/DEPLOY.md) (bearer / cloud
clients) and [`docs/mcp/CHATGPT.md`](docs/mcp/CHATGPT.md) (OAuth 2.1 recipe
with per-IdP flag sets).

-

---

## Testing and quality gates

The default suite is hermetic: it uses fake LLM and embedding providers, never
contacts a model service, and is safe to run anywhere.

```bash
pip install -e '.[dev]'
python -m pytest -q
```

Model-dependent quality is measured separately by two opt-in gates, so CI never
depends on a model being available. Both require a local Ollama and are skipped
unless `RUN_E2E=1` is set.

| Gate | Measures | Published baseline |
| --- | --- | --- |
| [`test_fact_extraction_quality.py`](tests/e2e/test_fact_extraction_quality.py) | How much the extractor gets out of a passage | micro precision **0.2500**, micro recall **0.3333** |
| [`test_retrieval_quality.py`](tests/e2e/test_retrieval_quality.py) | Whether `retrieve_context` returns the right source pages | r-precision **0.7353**, recall@3 **0.9118** |

```bash
pip install -e '.[dev,e2e]'
RUN_E2E=1 python -m pytest tests/e2e -q -s
```

Each gate publishes a machine-readable metrics line and enforces thresholds
pinned just below its measured baseline, so a regression fails the run without
the threshold implying the current number is good. The extraction baseline in
particular is low, and it is published as-is.

**These numbers are narrow claims.** Each covers one small corpus against one
small model — `qwen2.5:3b`, plus `nomic-embed-text` for retrieval. Read
[`tests/e2e/README.md`](tests/e2e/README.md) before citing either — it records
the corpora, the gold methodology, and specifically what each metric does and
does not show, including why retrieval recall is near-saturated on a corpus this
size and why fixed-cutoff precision is not reported.

Retrieval and extraction quality are separate from the injection scanner's
evaluation status, described under Security by Design above. Neither gate
retires that claim.

---

## 🤖 Living Docs for AI Assistants

**You shouldn't have to read tons of documentation to use UMA.** Ask your coding agent instead.

UMA ships nine Agent Skills under `.claude/skills/`. They're structured markdown files with YAML frontmatter that Claude Code (and any [Agent Skills](https://docs.claude.com/en/agents-and-tools/agent-skills/overview)-compatible assistant) automatically loads as context when you ask questions about the project. No setup. No `@` mentions. Just ask:

> *"How do I integrate UMA into my chatbot?"*
> → `agent-loop.md` loads — end-to-end pattern with code

> *"What happens when a user sends a prompt injection?"*
> → `security.md` + `quarantine.md` load — full flow from scan to storage

> *"How do I write a custom vector backend?"*
> → `vector-contract.md` loads — the contract, atomicity, score normalization

> *"How do I filter by lane?"*
> → `lanes.md` loads — the six public filter lanes plus the profile and optional graph views

> *"How does fact promotion work?"*
> → `promotion.md` loads — agent profiles, eligibility gates, scope changes, and provenance

> *"My YAML — can you help me configure Anthropic as the LLM?"*
> → `configure.md` loads — full YAML reference

### The nine skills

| Skill | Covers |
| --- | --- |
| [`overview.md`](.claude/skills/overview.md) | What UMA is, design philosophy, DAT invariants, security primitives at a glance |
| [`api.md`](.claude/skills/api.md) | Full public API — every method, every management function, scope fields |
| [`lanes.md`](.claude/skills/lanes.md) | Six public filter lanes plus the profile and optional graph planner/plugin views |
| [`configure.md`](.claude/skills/configure.md) | YAML reference, LLM/embedding providers, security configuration, install surfaces |
| [`security.md`](.claude/skills/security.md) | Two-layer scanning, pattern catalog, severity behavior, integrity verification |
| [`agent-loop.md`](.claude/skills/agent-loop.md) | End-to-end integration: scan → retrieve → LLM → process_turn |
| [`promotion.md`](.claude/skills/promotion.md) | Public agent-profile API, promotion gates, ownership changes, and provenance |
| [`vector-contract.md`](.claude/skills/vector-contract.md) | Vector isolation contract, push-down filters, custom backend authoring |
| [`quarantine.md`](.claude/skills/quarantine.md) | Quarantine lifecycle, management API, composition with trust scoring |

Each skill is under 500 lines, follows the Agent Skills specification (third-person `description` field for discovery), and is verified against the patched codebase — no phantom APIs.

Assistants that don't follow `.claude/skills/` can read the same files directly, or via a symlink at `.agents/skills/` if your tooling uses that path.

---

### Other docs:

- [**UMA Documentation**](https://uma.ai-mem-engineering.com/docs.html)
- [**Contributing**](CONTRIBUTING.md)


For the architectural deep dive, see [`ARCHITECTURE.md`](ARCHITECTURE.md). For everything else, ask your assistant.

This repository is licensed under the [Apache-2.0 License](LICENSE).
