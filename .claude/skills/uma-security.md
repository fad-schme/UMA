---
name: uma-security
description: Complete security model for UMA — the four-layer defense-in-depth framework for ASI06 memory poisoning (pre-write sanitization, provenance tracking, temporal decay, memory isolation), the two-layer injection-scan architecture (pre-LLM gate + write-time boundary scan), severity behavior and trust-score adjustment, the YAML pattern catalog with all built-in rules, custom pattern extension, MIME consistency and file-size limits at ingest, HTML/Markdown sanitization, the OWASP LLM Top 10 2025 and Agentic AI baseline UMA enforces, and how UMA's defenses compose against prompt injection, vector poisoning, and integrity tampering. Use this skill when answering questions about how UMA defends against prompt injection or memory poisoning, how to extend the pattern catalog, what `severity` levels mean, how `trust_score` is adjusted, what happens to a flagged user message, what `InjectionDetectedError` indicates, how to disable scanning, or any question about the security primitives across the ingest and retrieval paths.
---

# UMA — Security Model

UMA is a memory SDK, not a security boundary against malicious operators. What it defends against is **untrusted content reaching the storage layer or biasing retrieval**: prompt-injection payloads in user messages and ingested documents, vector poisoning, integrity tampering, and cross-tenant leakage.

The security model has five primitives. They compose:

1. **Two-layer injection scanning** — pre-LLM gate + write-time boundary scan
2. **Trust scoring + quarantine** — every artifact carries `trust_score` and `quarantined_at`
3. **Content hashing + integrity verification** — every typed artifact carries `content_hash`
4. **Ingest gating** — MIME consistency, file size limits, HTML sanitization
5. **Retrieval audit** — every retrieve call is logged with a hashed query preview

---

## Defense-in-Depth Against Memory Poisoning (ASI06)

Memory poisoning is a stateful attack: a single injected fact can silently corrupt an agent's behaviour across all future sessions. [Single-layer defenses are rarely enough](https://vectorize.io/articles/how-to-prevent-ai-memory-poisoning) — UMA implements all four layers of the recommended defense-in-depth model:

| Layer | What it means | UMA's implementation |
|---|---|---|
| **1. Pre-Write Sanitization** | Block malicious content before it enters memory stores | Two-layer scanning aligned with [OWASP Agent Memory Guard](https://owasp.org/www-project-agent-memory-guard/): `scan_user_input` (pre-LLM gate) + `scan_artifact_text` at every write boundary. 15-rule catalog. High severity → quarantine + `trust=0.0`. |
| **2. Provenance Tracking** | Tag and trace the origin of every stored artifact; require approval for untrusted inputs | `source_chunk_ids`, `content_hash`, classifier-derived `trust_score`, and `meta.security.audit_log` on every artifact. `verify_integrity` re-derives SHA-256 hashes; mismatch → quarantine. `lint_memory_drift` detects stale or drifted provenance. |
| **3. Temporal Decay** | Reduce influence of older memories so corrupted data doesn't anchor permanently | **Not implemented.** Trust scores are set at write time and do not decay automatically. Time-weighted ranking or TTL is caller responsibility. |
| **4. Memory Isolation** | Strict per-user isolation so one poisoned interaction can't infect others | `tenant_id` / `owner_type` / `owner_id` enforced at every SQL read; LanceDB pushes isolation into the `WHERE` clause before the k-nearest cap — cross-tenant leakage is impossible by construction. |

The primitives in sections below implement layers 1, 2, and 4 end-to-end. Layer 3 (temporal decay) is explicitly out of scope and noted where relevant.

---

## Layer 1 — Pre-LLM Gate (`scan_user_input`)

Call this at the **top of every agent turn**, before `retrieve_context` and before any LLM call. It is synchronous, never raises, and returns a result dict — the caller decides what to do.

```python
from uma import UMAMemory

scan = memory.scan_user_input(user_msg)
# {"severity": "none"|"low"|"medium"|"high", "matched_rules": [...], "score": float}

if scan["severity"] == "high":
    return "I can't process that request."
```

This layer protects the **LLM** from seeing the payload. UMA does not block — it surfaces the signal so your agent can decide.

---

## Layer 2 — Write-Time Boundary Scan

Every storage write boundary re-scans the content. This is defense in depth: even if the pre-LLM gate is skipped or bypassed, nothing flagged reaches durable storage without being marked.

```python
from uma import InjectionDetectedError

try:
    await memory.process_turn(user_id=..., user_msg=..., assistant_reply=..., session_id=...)
except InjectionDetectedError as e:
    # e.severity == "high", e.matched_rules, e.score
    handle_security_event(e)
```

### What gets scanned

| Boundary | What is scanned | When |
|---|---|---|
| `process_turn` (entry) | `user_msg` | Raises `InjectionDetectedError` on high |
| `_store_turn_chunks` | Each turn chunk | Quarantines on high |
| `EpisodicCore.store_episode` | `assistant_reply` | Quarantines on high |
| `SemanticCore.upsert_fact` | Fact text | Reduces trust on low/medium |
| `WorkingMemoryCore.append` | Each message | Quarantines on high (filtered from `get_context`) |
| `ingest_service` document chunks | Each chunk text | Quarantines on high |
| `UMARuntime.retrieve_context` | `query_text` | Severity propagates to downstream LLM hops |

### Severity behavior

| Severity | `scan_user_input` | `process_turn` | Artifact trust | Stored? |
|---|---|---|---|---|
| `none` | `{"severity": "none", ...}` | Proceeds normally | Unchanged | ✅ |
| `low` | `{"severity": "low", ...}` | Logged, proceeds | Reduced by 20% | ✅ |
| `medium` | `{"severity": "medium", ...}` | Logged, proceeds | Reduced by 50% | ✅ |
| `high` | `{"severity": "high", ...}` | Raises `InjectionDetectedError` for `user_msg`; quarantines derived artifacts | Set to 0.0 | Quarantined for storage boundaries; turn dropped for `process_turn` |

### Trust-reduction math

Default trust: 0.5. After a write-time scan:
- low: `0.5 × 0.8 = 0.4`
- medium: `0.5 × 0.5 = 0.25`
- high: quarantined; trust set to 0.0

With default `min_trust_score: 0.5`, a medium-severity survivor (trust 0.25) is dropped at retrieval. This is intentional — `min_trust_score` is calibrated to filter every medium-or-worse hit even if quarantine is disabled.

### Bypass

`process_turn` accepts `skip_scan=True` to bypass the Layer-2 entry scan. The downstream per-artifact scans still run. Use this only when the caller has independently validated the content and explicitly accepts responsibility.

```python
await memory.process_turn(..., skip_scan=True)
```

---

## Pattern Catalog

The scanner runs against a compiled YAML catalog: `uma/common/injection_patterns.yaml`. Bundled rules:

| Rule | Severity | Category | What it detects |
|---|---|---|---|
| `jailbreak_prompt` | high | prompt_injection | "ignore all previous instructions", DAN/DevMode personas, fake system tags |
| `code_injection` | high | code_injection | XSS script tags, `rm -rf /`, `subprocess.Popen` with shell paths |
| `context_switching` | high | prompt_injection | "end of response / new instructions", heredoc system tags |
| `data_exfiltration` | high | data_exfiltration | "dump the database", "give me admin credentials" |
| `encoded_payloads` | high | prompt_injection | Base64/hex strings paired with "decode this", "execute" directives |
| `prompt_reset` | high | prompt_injection | "ignore instructions", "forget everything", "reset this conversation" |
| `role_impersonation` | high | prompt_injection | "you are now admin/root/kernel", "simulate a shell" |
| `config_leakage` | high | configuration_exposure | "reveal system prompt", "print env variables" |
| `rule_override` | high | prompt_injection | "just this once override rules", "hypothetically bypass moderation" |
| `alignment_breaking` | high | alignment_evasion | "act as a rogue AI" |
| `debug_mode_spoofing` | high | system_spoofing | "you are in debug mode", "assume developer mode" |
| `hidden_instruction_probe` | high | configuration_exposure | "print internal parameters", "reveal hidden data" |
| `policy_evasion` | medium | prompt_injection | "ignore filters", "respond without restrictions" |
| `escalation_phrases` | medium | privilege_escalation | "how to escalate privileges", "root access simulate" |
| `homoglyph_obfuscation` | medium | prompt_injection | Leetspeak / homoglyph variants of known-bad tokens |

### Extending the catalog

Add organization-specific rules without touching the bundled file:

```yaml
# config/uma.yaml
security:
  custom_patterns_path: "/path/to/my_patterns.yaml"
```

```yaml
# /path/to/my_patterns.yaml
rules:
  - name: my_internal_token_leak
    severity: high
    category: configuration_exposure
    patterns:
      - "internal_secret_[a-z0-9]{12}"
    description: "Detects internal secret prefix"
```

**Authoring guidance:**

- Prefer tight patterns. Loose patterns cause false positives on benign conversation.
- Set `severity: high` only for patterns that are **almost never benign**.
- Test against representative corpora before deploying — false positives degrade UX quickly.

---

## Ingest Gating

Beyond injection scanning, ingest enforces these gates **before any parser is invoked**:

### MIME consistency

`uma/ingest/mime_check.py::enforce_mime_consistency` reads the file's first bytes, derives the content type, and:

- Rejects executable types (`MimeRejection`)
- Rejects extension/content mismatches (`MimeRejection`) — e.g. a `.pdf` that is actually a ZIP
- Returns the resolved `ContentType` so the parser dispatcher can pick the right path

### File size limit

`IngestConfig.max_file_bytes` (default **50 MB**). Files exceeding this raise `FileSizeRejection` from `enforce_file_size_limit`. The check happens via `os.stat()`; the file's bytes are never opened.

### PDF page count limit

`IngestConfig.pdf_max_pages` (default **5000**). PDFs declaring more pages raise before any text is extracted. Protects against zip-bomb-style PDFs that declare massive page counts.

### HTML / Markdown sanitization

`uma/ingest/parser.py::_sanitize_html` strips:

- `<script>` and `<iframe>` tags
- Inline event handlers (`on*=...`)
- `javascript:` and `data:` URLs
- Conditional comments
- Inline SVG

Per-category removal counts are stored in `meta["security"]["sanitization"]` on the document manifest.

### What happens to a flagged chunk

If a chunk's text scan hits high severity:

1. The chunk is persisted to SQL with `quarantined_at` set.
2. The chunk is **dropped before fact extraction** — injected text cannot seed the semantic lane.
3. The chunk is excluded from every retrieval query (`AND quarantined_at IS NULL`).
4. The hit is recorded in `meta["security"]["injection_scan"]` with rule name and severity.

---

## Trust Scoring at Retrieval

Trust flows through the canonical retrieval pipeline:

```
final_score = (1 - trust_weight) * existing_score + trust_weight * trust_score
```

After this blend, candidates with `trust_score < min_trust_score` are dropped before truncation.

| Source | Default `trust_score` | Notes |
|---|---|---|
| `user_msg` facts | 0.9 | User said it directly |
| `assistant_reply` facts | 0.7 | Assistant may synthesize / hallucinate |
| Document chunks | 0.5 | Adjusted by `score_source` classifier per ingest config |
| Injection-low survivor | × 0.8 | Reduced |
| Injection-medium survivor | × 0.5 | Reduced; below default `min_trust_score` |
| Injection-high | 0.0 | Quarantined; excluded from retrieval |
| Integrity-failure | 0.0 | Quarantined via `verify_integrity` |

`trust_weight` in `config/uma.yaml`:

```yaml
retrieval:
  trust_weight: 0.15
  min_trust_score: 0.5
```

Setting `trust_weight=0` disables trust influence on ranking. The `min_trust_score` floor still applies.

---

## Vector Isolation (C1)

Cross-tenant access is impossible **by construction**:

- **LanceDB** promotes `tenant_id`, `owner_type`, `owner_id` to first-class indexed columns. Queries push these into the engine's `WHERE` clause via DuckDB SQL (single-quote escaped) before the candidate cap is applied. The k-nearest cap can never starve a tenant.
- **InMemory** keeps isolation in a parallel `_scopes` dict; isolation is the first filter applied in the query loop.
- **FAISS** does not support pushed-down predicates. The adapter oversamples (`k × 4`) and post-filters in Python — acceptable for single-tenant deployments; LanceDB is recommended for multi-tenant.

All three adapters refuse empty isolation values at upsert time. See `uma-vector-contract.md` for the full contract.

---

## Retrieval Audit

Every retrieval call (`retrieve_context`, `retrieve_memory`) is recorded by default:

```python
from uma.api.management import list_retrieval_audit

rows = await list_retrieval_audit(
    memory,
    tenant_id="default",
    user_id="user-123",
    limit=100,
)
# Each row: timestamp, operation, tenant_id, user_id, session_id,
#           query_hash, query_preview, scan_severity, result_count, llm_hops_skipped
```

The audit table records:

- A **truncated hashed preview** of `query_text` — not the raw text. Hash is SHA-256.
- The pre-LLM scan severity (`scan_severity`).
- Whether downstream LLM hops were skipped (`llm_hops_skipped`).
- Scope: `tenant_id`, `user_id`, `session_id`.

Disable via `security.retrieval_audit_enabled: false` in your YAML. The default is on.

---

## Integrity Verification

Every typed artifact (Fact, Episode, Skill, Chunk) carries a canonical SHA-256 `content_hash` set at write time. `verify_integrity` re-derives the hash and compares:

```python
from uma.api.management import verify_integrity

result = await verify_integrity(
    memory,
    record_id="fact-abc",
    lane="semantic",
    owner_type="user", owner_id="user-123",
    tenant_id="default",
)

# result.status == "verified" | "failed"
# On mismatch: record is quarantined, audit log entry appended
```

`lint_memory_drift` automatically routes typed lane artifacts through `verify_integrity` so batch integrity checks can be run without calling the function directly.

Background scanning across the full dataset is an Enterprise capability and is not part of the public SDK.

---

## Rate Limiting (M6)

UMA exposes a single optional hook for SDK-level throttling:

```python
def hook(operation, ctx):
    if too_many_calls(ctx):
        raise RuntimeError("rate limited")

memory.set_rate_limit_hook(hook)
```

The hook fires at the top of `retrieve_context`, `retrieve_memory`, `process_turn`, and `ingest_document`. Both sync and async hooks are supported. **The hook raises to refuse**; returning normally allows the call.

UMA ships no default rate limiter. Operators integrate with their existing throttling stack (Redis, Envoy, in-process LRU counter, etc.).

See `uma-api.md` for the full hook signature.

---

## What UMA Does NOT Defend Against

Stating this explicitly so expectations are right:

- **Temporal decay / trust decay over time.** Trust scores are set at write time and do not automatically decrease as memories age. If your threat model includes agents being permanently anchored to older or corrupted data, implement time-weighted ranking or TTL expiry at the caller layer.
- **Malicious local operators.** UMA defends artifacts in motion through the pipeline. It does not sandbox the developer running it.
- **Network-level attacks against your LLM/embedding provider.** Use TLS, scope API keys, etc. — UMA does not.
- **Side-channel attacks against the SQLite database file.** Filesystem permissions are the operator's responsibility.
- **Adversarial example attacks against the embedding model itself.** No general defense exists in the public literature.
- **Dynamic obfuscation past the pattern catalog.** Pattern-based defenses are best-effort. Pair with allow-listing and your own model-based classifier for high-risk deployments.

The model is intentionally minimal: enforce the things that have construction-level fixes, surface signals for the things that don't.
