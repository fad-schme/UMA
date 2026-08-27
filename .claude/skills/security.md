---
name: uma-security
description: Complete security model for UMA — the seven security primitives (provenance, write-time trust scoring, cryptographic integrity, injection pattern detection, two-layer injection gate, quarantine, ingest boundary hardening), the full OWASP LLM Top 10 2025 mapping (LLM01/LLM04/LLM08 in scope; LLM02/LLM09/LLM10 partial; LLM03/LLM05/LLM06/LLM07 out of scope with explicit reasoning), the OWASP ASI06/ASI03/ASI05 coverage, the two-layer injection-scan architecture (pre-LLM gate + write-time boundary scan), severity behavior and trust-score adjustment, the bundled English and multilingual YAML pattern catalogs, custom pattern extension, MIME consistency and file-size limits at ingest, HTML/Markdown sanitization, and how UMA's defenses compose against prompt injection, vector poisoning, and integrity tampering. Use this skill when answering questions about how UMA defends against any OWASP LLM or ASI control, how to extend the pattern catalog, what `severity` levels mean, how `trust_score` is adjusted, what happens to a flagged user message, what `InjectionDetectedError` indicates, which OWASP controls are out of scope and why, or any question about the security primitives across ingest and retrieval.
---

# UMA — Security Model

UMA is a memory SDK, not a security boundary against malicious operators. What it defends against is **untrusted content reaching the storage layer or biasing retrieval**: prompt-injection payloads in user messages and ingested documents, vector poisoning, integrity tampering, and cross-tenant leakage.

The security model has five primitives. They compose:

1. **Two-layer injection scanning** — pre-LLM gate + write-time boundary scan
2. **Trust scoring + quarantine** — every artifact carries `trust_score` and `quarantined_at`
3. **Content hashing + integrity verification** — every typed artifact carries `content_hash`
4. **Ingest gating** — MIME consistency, file size limits, HTML sanitization
5. **Retrieval audit** — every retrieve call is logged with a query digest and a bounded preview

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

The scanner compiles `uma/adapters/scanner/injection_patterns.yaml`, `uma/adapters/scanner/injection_patterns.l10n.yaml`, and `uma/adapters/scanner/sqli_patterns.yaml`. The localized catalog covers French, Spanish, German, and Simplified Chinese; the SQLi catalog adds compound SQL and NoSQL rules with proximity constraints. Bundled English prompt-injection rule families:

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
| `delimiter_smuggling` | medium | obfuscation | Repeated zero-width formatting markers or ASCII controls used to hide payloads |
| `prompt_artifact_smuggling` | medium | prompt_injection | Exact privileged-role protocol tokens such as `<|system|>` and ChatML/Llama system delimiters |

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

`score_source` (`uma/common/trust.py`) assigns the base score per kind at write time:

| Source kind | Default `trust_score` | Notes |
|---|---|---|
| `turn_user` | 0.9 (0.5 if unauthenticated) | User said it directly in an authenticated session |
| `turn_assistant` | 0.7 (0.5 if unauthenticated) | Assistant may synthesize / hallucinate |
| `document` | 0.7 | Chunks and facts from `ingest_document` |
| `bootstrap_memory` / `bootstrap_diary` | 0.8 manual / 0.6 default | Depends on `import_mode` |
| `tool_output` | 0.5 | No production call site yet |
| `promotion` | Inherits parent, default 0.5 | Copied fact keeps the source's score |

Independently, injection scanning reduces `trust_score` on the same artifact after it's written:

| Scan severity | Effect | Notes |
|---|---|---|
| Low | × 0.8 | Reduced |
| Medium | × 0.5 | Reduced; below default `min_trust_score` |
| High | 0.0 | Quarantined; excluded from retrieval |
| Integrity failure | 0.0 | Quarantined via `verify_integrity` |

`trust_weight` in `config/uma.yaml`:

```yaml
retrieval:
  trust_weight: 0.15
  min_trust_score: 0.5
```

Setting `trust_weight=0` disables trust influence on ranking. The `min_trust_score` floor still applies.

---

## Vector Isolation (C1)

Cross-tenant access is enforced at the storage layer, not by application convention:

- **LanceDB** promotes `tenant_id`, `owner_type`, `owner_id` to first-class indexed columns. Queries push these into the engine's `WHERE` clause via DuckDB SQL (single-quote escaped) before the candidate cap is applied, so the k-nearest cap can't starve a tenant.
- **InMemory** keeps isolation in a parallel `_scopes` dict; isolation is the first filter applied in the query loop.
- **FAISS** does not support pushed-down predicates. The adapter oversamples (`k × 4`) and post-filters in Python — acceptable for single-tenant deployments; LanceDB is recommended for multi-tenant.

All three adapters refuse empty isolation values at upsert time. See `vector-contract.md` for the full contract.

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

See `api.md` for the full hook signature.

---

## OWASP LLM Top 10 Coverage

UMA addresses 6 of the 10 OWASP Top 10 for LLM Applications 2025 categories. The mapping below is the honest accounting — out-of-scope categories are listed because stating them explicitly is more useful than silence.

| Control | Scope | Mechanism |
|---|---|---|
| **LLM01 Prompt Injection** | In scope | Two-layer gate: `scan_user_input` (pre-LLM, advisory, never raises) + `process_turn` write-time rescan. High severity drops the turn entirely (`InjectionDetectedError`); lower severities reduce trust score. Every document chunk and episode also scanned at write boundary via `scan_artifact_text`. |
| **LLM02 Sensitive Information Disclosure** | Partial | Retrieval audit log stores SHA-256-hashed `query_text` preview, never raw text. HTML/Markdown sanitization strips scripts and active URLs at ingest. UMA does not control what the calling application puts into prompts or what the LLM returns. |
| **LLM03 Supply Chain** | Out of scope (document boundary adjacent) | UMA has no model training, fine-tuning, or plugin registry — core supply chain is out of scope. Adjacent: `PickleParser` removed (arbitrary code execution risk); MIME consistency checks reject executables before parsing. |
| **LLM04 Data and Model Poisoning** | In scope | Quarantined chunks excluded from fact extraction — injected text cannot seed the semantic lane. SHA-256 `content_hash` on every artifact; `verify_integrity` quarantines on mismatch. `AND quarantined_at IS NULL` in every retrieval query across all four SQL stores. |
| **LLM05 Improper Output Handling** | Out of scope | UMA returns context packs, not generated outputs. The calling application owns rendering, escaping, and output validation. |
| **LLM06 Excessive Agency** | Out of scope | UMA has no tool use, function calling, or autonomous action capability. Pure memory SDK. |
| **LLM07 System Prompt Leakage** | Out of scope | System prompts live entirely in the calling application. UMA never sees them. |
| **LLM08 Vector and Embedding Weaknesses** | In scope — primary | C1 isolation contract: `tenant_id` / `owner_type` / `owner_id` pushed as a SQL `WHERE` clause into the vector engine *before* the k-nearest cap is applied — a heavy tenant cannot occupy top-k and starve others. All three adapters (LanceDB, FAISS, InMemory) refuse empty isolation values at upsert. SQL stores apply the same filter on every read path. Two boundary-filter gaps that could have let a mismatched tenant slip through were found and fixed (see CHANGELOG); the enforcement described here is the current, patched behavior. Write-time injection scanning directly addresses the RAG poisoning sub-problem. |
| **LLM09 Misinformation** | Partial | Every fact carries `source_chunk_ids` (provenance back to source). Quarantined facts excluded at SQL retrieval layer. `provenance_valid` is a top-level field on every `retrieve_memory` result. UMA cannot prevent the LLM from hallucinating — it provides the provenance infrastructure to detect and verify. |
| **LLM10 Unbounded Consumption** | Partial | Ingest side (UMA-owned): `max_file_bytes` (default 50 MB) and `pdf_max_pages` (default 5000) cap resource use. Retrieval side (caller-owned): `set_rate_limit_hook` exposes a single plug-point that fires at the top of `retrieve_context`, `retrieve_memory`, `process_turn`, and `ingest_document`. UMA ships no default rate limiter and owns no throttling policy — the caller decides accounting, storage, timeouts, and refusal semantics. |

---

## What UMA Does NOT Defend Against

Stating this explicitly so expectations are right:

- **Malicious local operators.** UMA defends artifacts in motion through the pipeline. It does not sandbox the developer running it.
- **Network-level attacks against your LLM/embedding provider.** Use TLS, scope API keys, etc. — UMA does not.
- **Side-channel attacks against the SQLite database file.** Filesystem permissions are the operator's responsibility.
- **Adversarial example attacks against the embedding model itself.** No general defense exists in the public literature.
- **Dynamic obfuscation past the pattern catalog.** Pattern-based defenses are best-effort. Pair with allow-listing and your own model-based classifier for high-risk deployments.

The model is intentionally minimal: enforce the things that have construction-level fixes, surface signals for the things that don't.
