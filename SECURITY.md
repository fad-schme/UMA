# Security Policy

UMA is a memory and context runtime SDK for AI agents. Security is a
first-class design property, not a feature bolted on — but the honest scope
of what UMA defends against, and what it deliberately does not, matters.
This document states both.

For the architectural model, see [`ARCHITECTURE.md § Security Architecture`](ARCHITECTURE.md#security-architecture)
and the full deep dive in [`.claude/skills/security.md`](.claude/skills/security.md).

---

## Reporting a Vulnerability

**Please email [security@ai-mem-engineering.com](mailto:security@ai-mem-engineering.com).**

Do not file GitHub issues for security-relevant findings — those are public
the moment they are opened. If you would like to send encrypted mail, ask
for our current public key at the same address.

Please include:

- A clear description of the issue and the impact you observed
- Steps to reproduce (a minimal test case or PoC input if applicable)
- The UMA version (`python -c "import uma; print(uma.__version__)"`) and
  Python version
- Whether the finding affects the ingest boundary, the retrieval
  boundary, storage, or configuration

Acknowledgement will come from a human, not an autoresponder.

---

## Supported Versions

UMA is currently in **beta**. Only the latest published beta receives
security fixes. There are no backwards-compatibility guarantees for
schema or API between betas — see `ARCHITECTURE.md § Status`.

| Version           | Status              | Security fixes  |
|-------------------|---------------------|-----------------|
| 0.1.5-beta        | Current             | ✅ Yes          |
| < 0.1.5-beta      | Superseded          | ❌ Please upgrade |

Once UMA reaches 1.0, this table will grow into a proper support matrix
with a defined maintenance window per minor release.

---

## Triage Timeline

The following targets are on business days (Mon–Fri, excluding public
holidays) from the timestamp of your initial email:

| Stage                                            | Target |
|--------------------------------------------------|--------|
| Acknowledgement of receipt                       | 3 days |
| Initial assessment (severity, in-scope decision) | 10 days |
| Fix development for confirmed **critical** issues | 30 days |
| Fix development for confirmed **high** issues     | 60 days |
| Coordinated public disclosure after fix ships    | ≥ 14 days |

If we cannot meet a target, you will hear from us with a revised
timeline and reasoning. We will not disclose your report or your identity
without your consent. Where a finding affects downstream users or
integrators, we coordinate disclosure with you.

Findings that are out of scope (see below) receive a written explanation
of why, not silence.

---

## Trust Boundary

UMA's threat model has one guiding principle: **the operator is trusted;
data flowing through the operator's process is not.**

### Treated as adversarial (scanned, sanitized, quarantined)

Every piece of content that enters UMA from outside the operator's own
process is treated as potentially hostile. This is the surface where the
five security primitives live (see `ARCHITECTURE.md § Security
Architecture`):

- **User messages** — `user_msg` in `process_turn`. Scanned at the
  storage boundary; high severity raises `InjectionDetectedError` and
  the turn is dropped entirely.
- **Assistant replies** — `assistant_reply` in `process_turn`. Trust
  starts at `0.7` because the assistant may synthesize or hallucinate.
  Scanned at the episode-write boundary.
- **Query text** — `query_text` in `retrieve_context` and
  `retrieve_memory`. Scan severity propagates to downstream LLM hops
  (snippet refiner, fact pruner) which skip amplification on
  medium/high.
- **Ingested document content** — every chunk from `ingest_document`.
  MIME/extension consistency, file-size cap, PDF page cap enforced
  before parsing. HTML and Markdown sanitized (`<script>`, `<iframe>`,
  inline event handlers, `javascript:` / `data:` URLs stripped). Each
  chunk injection-scanned at write time; high-severity chunks
  quarantined and dropped before fact extraction.
- **Post-write artifact content** — every Fact, Episode, Skill, and
  Chunk carries a SHA-256 `content_hash`. `verify_integrity`
  recomputes and quarantines on mismatch.
- **Cross-tenant content** — impossible by construction. `tenant_id`,
  `owner_type`, `owner_id` are pushed into every vector query's
  `WHERE` clause before the k-nearest cap, and applied to every SQL
  read. See `.claude/skills/vector-contract.md`.

### Treated as trusted

UMA does not defend against actors inside the operator's own trust
boundary. Specifically:

- **The operator running UMA.** UMA is a library the operator embeds in
  their own process. It does not sandbox the developer.
- **The Python process and its filesystem.** Anything the process can
  read or write, UMA assumes is under legitimate operator control.
- **The SQLite database files under `.uma/db/`.** Filesystem
  permissions are the operator's responsibility. UMA does not encrypt
  the database at rest.
- **The vector index files under `.uma/vectors/`.** Rebuildable from
  SQL; treated as an accelerator, not a source of truth.
- **Configuration in `uma.yaml` at startup.** UMA does not validate
  that config values are not maliciously chosen by whoever writes the
  file — treat write access to `uma.yaml` the same as write access to
  the process.
- **LLM and embedding provider endpoints.** Network security (TLS,
  API-key scoping, endpoint authentication) belongs to the operator's
  deployment stack.
- **Caller-registered hooks.** The rate-limit hook, promotion policy,
  and any custom pattern catalog are trusted extensions — UMA calls
  them at every write or retrieval boundary. A hostile hook can refuse
  service or leak scope information to whatever backend it talks to.
  Register only code you control.

### The line between the two

If you can answer "which authenticated user or which document source
produced this bytes-of-content?" — those bytes get scanned. If you
cannot, because the bytes were produced by code you deploy, they are
trusted.

---

## Threat Model

The trust boundary above says *who* is trusted. This section says *what
can go wrong at that boundary and how UMA behaves when it does*.

### Input classes and their trust class

Every entry point UMA exposes accepts one of these input classes. The
trust class is fixed by the entry point, not by the value.

| Input class | Entry point(s) | Trust class | What UMA does |
|---|---|---|---|
| **User turn** (`user_msg`) | `process_turn`, `scan_user_input` | Adversarial | Two-layer scan; high severity raises `InjectionDetectedError` and the turn is dropped entirely. |
| **Assistant reply** (`assistant_reply`) | `process_turn` | Adversarial | Scanned at the episode-write boundary. Trust starts at `0.7` because the assistant may synthesize or hallucinate. |
| **Query text** (`query_text`) | `retrieve_context`, `retrieve_memory` | Adversarial | Scanned; severity propagates to downstream LLM hops which skip amplification on medium/high. |
| **Document content** | `ingest_document` | Adversarial | MIME/size/page-count gates before parsing; HTML/Markdown sanitized; every chunk scanned; high-severity chunks quarantined and dropped before fact extraction. |
| **Memory bootstrap file** (`MEMORY.md`, diary) | `load_memory_bootstrap`, `load_daily_diary_bootstrap` | Adversarial | Same as document content — routed through the ingest pipeline including MIME check, sanitization, and per-chunk scan. Treat these files as untrusted input, not privileged config. |
| **Operator profile file** (`USER.md`, `SOUL.md`) | `load_userprofile`, `load_agentprofile` | **Trusted** | Loaded into an in-memory profile cache without scanning. Do NOT stage user-supplied content here — anything read via these paths is treated as operator-authored. |
| **Tool output** | Any of the above (as `user_msg`, `assistant_reply`, or document) | Adversarial (same as whichever entry point receives it) | UMA has no tool-use primitive of its own; if your agent has one, its output arrives through one of the above entry points and inherits that class. Do not smuggle raw tool output past the scanner by writing directly to the SQLite store. |
| **Configuration** (`uma.yaml`, `custom_patterns_path`) | `UMAMemory.from_yaml` | Trusted | Loaded once at startup; no runtime validation for maliciousness. Write access to the YAML equals write access to the process. |
| **Caller-registered hooks** | `set_rate_limit_hook`, `promotion_policy`, custom pattern catalog | Trusted | Called at every write/retrieval boundary. Register only code you control. |

### Failure modes

The design assumes the primitives in `ARCHITECTURE.md § Security
Architecture` hold. Here is what happens when they don't.

**Scanner bypass.** If a novel injection pattern slips past the write-time
scan (the catalog is best-effort — see `SECURITY.md § Known Unsupported
Threats § Dynamic obfuscation past the pattern catalog`), the content
enters storage with `trust_score` unreduced and `quarantined_at IS NULL`.
It becomes eligible for retrieval and can bias future LLM turns. Downstream
mitigations that still apply:

- `min_trust_score` filtering still drops any artifact whose source classifier assigned a low trust — user-message facts start at 0.9, assistant-reply facts at 0.7, so a bypass that also convinces the classifier is required to reach retrieval with default settings.
- `verify_integrity` can quarantine the artifact after the fact if the content is later flagged (run `lint_memory_drift` on a schedule).
- Cross-tenant isolation still holds: a bypass poisons the writer's own scope, not other tenants'.
- Once identified, the specific pattern gets added to `custom_patterns_path` and all matching quarantined records surface via `list_quarantined`.

**Storage-adapter compromise.** If a vector adapter or SQL store is
subverted (hostile third-party adapter, memory-mapped file corruption,
malicious modification of the SQLite file on disk):

- **SQLite tampering**: `content_hash` on every Fact / Episode / Skill / Chunk lets `verify_integrity` detect post-hoc modification. Rows that fail verification are quarantined and excluded from retrieval. Detection is on-demand — schedule `lint_memory_drift` if you need continuous coverage.
- **Vector-adapter compromise**: the vector store is a rebuildable accelerator; SQL is authoritative. `await memory.rebuild_vector_indexes(tenant_id=...)` regenerates the index from SQL. A misbehaving adapter can degrade retrieval quality but cannot fabricate content that survives cross-check against SQL.
- **Cross-tenant leak via adapter**: the C1 isolation contract refuses empty isolation values at upsert and pushes `tenant_id` / `owner_type` / `owner_id` into the vector query before the k-nearest cap. An adapter that does not honour these — including a hostile third-party backend — is treated as broken. Regression tests in `tests/test_isolation_and_tenancy.py` guard the contract; run them against any custom backend.
- **What UMA cannot detect**: an attacker who can write to the SQLite file can also rewrite the `content_hash` column. Disk-level integrity and encryption are the operator's concern.

**Hostile hook.** A caller-registered rate-limit hook or promotion policy
is trusted code. A hostile hook can refuse service, leak scope information
to whatever backend it talks to, or (for the promotion hook) mis-classify
which artifacts get promoted. Mitigation: register only code you control;
review third-party hook implementations before adoption.

**Configuration tampering.** An attacker with write access to `uma.yaml`
can disable the injection scanner (`security.scan_enabled: false`),
disable quarantine (`security.quarantine_enabled: false`), lower
`min_trust_score`, or point `custom_patterns_path` at a file that
whitelists their own payloads. `uma.yaml` is trusted config — protect
write access to it at the filesystem level.

### Explicitly out of scope

The following are documented in full under `SECURITY.md § Known
Unsupported Threats` and are named here for cross-reference from the
threat model:

- **Temporal decay of trust.** No automatic time-weighting; trust set at write time persists. Layer TTL/decay above UMA if your risk model requires it.
- **Per-tenant rate limiting.** `set_rate_limit_hook` is a caller-owned plug-point, not a limiter. UMA ships no default policy.
- **Malicious operators.** UMA is a library, not a sandbox.
- **Adversarial ML against the embedding model.** No general defense exists in published literature.
- **Prompt injection via LLM output rendering (LLM05).** Caller owns escaping.
- **Autonomous-action risks (LLM06), system-prompt leakage (LLM07), model supply chain (LLM03).** Structurally out of scope for a memory SDK.

---

## Known Unsupported Threats

The following threats are explicitly **not** in scope for UMA as
shipped. Some are structural (UMA cannot defend against them by its
nature as a memory SDK); others are on the roadmap; others belong at
a different layer of the stack.

### Not defended — structural

- **Malicious local operators.** UMA runs in the operator's process. If
  the operator is hostile, the sandbox is not UMA.
- **Adversarial ML attacks on the embedding model.** Perturbations
  crafted to shift an embedding into a semantically unrelated region
  cannot be detected by UMA. No general defense exists in the
  published literature; the pattern catalog covers surface-form
  attacks only.
- **Prompt injection via LLM output rendering.** If the calling
  application renders `retrieve_context` results into a UI without
  escaping, XSS-shape attacks in stored content can reach the end
  user. UMA sanitizes HTML at ingest but does not control caller
  rendering. This is **LLM05** in the OWASP LLM Top 10, and it is out
  of scope by construction.
- **Autonomous-action / tool-use risks (LLM06).** UMA has no agency,
  no tool use, no function calling. It cannot escalate privilege
  because it cannot act. If the calling agent has tool use, those
  risks belong to the agent layer.
- **System-prompt leakage (LLM07).** System prompts live in the
  calling application. UMA never sees them.
- **Model supply chain (LLM03).** UMA has no training pipeline, no
  fine-tuning, no plugin registry. Choice of LLM/embedding provider
  and supply-chain assurance for those artifacts belongs to the
  operator. UMA does harden the *document* ingest boundary against
  executable payloads (MIME check, `PickleParser` removed) — a related
  but separate concern.

### Not defended — currently

- **Temporal decay of trust.** Trust scores are set at write time and
  do not decay automatically. A high-confidence fact from three years
  ago has the same influence at retrieval as one from yesterday.
  Time-weighted ranking or TTL-based expiry can be layered on by the
  caller (or via a future feature) but is not built in. Corresponds
  to Layer 3 of the four-layer defense-in-depth model referenced in
  `README.md § Defense-in-Depth Against Memory Poisoning`.
- **Per-tenant rate limiting.** `set_rate_limit_hook` is a plug-point
  for the caller's own limiter — UMA ships no default limiter, no
  built-in accounting, no fairness algorithm. A heavy tenant cannot
  starve another via the vector index (isolation is pushed into the
  `WHERE` clause before the k-nearest cap), but can still exhaust the
  process's CPU or the LLM provider's quota unless the caller
  registers a hook. See `ARCHITECTURE.md § Rate-Limit Hook`.
- **Cryptographic guarantees against a compromised host.**
  `content_hash` detects post-hoc tampering (via `verify_integrity`
  or `lint_memory_drift`), but does not prevent it. An attacker who
  can write to the SQLite file can also rewrite the hash. Storage
  encryption and disk-level integrity are the operator's concern.
- **Dynamic obfuscation past the pattern catalog.** Pattern-based
  detection is best-effort. Obfuscation via novel encoding, unicode
  homoglyph tricks not yet in the catalog, or model-specific
  jailbreak wording can slip past the injection scanner. Layer a
  model-based classifier or allow-listing above UMA for high-risk
  deployments. When compiled through the `google-re2` backend
  (`pip install uma[security]`), the scanner is at least immune to
  ReDoS in its own execution path.

  *Evaluation scope.* The catalog is currently regression-tested
  against an internal smoke corpus of 42 attack strings and 33
  benign controls (`tests/test_security_injection.py`). Public
  benchmark numbers against adversarial-injection corpora (LOCOMO,
  TensorTrust, HackAPrompt) are in progress and will be published
  with precision / recall / F1 per corpus and UMA version. The
  smoke-corpus pass rate is not a general-purpose accuracy claim.
- **Compliance certification (GDPR, HIPAA, SOC 2, etc.).** UMA
  provides primitives (owner-scoping, audit log, quarantine,
  integrity verification) that support compliance work, but ships no
  compliance package. The operator is responsible for legal review
  of their deployment.

### Not defended — the caller's stack

- **Network-level attacks against LLM/embedding providers.** TLS,
  API-key scoping, endpoint authentication.
- **Side-channel attacks against the SQLite file.** Filesystem
  permissions, disk encryption, backup encryption.
- **Denial-of-service via legitimate high-volume use.** UMA enforces
  isolation and per-file caps; it does not enforce fairness across
  tenants beyond that. Backpressure and quota belong upstream.

---

## Reporting Scope: What Counts

We consider the following in scope for a security report:

- Prompt-injection payloads that reach durable storage without being
  scanned, trust-reduced, or quarantined.
- Cross-tenant leakage — any query or retrieval path that returns rows
  scoped to a different `tenant_id`, `owner_type`, or `owner_id` than
  the caller supplied.
- Bypasses of `min_trust_score` or `quarantined_at IS NULL` filters in
  retrieval queries.
- Integrity failures — modifications to a stored artifact that
  `verify_integrity` fails to detect.
- Ingest-boundary bypasses — MIME/extension mismatches, size-cap
  bypasses, script survival past HTML sanitization.
- SQL injection or code execution paths reachable via any public API.
- Any misuse of the pattern catalog (e.g., a pattern that causes
  catastrophic backtracking on the shipped `re` fallback) —
  even though production is expected to use RE2, the fallback must
  not be a DoS vector.

Out of scope (please still let us know, but expect a "won't fix"
resolution unless we're wrong about the trust model):

- Findings against the LLM provider, embedding provider, or their
  network transport.
- Findings that require write access to `uma.yaml`, the SQLite files,
  or the caller-registered hook functions.
- Findings that require a malicious upstream Python package
  (`google-re2`, `lancedb`, etc.) — that belongs to your dependency
  audit, not to UMA.
- Latency observations without a corresponding DoS argument.

---

## Acknowledgement

We list reporters who ask to be credited in the CHANGELOG entry for the
release that ships the fix. Anonymous reports are equally welcome.

Thank you for helping keep UMA honest.
