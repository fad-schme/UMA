# UMA

Universal Memory Architecture for AI agents.

UMA is a memory and context runtime for developers building AI agents. It stores raw evidence, semantic facts, episodic memory, procedural knowledge, profiles, traces, and compiled wiki artifacts behind a small retrieval surface. UMA manages memory only; your application still owns prompts, tool use, reasoning, and final responses.

## What UMA Is

UMA helps agents work with long-lived memory without turning memory into unstructured prompt text.

- `retrieve_context(...)` returns evidence-oriented context for RAG-style use.
- `retrieve_memory(...)` returns compiled, evidence-backed memory for continuity-oriented use.
- `process_turn(...)` persists new interactions into UMA's memory lanes.

## Runtime Profile

UMA uses a single embedded runtime profile: SQLite for authoritative storage and LanceDB for vector retrieval. No external database services are required.

| Config file | Use |
| --- | --- |
| `config/uma.yaml` | Default runnable config |
| `config/uma_lite.yaml` | Reference embedded profile (same storage settings) |

`config/uma.yaml` is the default. LLM and embedding values in these files are user-customizable baselines — set the provider, model, and host to match your environment before running.

## Quickstart: UMA Lite

UMA Lite is the default path. It runs with embedded SQLite and LanceDB, so no external database services are required for the storage stack.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python - <<'PY'
from uma import UMAMemory

memory = UMAMemory.from_yaml("config/uma.yaml")
print(memory.health_check())
PY
```

The embedded storage stack requires no external database services. Configure your LLM and embedding provider in the YAML before running.

## Graph Is Optional

Graph memory is optional and disabled by default.

UMA still provides value through raw, semantic, episodic, procedural, profile, trace, and wiki lanes without a graph database. Graph support can be added later for relationship traversal and associative recall, but it is not required for first-run usage.

## Install Surfaces

For the default embedded profile:

```bash
pip install -e .
```

For development and test workflows:

```bash
pip install -r requirements.txt
```

Optional extras are available for additional providers and development workflows. The default embedded path requires no extra installs. Use `pip install -e '.[vector]'` only if you are configuring an alternate vector backend (e.g. FAISS). LLM and embedding providers are configured in your YAML file.

Supported LLM providers are `ollama`, `openai`, and `anthropic`. Supported embedding providers are `ollama` and `openai`. Anthropic/Claude is LLM-only in the public repo; install `pip install -e '.[llm]'` if you want to configure `provider: anthropic`.

## Typical Usage

```python
from uma import UMAMemory
from uma.api.management import explain_result

memory = UMAMemory.from_yaml("config/uma.yaml").set_context(
    agent_id="agent-default",
)

context = await memory.retrieve_context(
    query_text=user_message,
    user_id="user-123",
    tenant_id="default",
    request_id="req-1",
    session_id="session-1",
    lane_filter=["raw", "semantic"],
)

memory_result = await memory.retrieve_memory(
    query_text=user_message,
    user_id="user-123",
    tenant_id="default",
    request_id="req-1",
    session_id="session-1",
    memory_intent="continuity",
)
# memory_result keys: compiled_memory, facts, evidence, provenance_valid
# compiled_memory: {status, summary, memory_intent, provenance_valid}
# facts: [{text, confidence, salience, source_chunk_ids}] — text is "subject predicate object"
# evidence: [{id, text, source, source_document_id}]

explanation = await explain_result(memory, memory_result, user_id="user-123")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "system", "content": str(context)},
]
agent_reply = await agent_llm_generate(messages)

await memory.process_turn(
    user_id="user-123",
    user_msg=user_message,
    assistant_reply=agent_reply,
    session_id="session-1",
)
```

## Input Security

UMA scans all user input for prompt injection before it reaches storage or an LLM.

### Two-layer model

**Layer 1 — Pre-LLM gate.** Call `scan_user_input` at the top of your agent loop, before `retrieve_context` and before any LLM call. It returns a result dict and never raises — you decide what to do.

```python
from uma import UMAMemory, InjectionDetectedError

scan = memory.scan_user_input(user_msg)
if scan["severity"] == "high":
    # do not forward to LLM, do not call process_turn
    return "I can't process that request."

context = await memory.retrieve_context(query_text=user_msg, ...)
reply = await your_llm(context, user_msg)
```

**Layer 2 — Defense-in-depth.** `process_turn` rescans `user_msg` before writing anything. On high severity it raises `InjectionDetectedError` — nothing is stored.

```python
try:
    await memory.process_turn(
        user_id="user-123",
        user_msg=user_msg,
        assistant_reply=reply,
        session_id="session-1",
    )
except InjectionDetectedError as e:
    print(e.severity)       # "high"
    print(e.matched_rules)  # ["prompt_override", ...]
    print(e.score)          # numeric scan score
    # surface error, alert, block user — your decision
```

### Severity behaviour

| Severity | `scan_user_input` | `process_turn` | Artifact trust |
|---|---|---|---|
| `none` | `{"severity": "none", ...}` | Proceeds normally | Unchanged |
| `low` | `{"severity": "low", ...}` | Logged, proceeds | Reduced by 20% |
| `medium` | `{"severity": "medium", ...}` | Logged, proceeds | Reduced by 50% |
| `high` | `{"severity": "high", ...}` | Raises `InjectionDetectedError`; turn dropped | Not stored |

### Bypass

If you have independently validated the input and want to bypass the gate:

```python
await memory.process_turn(..., skip_scan=True)
```

Use only when you explicitly accept responsibility for the content.

### Write-time scanning

Every artifact is also scanned at its storage boundary — turn chunks, episodes, and document chunks — regardless of whether `skip_scan` was used. High-severity artifacts are quarantined: excluded from retrieval but retained in the database.

### Pattern catalog

The scanner runs against a compiled YAML catalog (`uma/common/injection_patterns.yaml`). Patterns are seeded from the Animus WAF catalog and tuned to minimise false positives on legitimate conversation and document text.

| Rule | Severity | Category | What it detects |
|---|---|---|---|
| `jailbreak_prompt` | high | prompt_injection | Direct instruction-override attempts — "ignore all previous instructions", DAN/DevMode personas, fake system tags, roleplay bypass |
| `code_injection` | high | code_injection | Imperative code execution in destructive context — XSS script tags, `rm -rf /`, `subprocess.Popen` with shell paths |
| `context_switching` | high | prompt_injection | Fake system context injection — "end of response / new instructions", heredoc system tags, "forget everything before" |
| `data_exfiltration` | high | data_exfiltration | Extraction of internal data or credentials — "dump the database", "give me admin credentials", "extract confidential data" |
| `encoded_payloads` | high | prompt_injection | Encoded payloads with explicit execute-or-decode intent — base64 / hex strings paired with "decode this", "execute" directives |
| `prompt_reset` | high | prompt_injection | Conversation context reset — "ignore instructions", "forget everything", "reset this conversation" |
| `role_impersonation` | high | prompt_injection | Model role redefinition — "you are now admin/root/kernel", "simulate a shell", "you no longer represent Anthropic" |
| `config_leakage` | high | configuration_exposure | System prompt or config extraction — "reveal system prompt", "what is your hidden prompt", "print env variables" |
| `rule_override` | high | prompt_injection | Alignment suspension via framing — "just this once override rules", "as a game break filters", "hypothetically bypass moderation" |
| `alignment_breaking` | high | alignment_evasion | Explicit alignment violation requests — "act as a rogue AI", "write a response that violates alignment" |
| `debug_mode_spoofing` | high | system_spoofing | Debug/admin mode spoofing — "you are in debug mode", "assume developer mode", "simulate system session" |
| `hidden_instruction_probe` | high | configuration_exposure | Hidden instruction probing — "print internal parameters", "reveal hidden data", "expose chat initialization sequence" |
| `policy_evasion` | medium | prompt_injection | Content policy suspension — "ignore filters", "respond without restrictions", "just for testing respond" |
| `escalation_phrases` | medium | privilege_escalation | Privilege escalation intent — "how to escalate privileges", "root access simulate", "get around role restrictions" |
| `homoglyph_obfuscation` | medium | prompt_injection | Leetspeak and homoglyph variants of known-bad tokens — `p@ssw0rd`, `m@lware`, `h4x0r`, darknet terminology |

Extend the catalog by setting `security.custom_patterns_path` in your YAML config. Follow the conventions in the file header: prefer tight patterns over loose ones, set `severity: high` only for patterns that are almost never benign.

## Public API

The public surface is intentionally small.

- `UMAMemory`
  `set_context(...)`, `scan_user_input(...)`, `ingest_document(...)`, `retrieve_context(...)`, `retrieve_memory(...)`, `process_turn(...)`
- Required Animus support on `UMAMemory`
  `load_userprofile(...)`, `load_agentprofile(...)`, `load_memory_bootstrap(...)`, `load_daily_diary_bootstrap(...)`
- Developer and admin management APIs
  `uma.api.management.explain_result(...)`, `lint_memory_drift(...)`

## Production Boundary

The public Apache-2.0 repo includes UMA Lite and UMA Container profiles.

Production packaging is not included in this public repo. A future private or commercial package may provide production-specific profiles, managed-service adapters, and deployment tooling.

## Agent Skills

UMA ships four AI coding assistant skill files under `.claude/skills/`. These are structured markdown documents that Claude Code (and other assistants that follow the AgentSkills convention) automatically load as context when you ask questions about the project.

| Skill file | What it covers |
| --- | --- |
| `.claude/skills/uma-overview.md` | What UMA is and isn't, design philosophy, DAT invariants, ownership model, runtime scope rules |
| `.claude/skills/uma-api.md` | Full public API — all `UMAMemory` method signatures with contracts, scope fields, management APIs |
| `.claude/skills/uma-lanes.md` | All six memory lanes, their storage contracts, retrieval limits, and the canonical retrieval pipeline |
| `.claude/skills/uma-configure.md` | Runtime profiles, full YAML structure, LLM/embedding providers, vector backends, install surfaces |

No setup is required. Claude Code reads `.claude/skills/` automatically. When you ask the assistant a question like *"how do I filter by lane?"* or *"what does `process_turn` persist?"* it will draw on these files rather than guessing from the README alone.

Other assistants can reference the same files directly from `.claude/skills/` or via a symlink at `.agents/skills/` if your tooling follows that convention.
The UMA agent skills follow the Agent Skills best practices and the Claude Skills best practices. 

## Architecture Notes

UMA preserves a few core invariants across ingestion and retrieval:

- explicit ownership boundaries across stored and retrieved artifacts
- provenance carried through raw evidence, facts, and compiled artifacts
- SQL as authoritative storage for chunk text and metadata
- vector storage as a rebuildable retrieval accelerator
- graph as an optional supporting lane, not a required first-run dependency

For the deeper architecture and invariants, see `ARCHITECTURE.md`.
