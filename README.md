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

## Public API

The public surface is intentionally small.

- `UMAMemory`
  `set_context(...)`, `ingest_document(...)`, `retrieve_context(...)`, `retrieve_memory(...)`, `process_turn(...)`
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
