# UMA-RLM

Universal Memory Architecture for AI agents.

UMA-RLM is a memory and context runtime for developers building AI agents. It stores raw evidence, semantic facts, episodic memory, procedural knowledge, profiles, traces, and compiled wiki artifacts behind a small retrieval surface. UMA manages memory only; your application still owns prompts, tool use, reasoning, and final responses.

## What UMA-RLM Is

UMA-RLM helps agents work with long-lived memory without turning memory into unstructured prompt text.

- `retrieve_context(...)` returns evidence-oriented context for RAG-style use.
- `retrieve_memory(...)` returns compiled, evidence-backed memory for continuity-oriented use.
- `process_turn(...)` persists new interactions into UMA's memory lanes.

The public Apache-2.0 repo now exposes two open-source runtime profiles.

## Runtime Profiles

| Profile | Config file | Runtime model | Best for |
| --- | --- | --- | --- |
| UMA Lite | `config/uma.yaml` | Embedded SQLite + LanceDB | First run, local agents, demos |
| UMA Lite | `config/uma_lite.yaml` | Embedded SQLite + LanceDB | Explicit lite config |
| UMA Container | `config/uma_cont.yaml` | SQLite + Qdrant Docker Compose service | Local service-style development |

`config/uma.yaml` is the default runnable profile and is equivalent to `config/uma_lite.yaml`.
If you install the public vector extras, you can also point `vector_backend` at `uma.adapters.vector.faiss_adapter:FaissIndex` for an in-process FAISS alternative.

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

The embedded storage stack does not require Qdrant, Docker, Postgres, or a graph database. LLM and embedding providers are still configured according to your application needs.

If you want the explicit Lite config instead of the default alias file:

```python
from uma import UMAMemory

memory = UMAMemory.from_yaml("config/uma_lite.yaml")
```

## Optional: UMA Container With Qdrant

UMA Container is the local service-style profile. It keeps SQLite embedded in the UMA process and runs Qdrant as a separate Docker Compose service.

Start Qdrant with:

```bash
pip install -e '.[vector]'
docker compose -f docker/uma_cont/docker-compose.yml up -d
```

`config/uma_cont.yaml` is written for UMA running inside the same Docker Compose network as Qdrant, so it uses:

```text
http://qdrant:6333
```

If UMA runs on the host while Qdrant runs in Docker, override that URL to:

```text
http://localhost:6333
```

UMA Container is not an all-in-one container. Qdrant runs as a separate service, and Docker Compose keeps the local stack easy to start with one command.
Container describes the runtime topology, not a fixed vector engine. The default container profile uses Qdrant, and you can also configure LanceDB or the packaged FAISS adapter if you want the UMA process to keep vectors in-process.

## Configuration Files

- `config/uma.yaml`
  Default runnable config. Equivalent to UMA Lite.
- `config/uma_lite.yaml`
  Explicit embedded profile using SQLite + LanceDB.
- `config/uma_cont.yaml`
  Container-backed local profile using SQLite + Qdrant.

The `profile` field is declarative metadata. UMA still initializes through the same path:

```python
from uma import UMAMemory

memory = UMAMemory.from_yaml("config/uma.yaml")
```

There are no separate `init_lite()` or `init_cont()` entry points.

## Graph Is Optional

Graph memory is optional. The public UMA Lite and UMA Container profiles disable graph by default.

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

Optional extras remain available for additional providers and development workflows. In particular, `pip install -e '.[vector]'` adds the public Qdrant and FAISS client dependencies for alternate vector backends, while the default UMA Lite path does not require users to install separate vector or graph infrastructure before trying UMA.

Supported LLM providers are `ollama`, `openai`, and `anthropic`. Supported embedding providers are `ollama` and `openai`. Anthropic/Claude is LLM-only in the public repo; install `pip install -e '.[llm]'` if you want to configure `provider: anthropic`.

## Typical Usage

```python
from uma import UMAMemory
from uma.api.management import explain_result, export_wiki_projection

memory = UMAMemory.from_yaml("config/uma.yaml").set_context(
    user_id="user-123",
    agent_id="agent-default",
    tenant_id="default",
    request_id="req-1",
    session_id="session-1",
)

context = await memory.retrieve_context(
    query_text=user_message,
    lane_filter=["raw", "semantic"],
)

memory_result = await memory.retrieve_memory(
    query_text=user_message,
    memory_intent="continuity",
)

explanation = await explain_result(memory, memory_result)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "system", "content": str(context)},
]
agent_reply = await agent_llm_generate(messages)

await memory.process_turn(
    user_id="user-123",
    user_msg=user_message,
    assistant_reply=agent_reply,
    extra_meta={"session_id": "session-1"},
)
```

## Public API

The public surface is intentionally small.

- `UMAMemory`
  `set_context(...)`, `ingest_document(...)`, `retrieve_context(...)`, `retrieve_memory(...)`, `process_turn(...)`
- Required Animus support on `UMAMemory`
  `load_userprofile(...)`, `load_agentprofile(...)`, `load_memory_bootstrap(...)`, `load_daily_diary_bootstrap(...)`
- Developer and admin management APIs
  `uma.api.management.explain_result(...)`, `update_wiki_page(...)`, `export_wiki_projection(...)`, `lint_memory_drift(...)`

## Production Boundary

The public Apache-2.0 repo includes UMA Lite and UMA Container profiles.

Production packaging is not included in this public repo. A future private or commercial package may provide production-specific profiles, managed-service adapters, and deployment tooling.

## Architecture Notes

UMA-RLM preserves a few core invariants across ingestion and retrieval:

- explicit ownership boundaries across stored and retrieved artifacts
- provenance carried through raw evidence, facts, and compiled artifacts
- SQL as authoritative storage for chunk text and metadata
- vector storage as a rebuildable retrieval accelerator
- graph as an optional supporting lane, not a required first-run dependency

For the deeper architecture and invariants, see `ARCHITECTURE.md`.
