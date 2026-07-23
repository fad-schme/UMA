---
name: uma-agent-loop
description: End-to-end developer pattern for integrating UMA into an agent loop — the canonical sequence (scan → retrieve → LLM → process_turn), how to wire it with Anthropic / OpenAI / Ollama providers, how to handle InjectionDetectedError and rate-limit refusals, how to use lane_filter to scope context, how to use session_id correctly, and how to register custom rate-limit hooks. Use this skill when answering questions about how to actually USE UMA in a chatbot, what the correct call order is, how to handle high-severity injection, what to pass as session_id versus request_id, how to integrate UMA with an LLM provider, or any "show me a working example" question.
---

# UMA — Agent Loop Integration

UMA does not generate replies. It manages memory and returns context for your LLM. The canonical pattern is:

```
1. scan_user_input         (pre-LLM gate, advisory)
2. retrieve_context         (curated evidence for the prompt)
3. Your LLM call            (you own this — Anthropic, OpenAI, Ollama, whatever)
4. process_turn             (persist user_msg + assistant_reply)
```

Steps 1, 2, and 4 are UMA. Step 3 is yours.

---

## Minimal Working Loop

```python
from uma import UMAMemory, InjectionDetectedError
import anthropic  # or openai, ollama, etc.

memory = UMAMemory.from_yaml("/path/to/your/uma.yaml").set_context(agent_id="my-agent")
llm = anthropic.Anthropic()

TENANT = "acme-corp"
USER = "alice"
SESSION = "session-abc"

async def handle_turn(user_msg: str) -> str:
    # 1. Pre-LLM gate (advisory, does not raise)
    scan = memory.scan_user_input(user_msg)
    if scan["severity"] == "high":
        return "I can't process that request."

    # 2. Retrieve evidence-oriented context
    context = await memory.retrieve_context(
        query_text=user_msg,
        user_id=USER,
        tenant_id=TENANT,
        session_id=SESSION,
    )

    # 3. Call your LLM — UMA does not do this for you
    system = build_system_prompt(context)   # your code
    response = llm.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    reply = response.content[0].text

    # 4. Persist the turn
    try:
        await memory.process_turn(
            user_id=USER,
            user_msg=user_msg,
            assistant_reply=reply,
            session_id=SESSION,
            tenant_id=TENANT,
        )
    except InjectionDetectedError:
        # Layer-2 boundary scan caught a high-severity payload that the
        # advisory Layer-1 scan missed (or that the caller skipped).
        # The turn is NOT persisted. Surface the event but the user
        # already has the reply — your call whether to invalidate it.
        log_security_event()

    return reply
```

That's the whole loop. The patterns below extend it.

---

## Why scan at Layer 1 even though Layer 2 also scans

UMA defends against memory poisoning (ASI06) using a defense-in-depth model. The two injection-scanning layers are the Pre-Write Sanitization layer of that model:

- **Layer 1 (`scan_user_input`)** protects the **LLM** — payload never reaches the model if you act on the result. Cheap, synchronous, never raises.
- **Layer 2 (`process_turn`)** protects **storage** — even if Layer 1 was skipped or a payload slipped through, nothing flagged reaches the durable layer untagged.

You can skip Layer 1 and rely entirely on Layer 2. The LLM will see the payload, but nothing toxic will be stored. The Layer-1 step exists for the case where you want to refuse the request entirely before paying for an LLM call.

---

## Scope Discipline

The scope fields are not interchangeable. Treat them as a hierarchy:

| Field | Lifetime | What changes it |
|---|---|---|
| `tenant_id` | Forever | Customer / org boundary; rarely changes |
| `user_id` | Forever within tenant | One person |
| `agent_id` | Forever within instance | Bound once via `set_context()` |
| `session_id` | Conversation thread | New chat, new tab, new continuation |
| `request_id` | Single API call | Auto-generated if you don't pass one |
| `workspace_id` | Project / channel | Optional secondary scope |

**Common mistake:** Treating `session_id` as a request id. Don't — a session covers many turns. If you give every turn a new `session_id`, you fragment working memory and episodic recall.

**Correct shape:**

```python
# At conversation start
session_id = f"sess-{uuid4()}"

# Every turn in the conversation reuses it
await memory.process_turn(session_id=session_id, ...)
context = await memory.retrieve_context(session_id=session_id, ...)
```

---

## Lane Filtering

By default, `retrieve_context` queries every available lane and merges the result. To narrow:

```python
# RAG over ingested documents only — useful for "answer from the docs" mode
context = await memory.retrieve_context(
    query_text=user_msg,
    user_id=USER,
    tenant_id=TENANT,
    lane_filter=["raw", "semantic"],
)

# Continuity / recall — what did we discuss before
context = await memory.retrieve_context(
    query_text=user_msg,
    user_id=USER,
    tenant_id=TENANT,
    session_id=SESSION,
    lane_filter=["working_memory", "episodic"],
)

# Procedural recall — "how do I do X"
context = await memory.retrieve_context(
    query_text=user_msg,
    user_id=USER,
    tenant_id=TENANT,
    lane_filter=["procedural"],
)
```

Valid lane names: `raw`, `semantic`, `episodic`, `procedural`, `wiki`, `working_memory`. See `uma-lanes.md`.

---

## Reading the Context Pack

`retrieve_context` returns a dict. The shape:

```python
{
    "snippets": [...],          # ranked, refined text snippets
    "facts": [...],             # extracted facts triplets (subject predicate object)
    "working_memory": [...],    # recent session turns
    "meta": {...},              # provenance, scope info
    "query_scan_severity": "none",  # severity flag for the query itself
}
```

How you assemble the system prompt is up to you. A simple pattern:

```python
def build_system_prompt(context: dict) -> str:
    parts = ["You are a helpful assistant."]

    if context.get("snippets"):
        parts.append("Relevant context:\n" + "\n\n".join(
            s["text"] for s in context["snippets"]
        ))

    if context.get("facts"):
        parts.append("Known facts:\n" + "\n".join(
            f["text"] for f in context["facts"]
        ))

    if context.get("working_memory"):
        parts.append("Recent conversation:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in context["working_memory"]
        ))

    return "\n\n".join(parts)
```

If `query_scan_severity` is `medium` or `high`, downstream LLM hops (snippet refiner, fact pruner) have already skipped their refinement step — you got raw chunks instead of LLM-polished snippets. That's deliberate; the refiner won't amplify a hostile query.

---

## Handling `InjectionDetectedError`

`process_turn` raises this only on high-severity hits in `user_msg`. The exception carries:

- `severity` — always `"high"` when this is raised
- `matched_rules` — list of rule names that fired (e.g. `["prompt_reset", "role_impersonation"]`)
- `score` — numeric scan score

```python
try:
    await memory.process_turn(...)
except InjectionDetectedError as e:
    # Already replied — but nothing was stored
    log_security_event({
        "user_id": USER,
        "tenant_id": TENANT,
        "severity": e.severity,
        "rules": e.matched_rules,
        "score": e.score,
    })
    # Optionally invalidate the reply or warn the user.
```

If your Layer-1 gate caught the same content, Layer 2 never sees it — `process_turn` only raises when something slipped past Layer 1 (or you used `skip_scan=True`).

---

## Rate-Limit Hook

Register a single optional hook that fires at the top of every expensive UMA operation:

```python
import time
from collections import defaultdict

call_log = defaultdict(list)

def hook(operation, ctx):
    """Per-tenant per-operation 10-calls-per-60-seconds throttle."""
    tenant = ctx.tenant_id if ctx else "global"
    key = (tenant, operation)
    now = time.time()
    call_log[key] = [t for t in call_log[key] if t > now - 60]
    if len(call_log[key]) >= 10:
        raise RuntimeError(f"rate limit: {tenant}/{operation}")
    call_log[key].append(now)

memory.set_rate_limit_hook(hook)
```

The hook raises to refuse. Async hooks work too:

```python
async def async_hook(operation, ctx):
    if await redis.incr(f"rl:{ctx.tenant_id}:{operation}") > 10:
        raise RateLimitExceeded()

memory.set_rate_limit_hook(async_hook)
```

`ctx` is a `RuntimeContext` for `retrieve_context`, `retrieve_memory`, `process_turn`. For `ingest_document`, `ctx=None` (the API takes `owner_type`/`owner_id`, not a user scope) — the hook can still throttle by operation name.

UMA does **not** ship a default rate limiter. Operators wire whatever they already use.

---

## Document Ingestion

For RAG over user-uploaded documents:

```python
report = await memory.ingest_document(
    file_path="/uploads/policy.pdf",
    owner_type="user",
    owner_id=USER,
    tenant_id=TENANT,
)
```

The pipeline does MIME check → size limit → page count limit → parse → chunk → sanitize → scan-per-chunk → embed → persist (SQL first, then vector, then manifest). If embedding fails, no manifest is written; re-ingest is safe.

Failures you should handle:

- `FileNotFoundError` — path doesn't exist
- `MimeRejection` — wrong type / extension mismatch
- `FileSizeRejection` — over `max_file_bytes`
- `ValueError` — bad arguments (empty owner, etc.)

After ingest, retrieval automatically includes the new chunks (no separate refresh step).

---

## Multi-Tenant Pattern

Every UMA call takes `tenant_id` explicitly. Pass the right one for each request:

```python
async def serve_request(http_request):
    tenant = authenticate(http_request)   # your code
    user = http_request.user_id
    session = http_request.session_cookie

    memory = get_shared_memory_instance()  # one UMAMemory per process is fine

    context = await memory.retrieve_context(
        query_text=http_request.body,
        tenant_id=tenant,
        user_id=user,
        session_id=session,
    )
    # tenant A's request CANNOT see tenant B's data — enforced at storage layer
```

Cross-tenant isolation is enforced by the vector index (LanceDB pushes tenant into the `WHERE` clause) and by every SQL query — not by application-layer logic. You cannot accidentally leak across tenants by forgetting a filter.

---

## Promotion Pattern (Optional)

Facts extracted from a turn are session-local by default. To make them durable across sessions:

```python
# After process_turn, the facts are extracted but session-scoped
# To promote — set up promotion_policy on the memory instance
memory.promotion_policy = MyPromotionPolicy(...)
```

Promotion is opt-in. The default behavior is session-local; this matches "the user said something just for this conversation" semantics.

---

## Performance Notes

- `from_yaml` returns when retrieval is ready; ingestion warms up in the background. Calling `ingest_document` immediately after `from_yaml` is safe — it waits internally.
- `process_turn` is async but does meaningful work (LLM fact extraction, embedding). Don't await it on the critical reply path if latency matters; defer with `pipeline.defer_post_turn: true` in your YAML.
- `retrieve_context` typically runs in 50-300 ms depending on lane filter and corpus size on the Lite profile. The slow path is LLM-driven snippet refinement; disable it (`snippet_refiner_enabled: false`) if you don't need it.
- The retrieval audit table grows ~1 row per retrieve call. Prune or rotate periodically if you serve a high volume.

---

## Pattern Cheat Sheet

| Goal | Pattern |
|---|---|
| Chatbot with memory | scan → retrieve_context → LLM → process_turn |
| RAG over uploaded docs | ingest_document, then retrieve_context with `lane_filter=["raw","semantic"]` |
| Continuity (recall earlier conversation) | retrieve_context with `lane_filter=["working_memory","episodic"]` |
| Refuse hostile input | scan_user_input, return on `severity == "high"` |
| Throttle per tenant | set_rate_limit_hook with a counter keyed by `ctx.tenant_id` |
| Multi-tenant SaaS | pass `tenant_id` from auth context on every call |
| Debug what UMA returned | `from uma.api.management import explain_result` |
| Inspect quarantined records | `list_quarantined` from management API |
| Audit retrieval history | `list_retrieval_audit` from management API |
