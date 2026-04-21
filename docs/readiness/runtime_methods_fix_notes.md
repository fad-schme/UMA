# Runtime Methods Fix Notes

## Real Contract Gaps

The missing methods were real supported-surface gaps, not stale names:

- `UMARuntime.bind(...)`
  - required by runtime-facing tests
  - implied by the bound-context usage already described around `UMARuntime`
  - needed to make request-scoped retrieval explicit without mutating shared runtime state

- `UMAMemory.process_turn(...)`
  - required by ingestion and working-memory tests
  - already had a near-equivalent implementation path through `MemoryPipeline`
  - `sync_memory(...)` existed as a thin wrapper but under the wrong public name for the expected contract

An adjacent runtime-surface gap was also part of the same contract:

- `UMARequestHandle`
  - tests expected an immutable bound handle object
  - `bind(...)` would remain incomplete without it

## Decisions

- `UMARuntime.bind(...)`: restored as a supported public runtime method
- `UMAMemory.process_turn(...)`: restored as the supported public turn-ingest method
- `UMARequestHandle`: restored as the immutable bound runtime handle and exported publicly

## What Changed

1. Added `UMARequestHandle` as a frozen dataclass in `uma/core/runtime/runtime.py`.
   - stores `runtime` and `context`
   - exposes context properties
   - delegates retrieval calls to the runtime or to test-injected bridge methods when present

2. Added `UMARuntime.bind(...)` as a thin constructor for `UMARequestHandle`.

3. Exported `UMARequestHandle` from:
   - `uma/core/runtime/__init__.py`
   - `uma/__init__.py`

4. Added thin compatibility bridges on `UMAMemory`:
   - `_retrieve_structured_context_for_context(...)`
   - `_retrieve_rendered_context_for_context(...)`
   - `_get_context_messages_for_context(...)`
   - `_build_retrieval_request(...)`

5. Restored `UMAMemory.process_turn(...)` as the public wrapper over the canonical `MemoryPipeline.process_turn(...)`.

6. Changed `UMAMemory.sync_memory(...)` into a backward-compatible alias that delegates to `process_turn(...)`.

7. Updated runtime-binding tests to import `UMARequestHandle` explicitly and aligned one stale assertion with the Phase 4 public `runtime.agent_id` property.

8. Adjusted `UMAMemory.agent_id` resolution order so explicit `_agent_id` overrides remain effective when tests intentionally switch agents between turns.

## Why This Solution Is Minimal and Correct

This phase restores only the missing contract pieces already implied by the codebase and tests:

- no runtime architecture redesign
- no duplicate retrieval logic
- no duplicate turn-ingest logic
- no hidden mutable request scope on `UMARuntime`

The restored methods are thin wrappers over the existing canonical runtime and pipeline paths.
