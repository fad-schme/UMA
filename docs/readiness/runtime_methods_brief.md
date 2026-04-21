# Next Coding-Agent Brief: Fix Missing Runtime Methods

## Objective

Fix the missing runtime methods now surfaced after the `agent_id` contract repair.

Primary blockers:

- `UMARuntime.bind(...)`
- `UMAMemory.process_turn(...)`

Only implement these if they are part of the intended supported contract already implied by tests, docs, and surrounding runtime code.

## Why This Is Next

After the import-surface and `agent_id` fixes, the next focused failures are no longer identity-related. The remaining runtime-facing tests fail because core methods expected by the runtime contract are missing.

## Confirmed Current Symptoms

Focused runtime test failures now center on:

- `AttributeError: 'UMARuntime' object has no attribute 'bind'`
- broader suite failures also include:
  - `AttributeError: 'UMAMemory' object has no attribute 'process_turn'`
  - `AttributeError: 'UMAMemory' object has no attribute '_build_retrieval_request'`

## Focus Scope

Inspect at minimum:

- `uma/core/runtime/runtime.py`
- `uma/core/uma_memory.py`
- tests:
  - `tests/test_runtime_binding.py`
  - `tests/test_bound_context_retrieval.py`
  - `tests/test_retrieval_service_recall_scope.py`
  - `tests/test_turn_session_local_defaults.py`
  - `tests/test_working_memory_session_scope.py`
  - `tests/test_pipeline_graph.py`
  - `tests/test_turn_ingest_idempotent.py`

## Desired Outcome

At the end of that phase:

1. `UMARuntime.bind(...)` exists if it is part of the supported runtime contract
2. `UMAMemory.process_turn(...)` exists if it is the intended public ingestion entry point
3. tests move beyond missing-method failures into real behavioral assertions
4. no broad runtime redesign is introduced

## Guardrails

- Do not broaden into config cleanup
- Do not redesign retrieval behavior
- Do not weaken tests
- Keep the implementation minimal and contract-driven

## Notes

The current source already suggests these methods are intended:

- tests call them broadly
- docs/examples assume runtime-bound usage
- `UMAMemory.sync_memory(...)` and runtime retrieval methods indicate the surrounding execution path already exists in pieces

The task is to restore the intended supported contract cleanly, not to invent a new model.
