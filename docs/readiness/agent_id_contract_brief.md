# Next Coding-Agent Brief: Fix `agent_id` Contract Consistency

## Objective

Fix UMA's `agent_id` contract so the runtime identity surface is explicit, consistent, and no longer split between:

- internal `_agent_id` mutation
- missing `agent_id` public/property access
- runtime-context-driven request scope

This is the next blocker because full pytest now clears import collection and the dominant remaining failures are `agent_id`-related.

## Confirmed Current Symptoms

From the current full test run:

- many tests fail with `AttributeError: 'UMAMemory' object has no attribute 'agent_id'`
- helper/test scaffolding still writes `memory._agent_id = ...`
- `UMAMemory._build_runtime_context()` references `self.agent_id`
- public-surface tests expect `agent_id` setter behavior to be constrained

## Focus Scope

Inspect and align:

- `uma/core/uma_memory.py`
- any runtime identity helpers / scope types
- tests/helpers/runtime.py`
- failing tests that currently reference `memory.agent_id`
- any code that still relies on `_agent_id` as ambient mutable state

## Desired Outcome

At the end of that phase:

1. `UMAMemory` has a clear supported read contract for agent identity, or tests/docs are updated if it should not
2. `_agent_id` and `agent_id` usage are no longer contradictory
3. tests stop failing due to missing `memory.agent_id`
4. the chosen behavior remains consistent with the explicit runtime-context design
5. no unrelated startup/config/retrieval redesign is introduced

## Guardrails

- Do not broaden into config cleanup
- Do not redesign retrieval
- Do not fix `bind` / `process_turn` unless directly required by the `agent_id` contract
- Keep the change minimal and contract-driven

## Immediate Evidence to Start From

Key failing areas from the latest full test run:

- `tests/test_bound_context_retrieval.py`
- `tests/test_environment_api.py`
- `tests/test_isolation_matrix.py`
- `tests/test_legacy_turn_compatibility.py`
- `tests/test_promotion_v2.py`
- `tests/test_retrieval_service_recall_scope.py`
- `tests/test_turn_session_local_defaults.py`
- `tests/test_working_memory_session_scope.py`

## Notes

The import-surface blocker is already resolved in Phase 3.

Do not reopen packaging or public API export work unless the `agent_id` contract fix directly requires a tiny follow-on adjustment.
