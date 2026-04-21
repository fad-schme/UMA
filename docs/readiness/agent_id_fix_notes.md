# Agent ID Fix Notes

## Identity Contract Drift

The codebase had a narrow but critical mismatch:

- `UMAMemory` stored `_agent_id`
- multiple runtime paths already referenced `self.agent_id`
- no public `agent_id` property existed
- tests/helpers still seeded `_agent_id` directly because the public contract was missing

That caused many retrieval, environment, and isolation tests to fail before their real assertions with:

- `AttributeError: 'UMAMemory' object has no attribute 'agent_id'`

## Chosen Contract

Supported runtime identity surface:

- `UMAMemory.agent_id`
  - public
  - read-only
  - returns the current runtime agent identity if known
  - returns `None` if identity has not been established yet

- `UMARuntime.agent_id`
  - public
  - read-only
  - reflects the bridged `UMAMemory.agent_id` when a memory bridge exists
  - returns `None` when no bridge identity is available

Identity remains runtime/request state, not durable configuration.

## What Changed

1. Added a read-only `agent_id` property to `UMAMemory`.
   - It resolves identity from the bound runtime context first.
   - It falls back to the internal `_agent_id` storage when present.

2. Added a read-only `agent_id` property to `UMARuntime`.
   - It exposes the bridged runtime identity without introducing ambient mutable request state on the runtime itself.

3. Updated the shared test helper in `tests/helpers/runtime.py` to seed agent identity through `set_context(...)` instead of mutating `_agent_id` directly.

4. Updated the procedural-surface test bootstrap in `tests/test_public_scope_surfaces.py` to use the same public setup path.

## Why This Fix Is Correct and Minimal

This repair does not redesign the runtime model.

It only makes the already-assumed identity contract explicit and readable:

- code using `self.agent_id` now works
- the public surface is read-only, so accidental mutation remains blocked
- missing identity is still explicit via `None`
- request-scoped boundaries still decide when identity is mandatory

This phase intentionally did not implement unrelated missing runtime methods such as:

- `UMARuntime.bind(...)`
- `UMAMemory.process_turn(...)`

Those are now the next real blockers after the identity layer is repaired.
