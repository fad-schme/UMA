# Public API Fix Notes

## Mismatch

The package root and the rest of the repo were out of sync:

- `README.md` documented `from uma import UMARuntime`
- several tests imported `UMARuntime` from `uma`
- `uma/__init__.py` exported only `UMAMemory`

That mismatch caused full `pytest` collection to stop at the import layer before deeper runtime failures could be evaluated.

## Chosen Public Contract

Canonical public runtime surface:

- `UMAMemory`
- `UMARuntime`

Decision rationale:

- `UMARuntime` already exists as a real runtime class in source
- the README and tests already refer to it
- supporting it at the package root is the least disruptive and most consistent option

`UMAMemory` remains public.

This phase did not add or promise a public `UMARequestHandle` symbol because the current source tree does not define that class explicitly.

## What Changed

1. Updated `uma/__init__.py` to export both `UMAMemory` and `UMARuntime`.
2. Updated the README import example to use `from uma import UMAMemory, UMARuntime`.
3. Updated README prose to avoid naming `UMARequestHandle` directly in user-facing usage guidance.
4. Updated the example app to import `UMAMemory` from the package root instead of an internal module path.
5. Updated stale test imports so tests that use `UMARuntime` import it explicitly from `uma`.

## Why This Is the Correct Supported Surface

This is the smallest change that makes the public package contract coherent:

- package exports now match documented usage
- tests no longer fail at collection because of missing `UMARuntime`
- examples use supported package-level imports rather than internal module paths

This phase intentionally did not fix deeper runtime behavior issues such as:

- missing `agent_id` contract
- missing `UMARuntime.bind(...)`
- missing `UMAMemory.process_turn(...)`

Those are now visible because the import-surface blocker has been removed.
