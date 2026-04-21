# Packaging Fix Notes

## Root Cause

The broken artifact path was caused by a build-tool compatibility gap, not by the `uma/` source tree itself.

- The repo declares modern PEP 621 metadata in `pyproject.toml`.
- In this environment, isolated builds could not fetch `setuptools>=64` because network access is restricted.
- When build isolation was disabled, the build fell back to host `setuptools 58.0.4`.
- That legacy setuptools path did not correctly interpret the `pyproject.toml` project metadata and package discovery config, which caused:
  - wheel identity to collapse to `UNKNOWN-0.0.0`
  - an empty artifact containing only `.dist-info`

## What Was Changed

1. Added a minimal `setup.py` compatibility shim.

The shim explicitly defines:

- package name: `uma`
- version loaded from `uma/version.py`
- package discovery with `find_packages(include=["uma", "uma.*"])`
- core metadata and dependencies matching the existing `pyproject.toml`

## Why This Fix Works

The new `setup.py` gives legacy setuptools a concrete packaging definition to use when the intended isolated `pyproject.toml` build path is unavailable.

That fixes the exact failure mode seen in Phase 1:

- the wheel is now built as `uma-0.1.5-py3-none-any.whl`
- the wheel contains `uma/...` package files
- the wheel installs into a fresh virtual environment
- `import uma` works from outside the repo root

This is intentionally narrow:

- no runtime behavior was changed
- no repo layout was redesigned
- no unrelated API cleanup was done

## Validation Limitations

- Isolated `pyproject.toml` builds are still limited by this environment’s restricted network access if they need to fetch build requirements.
- Packaging integrity itself was still validated successfully through:
  - `pip wheel . --no-deps --no-build-isolation`
  - `python3 setup.py bdist_wheel --dist-dir /tmp/uma_phase2_dist`
  - wheel content inspection
  - clean venv install
  - clean `import uma` outside the repo root

## Phase Scope

This phase fixed packaging/build integrity only.

It did not address:

- public API/export drift
- documented startup flow
- config safety
- runtime initialization behavior
- `agent_id` contract consistency
- test failures unrelated to packaging
