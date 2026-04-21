# UMA Baseline Blockers

## Critical

### 1. Invalid release artifact

- Impact: The package cannot be trusted as deployable because the built wheel is `UNKNOWN-0.0.0` and does not contain the `uma` package.
- Likely root cause: Packaging metadata and/or setuptools discovery are misconfigured or not resolving correctly in the actual build path.

### 2. Public API export mismatch

- Impact: Full test collection fails and the documented import path is broken because `UMARuntime` is expected publicly but is not exported from `uma`.
- Likely root cause: Partial migration of the public surface without synchronized updates across `uma/__init__.py`, README, examples, and tests.

### 3. Documented startup path is broken

- Impact: A clean-checkout user cannot start the example app with the command currently documented in the repo.
- Likely root cause: The docs assume installed-package semantics or `PYTHONPATH` setup, but the command shown uses neither.

## High

### 4. Default config contains secrets and machine-specific infrastructure

- Impact: The repo baseline is unsafe to publish and not reproducible across environments.
- Likely root cause: A developer-local operational config was committed as the default shared config.

### 5. Eager initialization couples baseline startup to optional infrastructure

- Impact: `UMAMemory.from_yaml()` fails before the runtime is usable when optional backends or dependencies are absent.
- Likely root cause: `from_yaml()` eagerly performs retrieval-ready initialization against the configured vector, graph, LLM, and embedder stack.

### 6. `agent_id` contract drift

- Impact: Retrieval, environment, chunk, semantic, and isolation tests fail with missing `agent_id`, blocking confidence in runtime identity and scoping behavior.
- Likely root cause: Incomplete migration between internal `_agent_id` mutation and an intended explicit runtime-context contract.

### 7. Critical-path tests are partially blocked by API drift

- Impact: Important tenant-boundary and retrieval-path tests cannot currently serve as release evidence because they fail before core assertions.
- Likely root cause: Public API drift and identity-contract inconsistencies are masking deeper behavior validation.

## Medium

### 8. Dependency declaration surfaces are inconsistent

- Impact: Install guidance and runtime requirements are not cleanly represented for developers or CI.
- Likely root cause: `pyproject.toml` and `requirements.txt` have diverged in purpose and contents.

### 9. Test helpers still rely on internals

- Impact: Some tests validate useful behavior, but confidence is reduced because helpers directly mutate `_agent_id` and use fake providers for major paths.
- Likely root cause: Test infrastructure has not fully caught up with the intended public/runtime contract.

### 10. Brittle source-text assertion in ownership test

- Impact: At least one failure is caused by substring matching rather than behavioral validation, which adds noise to the baseline.
- Likely root cause: `tests/test_ownership_only_retrieval.py` uses text-level forbidden-string checks that overmatch.

## Recommended Remediation Order

1. Packaging/build artifact validity
2. Public API/export surface alignment
3. `agent_id` contract consistency
4. Example startup path correctness
5. Safe default config
6. Re-run critical-path tests after the above
7. Tighten test credibility issues that remain

No fixes were implemented in Phase 1.
