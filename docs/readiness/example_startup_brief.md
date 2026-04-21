# Next Coding-Agent Brief: Fix Example Startup Path And Import Assumptions

## Objective

Fix the documented example startup path so a clean checkout can run the example using the supported package/import surface.

## Why This Is Next

The packaging, public API, identity contract, and missing runtime methods have all been repaired enough for deeper tests to run.

The next release-surface blocker remains the documented startup path:

- the example command in repo docs was previously not runnable from a clean checkout
- import assumptions and startup expectations still need to match the supported package surface

This is a clearer release-surface blocker than the remaining behavioral retrieval failure in the targeted runtime slice.

## Confirmed Prior Symptoms

Previously observed:

- `python examples/chatbot_app/main.py ...` failed with `ModuleNotFoundError: No module named 'uma'`
- module-form execution moved past import resolution but then failed on environment/config dependency expectations

## Focus Scope

Inspect and align:

- `examples/chatbot_app/README.md`
- `examples/chatbot_app/main.py`
- relevant README startup instructions
- package/import assumptions for clean-checkout usage

## Desired Outcome

At the end of that phase:

1. the documented example startup command is correct
2. import assumptions are explicit and consistent
3. failures, if any, are due to real runtime/config dependencies rather than bad invocation or bad import paths

## Guardrails

- Do not broaden into config redesign unless a tiny startup-safe adjustment is unavoidable
- Do not redesign retrieval
- Keep the fix limited to the documented example startup surface

## Notes

The remaining retrieval behavior failure in `test_bound_context_workspace_id_does_not_broaden_retrieval_owner_support` is real, but it is not a runtime-surface gap. The next release-surface remediation step should focus on example startup correctness first.
