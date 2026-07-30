# Fact extraction quality gate

The default test suite keeps `tests/helpers/providers.py::fake_llm` for fast,
hermetic contract tests. This directory separately measures extraction quality
with a real small model served by local Ollama.

Install the opt-in dependency and run:

```bash
pip install -e '.[dev,e2e]'
RUN_E2E=1 OLLAMA_MODEL=qwen2.5:3b \
  python -m pytest tests/e2e/test_fact_extraction_quality.py -q -s
```

The test publishes a `FACT_EXTRACTION_QUALITY` JSON line and enforces micro
precision and recall thresholds over the held-out fixture. Matching requires
the expected predicate and at least 60% recall of the expected object tokens.
Override `E2E_MIN_PRECISION` or `E2E_MIN_RECALL` only for an explicit
calibration run.

## Published baseline

On 2026-07-30, `qwen2.5:3b` produced:

- micro precision: **0.2500**
- micro recall: **0.3333**
- matched / predicted / expected: **2 / 8 / 6**

The enforced defaults (0.20 precision, 0.30 recall) sit just below this measured
baseline so the gate catches regressions without claiming that current
extraction quality is strong. The model copied one example object from the
prompt, used unstable predicate labels for two facts, split one education fact
into two, and incorrectly retained a transient bus-waiting detail.

The same fixture and scoring function are intended to feed the LOCOMO quality
track when that track is available. The default CI suite does not contact model
services.
