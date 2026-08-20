# Opt-in quality gates

Two measurements live here, both requiring local Ollama and both off by
default. `test_fact_extraction_quality.py` asks how much the extractor gets
out of a passage. `test_retrieval_quality.py` asks whether `retrieve_context`
surfaces the right source pages for a question. They gate different stages and
their numbers are not comparable.

---

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

---

# Retrieval quality gate

`test_retrieval_quality.py` ingests a hand-authored corpus, runs every gold
query through the public `retrieve_context` surface, and scores which source
pages come back.

```bash
pip install -e '.[dev,e2e]'
RUN_E2E=1 OLLAMA_MODEL=qwen2.5:3b OLLAMA_EMBED_MODEL=nomic-embed-text \
  python -m pytest tests/e2e/test_retrieval_quality.py -q -s
```

The run takes roughly 90 seconds and publishes a `RETRIEVAL_QUALITY` JSON line
with per-query detail. Override `E2E_MIN_RETRIEVAL_RECALL` or
`E2E_MIN_RETRIEVAL_R_PRECISION` only for an explicit calibration run.

## Corpus and gold

`fixtures/retrieval_corpus/` holds eight hand-authored pages describing a
fictional transit network. Pages deliberately share vocabulary — depots appear
in four of them, the night network in three — so that lexical overlap alone
does not identify the right page.

`fixtures/retrieval_gold.json` holds 17 queries. Each names the corpus
**filenames** that actually contain the answer, curated by hand and never
derived from a retrieval run. Page-level gold is deliberate: chunk ids are
generated at ingest and cannot be hand-authored, so gold keyed on chunk ids
would have to be regenerated from a run and would stop being independent
evidence. The gold file is read only after ingest has completed, so it cannot
reach the ingest path.

## Metrics

- **R-precision** — precision over the first *n* retrieved pages, where *n* is
  the size of that query's gold set. This is the primary gate.
- **Recall@3** — did the gold pages appear in the top three distinct pages.

Fixed-cutoff precision (P@5 and friends) is deliberately **not** reported. With
eight pages and gold sets of one or two, P@5 is capped at 0.2–0.4 by gold-set
size rather than by ranking quality, so it cannot move when ranking changes.
R-precision is rank-sensitive and normalised for gold-set size.

## Published baseline

On 2026-08-11, `qwen2.5:3b` + `nomic-embed-text` over 8 pages and 17 queries
produced:

- r-precision: **0.7353**
- recall@3: **0.9118** (0.9412 on a repeat run)

The enforced defaults (0.65 r-precision, 0.82 recall) sit below the lower
observed value so the gate catches regressions without overfitting to run
variance.

**Read these numbers narrowly.** Recall is near-saturated and has little
headroom: with eight topically distinct pages the right page is almost always
somewhere in the top three, and recall stays above 0.85 even at a cutoff of
two. It is a coverage regression guard — it will catch a catastrophic break,
not demonstrate that retrieval is strong. R-precision is the number with real
spread; per-query values across the gold set range from 0.0 to 1.0. Four
queries currently rank a plausible-but-wrong page first: asking how often
vehicles rotate *between depots* returns `depot-locations.md` above
`fleet-maintenance.md`, which is where the rotation interval actually is.

Running the same corpus through the hermetic `fake_llm` / `fake_embed`
providers yields the same r-precision (0.7353), which indicates page selection
on this corpus is driven by the lexical side of the hybrid rather than by
embedding quality. Do not read this gate as a measurement of the embedding
model.

The default CI suite does not contact model services.
