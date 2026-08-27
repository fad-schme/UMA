# UMA examples

Four runnable examples, ordered by how much they ask of you. Every one takes
`--config path/to/uma.yaml` — the SDK has no default location it reads, so you
always pass the path explicitly. [`../config/uma.yaml`](../config/uma.yaml) is
a reference config in this repo for the examples to copy and edit; it is not
a config UMA loads automatically.

## Before you start

```bash
pip install -e '.[parsers]'      # document ingestion
pip install -e '.[ollama]'       # or '.[openai]' — the examples need an LLM client
```

Confirm the runtime works before running anything else. This checks the stores,
the vector indexes, and that both providers are actually reachable:

```bash
uma --config path/to/uma.yaml health
```

Exit code 0 means you are ready. If a provider is unreachable the output names
it, the host, and why.

> **First call is slow.** A local model provider loads the model into memory on
> first use, so the first request after boot is much slower than later ones.
> `uma health` warms both models, so running it first makes the examples feel
> instant.

## Where UMA stops

UMA manages memory. It does not generate replies, and it does not own your
prompts. Each example that needs a model therefore builds **its own** LLM client
from the `llms.uma` block of your config. That is deliberate: reading the same
config file is a convenience, not a UMA API. The retrieve → prompt → generate →
`process_turn` loop in these files is the integration pattern.

## The examples

| Example | What it shows | Needs an LLM |
| --- | --- | --- |
| [`memory_app/main.py`](memory_app/main.py) | Interactive `retrieve_memory` loop — compiled memory, facts, evidence, provenance. The smallest useful starting point. | no |
| [`batch_test.py`](batch_test.py) | Runs a fixed question set through `retrieve_context` and prints which lanes answered. Good for eyeballing recall after an ingest. | no |
| [`chatbot_app/main.py`](chatbot_app/main.py) | Full interactive chatbot: document ingest, retrieval, generation, `process_turn`. See [`chatbot_app/README.md`](chatbot_app/README.md). | yes |
| [`chatbot_app/sim.py`](chatbot_app/sim.py) · [`github_chat_eval.py`](github_chat_eval.py) | Scripted conversations that exercise several lanes at once. `sim.py` spans document + profile + diary; `github_chat_eval.py` focuses on one ingested PDF and reports retrieval hit rates. | yes |

## A first run

```bash
# 1. Seed the store from the markdown fixtures in this directory
python examples/memory_app/main.py --config path/to/uma.yaml --load

# 2. Ask it things
#    (the same command without --load skips re-seeding)
python examples/memory_app/main.py --config path/to/uma.yaml

# 3. See what retrieval finds across the whole question set
python examples/batch_test.py --config path/to/uma.yaml
```

The fixtures loaded by `--load` — [`MEMORY.md`](MEMORY.md) and
[`DAILY_DIARY.md`](DAILY_DIARY.md) — describe a fictional engineer, which is
what the questions in `batch_test.py` and `sim.py` ask about. `USER.md` and
`SOUL.md` are no longer loaded by anything: they fed a profile-overlay path
that was removed because no supported API consumed it.

## Reading the results

The two retrieval products return different shapes on purpose:

- **`retrieve_context`** → `ContextBundle`. A Pydantic model, read by attribute.
  Its `facts` are `Fact` domain objects carrying a subject-predicate-object
  triple — there is no `.text` field. `chunks` and `episodic` are `Chunk` and
  `Episode`.
- **`retrieve_memory`** → `MemoryResult`. Also read by attribute, but its
  `facts` and `evidence` are deliberately narrower **dict** projections, so
  those are read by key (`fact["text"]`).

Neither supports `.get()`. If you find yourself reaching for dict access on a
bundle, you want the attribute.
