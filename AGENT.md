codex.AGENT.md — UMA-RLM (Codex Coding Agent Guide)

Purpose of this file

This file exists to make an AI coding agent productive in UMA-RLM quickly, safely, and consistently. It is the single source of truth for:
	•	project goals and non-goals
	•	invariants (DAT + ownership scoping)
	•	how to run tests and validate changes
	•	how to implement patches in a lean way (no code clutter)

(AGENTS.md concept: a dedicated, predictable “README for agents”.)

⸻

0) Project identity (read this first)

What UMA-RLM is

UMA-RLM is a memory and context manager SDK for developers building AI agents. It ingests data (documents, conversations), stores it in multiple lanes (SQL, vectors, graph), and retrieves high-quality compressed context.

What UMA-RLM is NOT
	•	Not a chat app.
	•	Not a “big prompt builder”.
	•	Not a knowledge-graph-first system (graph is a supporting lane, not the only lane).
	•	Not a framework that keeps legacy/backward compatibility: remove obsolete paths.

Core product principle

RLM is always enabled. Retrieval is LLM-controlled search for context, not answering. The “long context” lives in the environment, not in the prompt.

⸻

1) Non-negotiable invariants (DAT / Ownership / Safety)

DAT invariants (must hold end-to-end)

Every stored/retrieved memory artifact must be owner-scoped.
	•	owner_type ∈ {agent, user, project}
	•	owner_id is required and consistent with the lane
	•	Applies to: chunks, facts, episodes, graph nodes/edges, promotions

Graph must preserve provenance and ownership:
	•	fact → graph edge must carry owner_type, owner_id, fact_id, source_chunk_id, timestamps
	•	episode → fact edges must also carry ownership

No unscoped reads:
	•	Any read path that returns memory must require ownership filters (or derive them deterministically from the request context).
	•	If an internal “unsafe query” exists, it must be strongly gated (private, explicit name, logs) — otherwise remove.

Practical rule

If you can’t answer: “which user/agent/project is allowed to see this row/edge?” then the design is wrong.

⸻

2) Retrieval lanes (what exists, what they mean)

UMA-RLM retrieval uses multiple “lanes”:
	1.	Working Memory (WM) — always included; short conversational continuity.
	2.	Semantic facts — structured statements extracted from chunks/episodes.
	3.	Chunks — authoritative source text (from SQL), discovered via vector + lexical search.
	4.	Episodic — time-ordered memory of interactions / ingest events.
	5.	Graph — relationship navigation / predicate-scoped expansion.

Lane responsibilities
	•	Vector search: candidate discovery (fast recall). Adapter must return (id, score).
	•	Lexical search: candidate discovery for exact terms/IDs; implemented as an OPTIONAL capability on the same adapter as vector search.
	•	Ranking: one canonical module owns hybrid fusion + optional rerank + truncation. No distributed ranking logic.
	•	SQL: authoritative source of chunk text and metadata; vector DB stores only what’s needed for filtering and id mapping.
	•	Graph: routing/index for entity/predicate expansion, not primary truth.
	•	Facts: preferred “truth layer”; chunks are evidence.

⸻

3) Absolute design constraints for this repo

Backward compatibility

Do not implement legacy fallbacks. Remove obsolete paths rather than preserving them.

Code style
	•	Keep files lean and modular: one responsibility per module
	•	Avoid over-abstraction; prefer small helpers
	•	Prefer explicitness over cleverness
	•	Add logs only where they pay for themselves (start/end of operations, error context, key counters)

Error handling
	•	Never swallow exceptions silently in core flows
	•	Always log with enough context to debug (owner scope, ids, counts)
	•	Prefer “safe empty” returns only when explicitly intended by design

⸻

4) Chunking & storage rules (critical for retrieval quality)

Chunking is the #1 lever for downstream snippet quality.

Required chunking rules
	•	Never cut mid-sentence.
	•	Never produce chunks that “start like a fragment”.
	•	Prefer paragraph-level chunks.
	•	If paragraph is too long, split by sentences, but keep ≥2 sentences per chunk.
	•	Avoid too-short chunks (<80 chars).
	•	Overlap must align to sentence boundaries, not token counts.
	•	Overlap should never start mid-sentence.

Required chunk metadata

Every chunk must carry (at minimum):
	•	id, doc_id, text, position, page_range
	•	ownership: owner_type, owner_id (and any user_id/agent_id/project_id you track)
	•	provenance: source_uri/hash if available

Storage semantics
	•	SQL is authoritative for chunk text and must always be the source of truth during rendering.
	•	Vector store is an accelerator and must be rebuildable from SQL.
	•	Vector store payload must be minimal (ids + filterable metadata). Do not duplicate full chunk text in the vector DB unless explicitly required by a feature.

⸻

5) Fact extraction & discoverability (avoid “facts exist but RLM can’t see them”)

Requirements
	•	Facts must be generated deterministically enough that ingestion can’t “silently create nothing” without visibility:
	•	log extraction counts per doc and per chunk window
	•	log parse failures (non-JSON)
	•	Facts must be retrievable by the RLM environment:
	•	do not hard-bind subject namespaces in a way that prevents retrieval
	•	treat “subject” as an attribute of facts, not the primary retrieval gate for doc-derived knowledge

Practical guidance
	•	Use ownership scoping to isolate access.
	•	Use subject filters only as a ranking/boost mechanism, not a hard requirement, unless it is a recall-intent query.

⸻

6) Snippets: what “final_snippets” must represent

Principle

Chunks are retrieval units; snippets are evidence units.

Output contract

A final snippet must be:
	•	coherent standalone text
	•	bounded (length limits)
	•	traceable to sources (doc_id, chunk_ids, page_range)
	•	relevance-filtered and non-fragmentary

Determinism
	•	Deterministic prefiltering should remove obvious junk.
	•	LLM evaluation can refine/score, but must be bounded and safe.

⸻

7) Single-path rule (avoid gold-vs-app inconsistencies)

Required policy

There must be one canonical production path from query → context pack → rendered snippet.

Canonical retrieval pipeline

All production callers must follow this sequence:
	1.	Candidate retrieval: dense vector search (top_k_dense) plus optional lexical search (top_k_sparse), both owner-scoped.
	2.	Fusion: merge dense + lexical candidates (RRF or simple boost-on-overlap), producing a single candidate pool.
	3.	Optional rerank: rerank only within the candidate pool. Rerank must never expand the pool.
	4.	Selection: deterministic truncation to max_chunks/max_facts.
	5.	Snippet rendering: presentation-only (merge adjacency, bound length, preserve traceability).

Policy: do not implement ranking inside stores, snippet rendering, or controller layers.

Both:
	•	gold runner
	•	example app

must use the same:
	•	retrieval entry point
	•	context pack type (prefer one canonical representation)
	•	snippet rendering pipeline

Any parallel “dict-shaped vs object-shaped” path is technical debt:
	•	remove it or make it a thin adapter around the canonical pipeline.

⸻

8) Development workflow (how to work in this repo)

Local setup
	•	Create venv, install deps (project-specific)
	•	Ensure Neo4j / vector backend config is correct (or use stubs/mocks in tests)

Run tests

Preferred:
	•	PYTHONPATH=. python3 pytest -q

While iterating:
	•	run only failing tests first
	•	then run full suite before committing

Logging
	•	Keep logs structured and consistent: include trace_id, owner_type/owner_id, counts
	•	Avoid logging huge blobs of text

⸻

9) Patch expectations (how to submit changes)

Every patch must include
	•	tests updated/added where behavior changes
	•	removal of obsolete code paths (don’t keep unused helpers “just in case”)
	•	consistent parameter naming across layers (e.g., object not obj)
	•	end-to-end alignment: caller args match callee signature and return types

No “just to make the test pass”

If a unit test disagrees with baseline design:
	•	explain the mismatch
	•	patch the test only if the baseline design is correct and intentional

⸻

10) Debugging checklist (use when behavior is surprising)

When retrieval output looks wrong:
	1.	Confirm ingestion produced chunks with correct ownership + doc metadata.
	2.	Confirm vector retrieval returns ids AND scores end-to-end (no score dropped between adapter and selector).
	3.	Confirm facts were extracted and persisted (counts, thresholds, parse errors).
	4.	Confirm:
	    • dense + lexical candidates were both considered when enabled (hybrid)
	    • fusion + optional rerank ran in the canonical ranking module
	5.	Confirm snippet refiner removed fragments and short/junk chunks.
	6.	Confirm final snippets are exactly what the agent sees (not “raw chunks”).

When gold runner and app differ:
	•	inspect which entry points were used
	•	verify both call the same snippet rendering pipeline
	•	ensure both operate on the same pack type

⸻

11) Naming conventions & consistency rules
	•	Prefer object over obj everywhere.
	•	Prefer owner_type/owner_id over ad-hoc user_id gating.
	•	Use doc_id consistently for document provenance.
	•	Use chunk_id for chunk identity (don’t alias multiple fields).

⸻

12) “Lean initialization” guidance (startup performance)

Rule

Initialize only what’s required for the selected profile:
	•	profile="retrieval" must boot minimal retrieval stack fast.
	•	Heavy components (graph connections, clustering, optional vector stores) should:
	•	initialize lazily on first use, or
	•	initialize asynchronously after startup (but must fail safely if used before ready)

Contract

If a component is optional and not initialized:
	•	environment methods must return safe empty results + log a single warning (not spam)
	•	controller must degrade gracefully to baseline retrieval

⸻

13) Lean ranking rules (hybrid + rerank)

Goal

Improve retrieval accuracy without spreading ranking logic across the codebase.

Rules
	•	One ranking module owns fusion + optional rerank + final score computation.
	•	Stores/adapters return candidates and raw signals; they do not decide final ranking.
	•	Vector similarity score is mandatory plumbing (id, score) from adapter → selector.
	•	Lexical retrieval is optional and implemented as a capability on the SAME adapter as vector search.
	•	Lexical results must be fused with dense results BEFORE rerank.
	•	Reranking is optional and must be post-retrieval only (reorder candidates, never expand).
	•	All ranking must remain owner-scoped; never rerank across mixed owners.

Implementation guidance
	•	Prefer extending existing scoring helpers (e.g., lexical_score) rather than adding new scoring paths.
	•	Expose a debug “score card” per candidate (vector_score, lexical_score, rerank_score, final_score) under a flag.

⸻

14) What to ask for when you need more context

If you’re missing necessary files to patch correctly, ask for:
	•	the exact module that defines the type you’re working with (e.g., Fact, ContextPack)
	•	the environment API signatures used by the controller
	•	the retrieval service entry point used by the example app
	•	the module that currently computes lexical_score / snippet scoring so ranking can be consolidated without duplication

Do not request the whole repo; request only specific files.

⸻

15) Definition of Done

A change is “done” only when all of the following are true:

Quality & correctness
	•	✅ unit tests pass
	•	✅ gold runner and example app follow the same canonical pipeline
	•	✅ snippet output is coherent (no fragments, no 1-line junk)
	•	✅ ownership scoping enforced across all reads
	•	✅ no obsolete fallback/legacy paths remain
	•	✅ logs show counts and decisions (without dumping large text)

Reuse & consistency
	•	✅ changes reuse existing functionality when it exists (no duplicate implementations)
	•	✅ fixes/changes are end-to-end complete, never partial (all call sites updated, all return types consistent)
	•	✅ code remains production-ready: consistent interfaces, clear contracts, and stable behavior

Robustness
	•	✅ all relevant code paths include proper error handling, meaningful logging, and actionable messages
	•	✅ failures degrade safely (bounded outputs, safe empty returns only where designed)