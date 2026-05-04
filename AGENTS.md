codex.AGENT.md — UMA-RLM (Codex Coding Agent Guide)

You are a senior engineer. Understand the codebase deeply: its intent, component relationships, and how operations behave across the system. Before changing anything, inspect existing code and patterns, reuse what already exists, and add only the smallest clean change needed. Never duplicate functionality, over-engineer, or add code unless it is truly necessary.

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

UMA-RLM is a memory and context manager SDK for developers building AI agents. It ingests data (documents, conversations), stores it in multiple lanes (SQL, vectors, graph), and exposes two thin, sharp retrieval products: curated context retrieval for RAG and evidence-backed memory retrieval over compiled knowledge.

What UMA-RLM is NOT
	•	Not a chat app.
	•	Not a “big prompt builder”.
	•	Not a knowledge-graph-first system (graph is a supporting lane, not the only lane).
	•	Not a framework that keeps legacy/backward compatibility: remove obsolete paths.


Core product principle

RLM is always enabled. Retrieval is LLM-controlled search for context, not answering. The “long context” lives in the environment, not in the prompt.

One-sentence product test

A user must be able to understand what UMA-RLM is in one sentence and install it in one path.


Design implication
If a feature, module split, wrapper, helper, or API surface makes the system harder to explain, harder to install, or harder to trace end-to-end, simplify it before extending it.

Primary goals for every PR

Every PR must preserve or improve these properties:
• understandable
• lean
• easy to trace end-to-end
• production-appropriate
• evidence-backed where relevant
• consistent with canonical UMA paths
• free of conceptual clutter

Do not optimize for theoretical flexibility, framework elegance, or future abstraction at the cost of clarity.
Optimize for useful, direct, maintainable code.

⸻

1) Non-negotiable invariants (DAT / Ownership / Safety)

DAT invariants (must hold end-to-end)

Every stored/retrieved memory artifact must be owner-scoped.
	•	owner_type ∈ {agent, user, workspace, system}
	•	owner_id is required and consistent with the lane
	•	tenant_id is required for durable artifacts and must be preserved end-to-end
	•	session-local artifacts must also carry session_id and agent_id, and user_id when applicable
	•	Applies to: chunks, facts, episodes, graph nodes/edges, promotions

Graph must preserve provenance and ownership:
	•	fact → graph edge must carry owner_type, owner_id, fact_id, source_chunk_id, timestamps
	•	episode → fact edges must also carry ownership

No unscoped reads:
	•	Any read path that returns memory must require ownership filters or derive them deterministically from an explicit immutable runtime context.
	•	Runtime scope and persistent ownership are different concepts and must not be conflated.
	•	If an internal “unsafe query” exists, it must be strongly gated (private, explicit name, logs) — otherwise remove.

Practical rule

If you can’t answer: “which user/agent/project is allowed to see this row/edge?” then the design is wrong.

Runtime scope invariants (must hold end-to-end)

No shared mutable object may store current request scope.
	•	Do not implement or preserve runtime paths that depend on ambient mutable state such as memory.agent_id, memory.user_id, controller.current_scope, or equivalent patterns.
	•	Shared services must be stateless with respect to request identity.

Every runtime entry point must operate from explicit immutable context.
	•	Required runtime fields are context-dependent, but the canonical model includes: tenant_id, agent_id, request_id, optional user_id, optional workspace_id, optional session_id, and optional trace/policy metadata.
	•	Request scope must never be inferred from prior calls or stored mutable fields on shared objects.

Turn-derived memory defaults.
	•	Working memory is session-local by default.
	•	Episodic turn memory is session-local by default.
	•	Semantic facts extracted from turns are session-local by default and must be explicitly promoted to become durable user/workspace/agent memory.

Cross-scope behavior.
	•	Cross-tenant access must be impossible by construction.
	•	Cross-agent sharing is denied by default unless the artifact owner is intentionally broader than agent scope.
	•	Any scope widening must be explicit, auditable, and provenance-preserving.

⸻

2) Retrieval lanes (what exists, what they mean)

UMA-RLM retrieval uses multiple “lanes”:
	1.	Working Memory (WM) — short conversational continuity.
	2.	Semantic facts — structured statements extracted from chunks/episodes.
	3.	Raw evidence chunks — authoritative source text discovered via vector + lexical retrieval.
	4.	Episodic — time-ordered memory of interactions / ingest events.
	5.	Graph — relationship navigation / predicate-scoped expansion.
	6.	Compiled wiki pages — mutable, evidence-backed continuity artifacts used for synthesis and memory retrieval.


Lane responsibilities
	•	Vector search: candidate discovery (fast recall). Adapter must return (id, score).
	•	Lexical search: candidate discovery for exact terms/IDs; implemented as an OPTIONAL capability on the same adapter as vector search.
	•	Ranking: one canonical module owns hybrid fusion + optional rerank + truncation. No distributed ranking logic.
	•	SQL: authoritative source of chunk text and metadata; vector DB stores only what’s needed for filtering and id mapping.
	•	Graph: routing/index for entity/predicate expansion, not primary truth.
	•	Facts: preferred “truth layer”; chunks are evidence.

Canonical storage model

Raw evidence
• Stored through the normal document ingest path.
• Immutable.
• Source of truth for evidence.
• Chunked and indexed through the canonical UMA ingest flow.
• Expected metadata includes at minimum: kind="raw_source", kb_lane="raw", plus canonical ownership/scope metadata.

Compiled wiki pages
• Stored in UMA as normal documents or durable records, but logically marked as compiled artifacts.
• Mutable and updatable.
• Evidence-backed.
• Used for continuity, synthesis, and memory retrieval.
• Expected metadata includes at minimum: kind="wiki_page", kb_lane="wiki", page_slug, page_title, category, status.

Markdown projection
• wiki/*.md is a projection only.
• Human-readable, git-friendly, and Obsidian-friendly.
• Never the canonical source of truth.

Thin retrieval products
• Context retrieval: high-quality RAG over stored data to produce curated context for the LLM.
• Memory retrieval: compiled-knowledge retrieval over evidence-backed artifacts; not plain chunk retrieval and not “RAG with nicer formatting”.

Lane policy rule
Memory lanes are a first-class retrieval contract. Retrieval planning must choose lanes explicitly. Ownership alone is not enough to distinguish user KB, user profile, agent KB, raw evidence, and compiled knowledge.

⸻

3) Absolute design constraints for this repo

Backward compatibility

Do not implement legacy fallbacks. Remove obsolete paths rather than preserving them.

Before making any change:
	•	Inspect the existing code, architecture, conventions, and related files.
	•	Infer the real intent of the request from the current codebase context.
	•	Search for existing helpers, utilities, abstractions, and patterns that already solve part or all of the problem.
	•	Determine whether the requested behavior already exists in another form.

Mandatory execution stance
• Simplify before extending when the topology is confusing.
• Prefer direct code over indirection.
• Converge duplicate concepts into one canonical path.
• Keep public surfaces small and sharp.
• Avoid parallel paths for the same behavior.
• If a task can be solved either by adding a new helper/wrapper/abstraction or by simplifying and extending an existing canonical path, choose the second unless there is a strong concrete reason not to.

Code style
	•	Keep files lean and modular: one responsibility per module.
	•	Prefer explicit, direct code over cleverness or indirection.
	•	Do not over-engineer.
	•	Keep the codebase understandable, lean, and easy to trace end-to-end.
	•	Remove thin wrappers, pass-through helpers, and duplicated utility layers that only forward data, rename concepts, or hide the real path.
	•	Do not add abstractions, adapters, helper types, or module splits unless they remove real duplication or make a core contract materially clearer.
	•	Prefer extending canonical paths over creating alternate execution paths.
	•	Add logs only where they pay for themselves (start/end of operations, error context, key counters).
	•	Do not overbuild plugin systems, adapters, or abstraction layers.

When implementing:
	•	Add only the minimum new code needed.
	•	Prefer reuse, extension, or small refactors over introducing new duplicate logic.
	•	Never duplicate functionality that already exists.
	•	Keep changes lean, clean, and consistent with the surrounding code.
	•	Preserve existing abstractions unless there is a clear reason to improve them.
	•	Avoid speculative abstraction, overengineering, or introducing new layers without need.

Error handling
	•	Never swallow exceptions silently in core flows.
	•	Always log with enough context to debug safely: tenant_id, owner scope, relevant ids, counts, and operation name.
	•	Prefer “safe empty” returns only when explicitly intended by design.
	•	Keep error handling close to the boundary where recovery is meaningful; otherwise fail clearly and preserve context.
	•	Do not add defensive catch-all logic that hides broken invariants.

Comments and logging
	•	Keep comments high-signal and durable. Explain invariants, contracts, and non-obvious reasoning, not line-by-line mechanics.
	•	Do not add redundant comments that merely restate the code.
	•	Use logging to capture state transitions, scope decisions, counts, and failures that are operationally useful.
	•	Do not log huge payloads, full prompts, or large text blobs in normal flows.
	•	When adding logs, keep field naming consistent across modules.


Implementation rules for redesign work
	•	Every change must include or update tests when behavior, contracts, or storage semantics change.
	•	Do not merge changes that leave the codebase in a partially migrated or internally inconsistent state.
	•	Keep the implementation lean: do not add abstractions, layers, or helper types unless they remove real duplication or make a core contract clearer.
	•	Do not duplicate functionality across runtime, controller, store, feature, or helper layers.
	•	Prefer extending canonical paths over creating alternate execution paths.
	•	Ensure new code does not increase accidental complexity or overengineering.
	•	Keep compatibility wrappers thin, temporary, and only at boundary layers.

Codebase simplification rule
• PR1-class cleanup work takes priority over feature growth when the code topology is confusing.
• The coding agent must always keep the codebase understandable, lean, and free of thin wrappers, pass-through helpers, and duplicated utility layers that only forward data or rename concepts.
• If two modules express the same concept with slightly different names, shapes, or helper layers, converge them into one canonical path.
• If a new feature cannot be added without adding conceptual clutter, simplify first.

Canonical path rule
For each changed behavior, there must be one obvious path.
A contributor should be able to answer:
• where does it start?
• where is the main decision made?
• where is the real implementation?
• what does it return?
• how is it tested?

If the answer requires jumping across many wrappers or mirrored helpers, the path is too diffuse.

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

The same discipline applies to product paths: UMA-RLM must expose a small number of thin, sharp canonical paths, not a growing set of overlapping flows. “Context retrieval” and “memory retrieval” may share lower-level primitives, but they must remain distinct contracts with distinct outputs.

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

Testing rules
• Every behavior, contract, storage, or migration change must include or update tests.
• Prefer testing canonical paths, product behavior, artifact boundaries, retrieval contracts, and end-to-end scenarios for the changed flow.
• Do not rely only on tiny helper tests if the real risk is path drift, contract confusion, or artifact boundary regression.
• Tests should help a human understand the intended behavior.

Documentation rules
• Every PR must keep documentation aligned with runtime reality.
• If a PR changes behavior, contracts, canonical paths, install/use flow, or artifact semantics, update the relevant docs, comments, and examples in the same PR.
• Do not leave docs describing an older architecture.

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
	•	Prefer workspace over project in new code and docs unless you are editing a legacy compatibility surface.
	•	Prefer explicit runtime context objects over loose parameter bundles when request scope is required.
	•	Use doc_id consistently for document provenance.
	•	Use chunk_id for chunk identity (don’t alias multiple fields).
	•	Prefer direct product-facing names for public APIs where new names are introduced (for example: ingest_source, retrieve_context, retrieve_memory, update_wiki_page, export_wiki_projection).
	•	Do not expose internal architecture complexity through public API naming.

⸻

12) “Lean initialization” guidance (startup performance)

Rule

Initialize only what’s required for the selected profile:
	•	Installation and first-run setup must have one obvious path.
	•	Optional components must never complicate the baseline install story.
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
	•	✅ tests added or updated for every behavior, contract, storage, or migration change
	•	✅ no ambient mutable request scope remains in the changed execution path
	•	✅ code stays lean, avoids duplication, and does not introduce unnecessary abstractions
	•	✅ the changed path is easy to explain in one sentence and easy to trace through one canonical flow
	•	✅ no new thin wrappers, pass-through helpers, or rename-only utility layers were introduced
	•	✅ context retrieval vs memory retrieval contracts remain explicit where relevant
	•	✅ no obsolete fallback/legacy paths remain
	•	✅ logs show counts and decisions (without dumping large text)

Reuse & consistency
	•	✅ changes reuse existing functionality when it exists (no duplicate implementations)
	•	✅ public surfaces remain small, sharp, and understandable to a new user
	•	✅ fixes/changes are end-to-end complete, never partial (all call sites updated, all return types consistent)
	•	✅ comments, logging, and error handling follow the repo rules and remain production-appropriate
	•	✅ code remains production-ready: consistent interfaces, clear contracts, and stable behavior

Robustness
	•	✅ all relevant code paths include proper error handling, meaningful logging, and actionable messages
	•	✅ failures degrade safely (bounded outputs, safe empty returns only where designed)