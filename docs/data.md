Below is a formal DAT (Data Architecture & Transformation) invariant checklist for UMA-RLM.

This is written as a governance + engineering artifact:
	•	you can hand it to contributors,
	•	use it for code reviews,
	•	and turn it into automated assertions over time.

It is strict, non-ambiguous, and maps directly to the ingestion & retrieval flows you validated.

⸻

UMA-RLM — DAT Invariant Checklist

Purpose

The DAT invariants define what must always be true about data as it moves through UMA-RLM:
	•	from raw unstructured input
	•	through chunking, embedding, extraction
	•	into authoritative storage
	•	and finally into retrieval + RLM context construction

Violating these invariants leads to:
	•	silent data corruption
	•	incorrect recall
	•	broken access control
	•	unreliable RLM behavior

Canonical Storage Taxonomy

Persisted UMA artifacts use one explicit storage language defined in `uma.common.storage_metadata`.

Kinds
	•	raw_source — immutable source evidence stored through canonical ingest
	•	wiki_page — mutable compiled wiki artifact backed by evidence
	•	semantic_fact — durable semantic knowledge extracted from evidence
	•	episodic_event — time-ordered episodic or import event
	•	procedural_rule — durable procedural instruction or skill
	•	profile_fact — profile-oriented continuity fact
	•	decision_trace — persisted decision trace artifact
	•	query_artifact — persisted query-oriented artifact

Lanes
	•	raw
	•	wiki
	•	semantic
	•	episodic
	•	procedural
	•	profile
	•	trace

Shared metadata vocabulary
	•	kind
	•	kb_lane
	•	owner_type
	•	owner_id
	•	scope
	•	source_id
	•	source_type
	•	created_at
	•	updated_at
	•	provenance
	•	status

Projection rule
	•	`wiki/*.md` is projection/export only
	•	canonical wiki state must live in UMA records, not markdown files

⸻

0. Global Invariants (Apply Everywhere)

G-1. Authoritative vs Derived Storage
	•	SQL stores are authoritative
	•	Vector indexes are always derived
	•	Graph is always derived

❌ Vector or graph data must never be treated as source-of-truth
❌ Deleting vector data must never delete SQL data
✔ SQL must allow full reconstruction of vector + graph layers

⸻

G-2. Stable Identifiers

Every persisted object must have a stable ID:

Object	ID Requirement
Document	deterministic or content-hash based
Chunk	stable across re-ingestion
Fact	stable per fact triple + scope
Episode	unique, time-based
Graph edge	derived from fact ID

IDs must:
	•	be immutable
	•	be globally unique within their namespace

⸻

G-3. Ownership Is Mandatory

Every persisted artifact must carry ownership metadata:
	•	owner_type ∈ {user, project, agent}
	•	owner_id (stable string)

This applies to:
	•	documents
	•	chunks
	•	facts
	•	episodes
	•	graph edges

❌ No ownership-less rows allowed
❌ No implicit ownership inference

⸻

1. Document Ingestion Invariants

D-1. Manifest First

Before any chunks are persisted:

✔ A Document Manifest must exist in SQL
✔ Must include:
	•	doc_id
	•	source_path
	•	source_hash
	•	ingested_at
	•	ownership metadata

❌ Chunks must never exist without a document manifest

⸻

D-2. Immutability of Source
	•	source_hash must never change
	•	Re-ingesting identical content must:
	•	reuse doc_id
	•	update derived layers only if necessary

⸻

2. Chunking Invariants

C-1. Chunk Completeness

Each chunk must include:
	•	chunk_id
	•	doc_id
	•	text
	•	page_range
	•	position
	•	ownership metadata
	•	provenance (source_hash, source_type)

❌ Chunks without text or doc linkage are invalid

⸻

C-2. Chunk Independence

A chunk must be:
	•	independently retrievable
	•	independently embeddable
	•	independently auditable

Chunks may overlap textually, but not logically.

⸻

C-3. Chunk Ordering
	•	position must reflect document order
	•	Ordering must be stable across re-ingestion

⸻

3. Embedding Invariants

E-1. Embedder Contract

All embedders must satisfy:

embed(List[str]) -> List[List[float]]

	•	Fixed dimension per embedder
	•	Deterministic dimension validation

❌ Single-string embedding interfaces are forbidden

⸻

E-2. Embedding ≠ Authority

Embeddings:
	•	must never be the only stored representation
	•	must always reference an authoritative object ID

⸻

4. Semantic Fact Invariants

F-1. Fact Schema

Every fact must include:
	•	fact_id
	•	subject
	•	predicate
	•	object
	•	confidence ∈ [0,1]
	•	salience ∈ [0,1]
	•	provenance:
	•	source_chunk_id
	•	doc_id
	•	ownership metadata

⸻

F-2. Predicate Canonicalization

Predicates must be:
	•	uppercase
	•	normalized
	•	stable across ingestion runs

❌ Free-form predicates are forbidden

⸻

F-3. Deduplication Responsibility
	•	Fact deduplication is the responsibility of the Fact SQL Store
	•	Ingestion may emit duplicates; store must reconcile

⸻

5. Episodic Memory Invariants

EP-1. Episodic Purpose

Episodes represent:
	•	events
	•	summaries
	•	temporal groupings

They must never duplicate full facts.

⸻

EP-2. Episodic Optionality
	•	Episodic memory is optional per ingestion
	•	Retrieval must function without episodic data

⸻

6. Graph Invariants

GPH-1. Graph Is Derived

Graph edges must:
	•	be derived from facts
	•	reference fact_id
	•	carry ownership metadata

❌ No direct graph mutation without a fact

⸻

GPH-2. Predicate-Scoped Edges

Edges must be typed by predicate:

(subject) -[:PREDICATE]-> (object)

Graph traversal must always be:
	•	depth-bounded
	•	predicate-scoped when possible

⸻

7. Consolidation & Promotion Invariants

CONS-1. Asynchronous Only

Consolidation must:
	•	never block ingestion
	•	never block retrieval

⸻

CONS-2. Promotion Safety

Promotion (user → agent memory) must:
	•	preserve provenance
	•	preserve original owner
	•	record promotion metadata

❌ Promotion must never overwrite original facts

⸻

8. Retrieval Invariants

R-1. Working Memory Always Included

WM must always be present in final context.

No exceptions.

⸻

R-2. Retrieval Is Additive

Retrieval may:
	•	weight
	•	rank
	•	truncate

But must never:
	•	mutate authoritative storage
	•	delete data

⸻

R-3. Policy-Driven Selection

Recall intent and ownership influence:
	•	ranking
	•	stopping

They must never hard-filter stores.

⸻

R-4. RLM Boundedness

RLM retrieval must enforce:
	•	max steps
	•	max environment calls
	•	token budget

⸻

9. ContextPack Invariants

CP-1. Context ≠ Prompt

ContextPack must:
	•	be structured
	•	be auditable
	•	be prompt-agnostic

Agents decide how to render it.

⸻

CP-2. Traceability

Every ContextPack must contain:
	•	retrieval steps
	•	stop reason
	•	warnings (if any)

⸻

10. Failure Invariants

FAIL-1. Partial Failure Safety

If any stage fails:
	•	previously persisted data must remain valid
	•	ingestion must be resumable

⸻

FAIL-2. Silent Failure Forbidden

All failures must:
	•	be logged
	•	produce warnings
	•	never silently drop data

⸻

Summary (Why This Matters)

If all DAT invariants hold:
	•	UMA-RLM can rebuild everything
	•	Retrieval quality is predictable
	•	RLM recursion is safe
	•	Multi-agent expansion is possible
	•	Compliance & auditing are tractable

This checklist is the contract that makes UMA-RLM trustworthy.

⸻

If you want next, I can:
	•	turn this into assertions you can automate
	•	add runtime invariant checks (cheap ones)
	•	or produce a “DAT violations playbook” explaining how bugs manifest when invariants break
