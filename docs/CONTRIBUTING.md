Contributing to UMA-RLM

This guide explains how to validate, debug, and extend UMA-RLM using Gold Context Packs. It is written for engineers and coding agents working on memory ingestion, retrieval, and the RLM controller.

Rule #0: UMA-RLM is a memory and context manager, not a QA system. You debug retrieval quality by inspecting context, not answers.

⸻

1. What a Gold Context Pack Is

A Gold Context Pack is the expected retrieval outcome for a given user prompt.

It defines:
	•	Which facts must be present
	•	Which grounding chunks must support those facts
	•	Which graph relationships should appear
	•	When the RLM controller should stop

It does not define:
	•	The final answer
	•	Prompt wording
	•	Chain-of-thought

Canonical Schema

{
  "working_memory": [...],
  "facts": [
    { "subject", "predicate", "object", "salience", "confidence" }
  ],
  "chunks": ["supporting text"],
  "episodes": [...],
  "graph": [...],
  "coverage": {
    "semantic_enough": true,
    "graph_support": true,
    "novelty_recent": ">0"
  }
}


⸻

2. How to Use Gold Context Packs

Step 1 — Run Retrieval Only

Do not generate an answer.

pack = await rlm_controller.retrieve_context(user_id, prompt)


⸻

Step 2 — Compare UMA Output to Gold Pack

Evaluate evidence, not prose.

A. Facts
	•	Core facts from the Gold Pack must be present
	•	Minor phrasing differences are acceptable
	•	Ownership scope must be correct

Fail if:
	•	Key concepts are missing
	•	Facts are generic or irrelevant

B. Chunks (Grounding)
	•	Facts must be supported by chunks
	•	Chunks must come from the correct documents

Fail if:
	•	Facts have no grounding
	•	Chunks are duplicated or boilerplate

C. Graph
	•	Expected entity relationships must appear
	•	Traversal must be bounded

Fail if:
	•	Graph is empty when structure is expected
	•	Graph explodes with irrelevant nodes

D. Coverage & Stop Behavior
	•	semantic_enough == True when Gold expects stop
	•	RLM should converge in reasonable steps

Fail if:
	•	RLM loops without novelty
	•	RLM stops too early

⸻

3. Debugging Guide

Missing Facts
	•	Check search_semantic parameters
	•	Verify owner scoping
	•	Check dedup rules

Missing Chunks
	•	Ensure fetch_chunks_by_ids is executed
	•	Verify source_chunk_id propagation

Missing Graph
	•	Verify fact → graph ingestion
	•	Check predicate normalization
	•	Inspect graph expansion predicates

Excessive Looping
	•	Inspect novelty_history
	•	Check predicate offsets
	•	Review coverage thresholds

⸻

4. Writing New Gold Context Packs
	1.	Identify what an expert must know
	2.	Define facts first, not text
	3.	Add minimal grounding chunks
	4.	Add graph relationships only if they add value
	5.	Decide stop conditions explicitly

If a human expert needs it, UMA-RLM should retrieve it.

⸻

5. Automated Testing Pattern

def test_rlm_retrieval():
    pack = run_rlm(prompt)
    assert_has_facts(pack, GOLD_FACTS)
    assert_has_chunks(pack)
    assert_has_graph_edges(pack, GOLD_GRAPH)
    assert pack.coverage.semantic_enough

Use semantic matching, not string equality.

⸻

6. Definition of Done

UMA-RLM retrieval is correct when:
	•	Gold facts are consistently present
	•	Context is concise and sufficient
	•	Retrieval converges deterministically
	•	Adding documents improves context naturally

⸻

7. Final Principle

Never debug UMA-RLM by looking at answers.
Always debug it by comparing retrieved context to the Gold Context Pack.

If the context is right, the agent can reason.
If the context is wrong, no prompt engineering will help.

⸻

Happy hacking — and keep memory boring, deterministic, and auditable.