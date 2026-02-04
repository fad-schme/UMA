# UMA-RLM Production Readiness Checklist

This document summarizes practical steps to validate UMA-RLM for production
deployment. It is intentionally concise and actionable.

## Scope Summary
UMA-RLM is a modular Python SDK for agent memory and context management. It
combines episodic, facts, skills, and graph memories with an RLM-based
retrieval controller that can recursively query the environment instead of
packing long context into the prompt.
It provides configuration hooks for storage and observability, but developers
own deployment, infrastructure operations, and data lifecycle management.
UMA-RLM only retrieves context from its stores; agent behavior, reasoning, and
response generation are managed by developers.

## Release Gates (Must Be Green)
- Integration tests pass against the intended DB/vector/graph backends.
- LLM/embedding providers validated with retry/timeout behavior under load.
- Observability endpoints enabled (logs, metrics, tracing) and verified.
- Data durability confirmed (backups, restore tests, migration plan).
- Security review completed (secrets handling, PII redaction, rate limits).

## Core Stability
- Use Postgres in production; keep SQLite for dev/test only.
- Use a persistent vector backend (Pinecone/Weaviate) in production; keep FAISS/InMemory for dev/CI.
- Enable graph backends (Neo4j/Memgraph) only when needed and properly configured.
- Validate timeouts/retries for LLM and vector clients per environment.

## Reliability & Error Handling
- Review retry/backoff settings for all external services (LLM, vector DB).
- Ensure critical failures are logged at error level with context.
- Consider a reconciliation job for DB↔vector consistency if required by your SLA.
- Add error boundaries around dependency calls to keep UMA-RLM read paths resilient.

## Data Consistency & Recovery
- Provide a maintenance job to rebuild vector indexes from SQL data using `UMAMemory.rebuild_vector_indexes(...)`.
- Document recovery steps for partial index rebuilds (per user or per store).
- `user_id` is required to rebuild episodic/facts indexes; skills rebuilds can run globally.

## Observability
- Structured logs (JSON) with request IDs and user/session identifiers.
- Metrics: latency, error rate, queue size, token usage.
- Tracing (OpenTelemetry) around LLM calls, DB operations, vector queries.

## Security
- Secrets in env/secret manager, never in config files.
- Sanitize logs for sensitive content (PII, raw messages).
- Rate‑limit inbound requests and validate inputs.
- Treat config warnings about secrets as deployment blockers.

## Performance
- Load test ingestion + retrieval with realistic concurrency.
- Tune LLM batch sizes, vector query top‑k, and DB pool sizes.
- Confirm per‑request timeouts match your SLA.
- Use `python3 scripts/perf_retrieval.py` to measure end-to-end retrieval latency.
- Re-run perf against your production DB/vector backends and record p50/p95/p99 latencies.

## Operational Readiness
- Add liveness/readiness health checks that probe DB/vector/graph dependencies.
- Plan DB schema migrations (versioned migrations with rollback).
- Define backup/restore for Postgres and vector DB.

## Deployment
- Containerize with pinned dependencies and reproducible builds.
- Set resource limits and autoscaling policies (CPU/memory).
- Document and verify release/rollback procedures.

## Recommended Minimum Validation
- Run the full test suite (`python3 -m pytest`).
- Run a basic integration test with your intended DB/vector/graph backends.
- Confirm error visibility via logs/metrics dashboards.
- Add an application-level readiness gate using `UMAMemory.health_check()`.

## Validation Log
- Unit tests: passed locally (`python3 -m pytest`, 28 tests).
- Integration tests: out of scope (developers validate target backends).
- Observability verification: out of scope (developer environment dependent).
- Backup/restore drills: out of scope (developer-owned DB procedures).
