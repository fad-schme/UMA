---
name: uma-promotion
description: Explains UMA's public profile-gated fact promotion feature, including set_agent_profile and get_agent_profile, eligibility and scope-match gates, immutable per-agent instances, ownership widening, provenance, background execution, and safe no-op behavior. Use this skill for questions about making turn facts durable, agent knowledge, promotion safety, or promotion troubleshooting.
---

# UMA — Fact Promotion

## Purpose

Facts extracted by `process_turn` are session-local by default. Promotion is
UMA's conservative path for copying a qualifying fact into a broader durable
scope without changing or deleting the source fact.

Promotion is agent-specific, and the agent is named on every call:

```python
from uma import UMAMemory

memory = UMAMemory.from_yaml("config/uma.yaml")
AGENT = "infrastructure-agent"
```

The promotion policy is constructed from the `agent_id` of the turn being
processed. One instance serves every agent, and no agent can promote into
another agent's KB.

## Enable Promotion with an Agent Profile

```python
profile = await memory.set_agent_profile(
    agent_id=AGENT,
    description="An infrastructure assistant focused on Kubernetes operations",
    focus_areas=["kubernetes", "containers", "incident response"],
    tenant_id="default",   # optional; defaults to "default"
)
```

The profile is stored in the procedural SQL store as
`kind="agent_profile"`. Its description is embedded for semantic scope
matching, but the profile row is not added to the procedural vector index.
Calling `set_agent_profile` again for the same tenant and agent replaces the
profile.

Read it with:

```python
profile = await memory.get_agent_profile(tenant_id="default")
```

If this returns `None`, automatic promotion is a no-op. Setting a profile is
the explicit opt-in.

## What Happens During `process_turn`

After semantic extraction, UMA schedules a bounded background promotion pass.
The reply path does not wait for it. Promotion failures are logged and do not
make `process_turn` fail.

Each candidate must pass, in order:

1. Quarantine gate — quarantined facts never promote.
2. Eligibility gate — the fact must meet confidence, salience, source,
   predicate, subject, object-length, and sensitive-content rules.
3. Profile gate — a focus-area keyword must match the fact, or its embedding
   must meet the configured-in-code cosine threshold against the profile.
4. Scope gate — tenant identity must be preserved and the target scope must be
   an allowed one-hop widening.

At most five facts are promoted per turn by the default policy.

## Ownership and Scope

The default policy widens one scope level at a time:

| Source | Target |
| --- | --- |
| Session-local user fact | Durable fact owned by the same user |
| Durable user fact | Fact owned by the scoped agent |

Promotion never crosses tenants and never targets system scope. Cross-agent
sharing remains denied unless a fact is deliberately copied into the intended
agent scope.

## Copy and Provenance Contract

Promotion creates a new deterministic fact ID. It does not mutate the source.
The promoted fact:

- preserves source chunk IDs and origin fields;
- clears `session_id` because the target is durable;
- records the source fact, source owner, source session, target owner, tenant,
  policy version, and reason under `meta["promotion"]`;
- remains idempotent for the same source and target.

This copy-based contract makes scope widening explicit and auditable.

## Safe Defaults

- A scoped instance has a default agent-bound `PromotionPolicy`.
- No agent profile means no automatic promotion.
- Preferences and ephemeral predicates are blocked by default.
- Chat and working-memory sources are blocked.
- Likely secrets and personal identifiers are blocked.
- Facts without source provenance or embeddings do not promote.
- Background failures never broaden access and never fail the turn.

## Troubleshooting

If an expected fact was not promoted, check:

- `get_agent_profile()` returns a profile for the same tenant and agent;
- the extracted fact is not quarantined;
- confidence and salience pass policy thresholds;
- source type and source chunk provenance are present;
- a focus area matches, or the fact/profile embeddings are sufficiently close;
- the background turn task has had time to finish before inspecting storage.

Promotion is intentionally conservative. A rejected candidate stays in its
original scope and remains available through the normal scoped retrieval path.
