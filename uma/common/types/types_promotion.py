"""
Types for the memory-promotion feature.

Contains :class:`AgentProfile` — the read-side view of a per-agent
scope description stored in the procedural store as a Skill row with
``kind='agent_profile'``. Consulted by
:meth:`PromotionPolicy.qualifies_for_agent_kb` to decide which
user-owned facts qualify for elevation into the agent's global KB.

Also contains :class:`QualifierDecision` — the structured return value
of ``qualifies_for_agent_kb`` so callers and tests can distinguish
which gate failed (quarantine, eligibility, scope match) without
parsing log lines.

Coding agent instructions
-------------------------
- Keep these types minimal. They are read-shapes and decision records
  only; policy logic lives in ``uma.memory.promotion``.
- Do not add mutable fields on ``AgentProfile``; the qualifier treats
  it as an immutable snapshot for the duration of a promotion pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentProfile:
    """Read-side representation of an agent-profile row.

    Attributes
    ----------
    agent_id
        The agent this profile describes. Not prefixed with ``"agent:"``.
        The Skill row that backs it uses ``owner_id=f'agent:{agent_id}'``.
    description
        Free-text description of the agent's scope. Used to compute
        ``profile_embedding`` at write time via the configured embedder.
    focus_areas
        Deterministic keyword list. The promotion qualifier's fast path
        checks whether any focus_area appears (case-insensitive) in the
        candidate fact's ``subject + predicate + object`` text.
    profile_embedding
        The description's embedding, stored inline in the SQL row
        (``profile_embedding`` BLOB) so agent_profile rows never enter
        the vector index — this closes the retrieval-leakage risk.
    tenant_id
        Tenant scope. Defaults to ``"default"`` for single-tenant
        deployments.
    """

    agent_id: str
    description: str
    focus_areas: list[str]
    profile_embedding: list[float]
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("AgentProfile.agent_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("AgentProfile.description must be a non-empty string")
        if not isinstance(self.focus_areas, list):
            raise ValueError("AgentProfile.focus_areas must be a list of strings")
        for item in self.focus_areas:
            if not isinstance(item, str):
                raise ValueError("AgentProfile.focus_areas entries must be strings")
        if not isinstance(self.profile_embedding, list) or not self.profile_embedding:
            raise ValueError(
                "AgentProfile.profile_embedding must be a non-empty list of floats"
            )
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("AgentProfile.tenant_id must be a non-empty string")


@dataclass(frozen=True)
class QualifierDecision:
    """Structured result from :meth:`PromotionPolicy.qualifies_for_agent_kb`.

    Callers use ``passed`` for the gating boolean and ``reasons`` for
    diagnostic logging. The individual per-gate flags exist so tests
    can assert *which* gate short-circuited without parsing strings.

    Reason codes (order-matching the gate sequence in the qualifier):
        - ``"quarantined"``  — fact.quarantined_at is set
        - ``"ineligible"``   — is_eligible returned False
        - ``"scope_mismatch"`` — neither deterministic nor embedding
                                 branch matched the agent's profile
    """

    passed: bool
    reasons: list[str] = field(default_factory=list)
    quarantine_ok: bool = False
    is_eligible: bool = False
    scope_matched: bool = False

