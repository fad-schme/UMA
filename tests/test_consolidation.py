"""Consolidation cycle: per-call scope and the tenant it reads and writes.

Consolidation had no coverage, which is how it came to reference an
attribute that never existed on any object it held. These tests pin the
scope contract: tenant and user arrive per call, every store touch carries
them, and a failed fetch is reported rather than reported as "no episodes".
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pytest

from uma.api.management import consolidate
from uma.common.registry import FeatureContext
from uma.common.types import Episode, Fact
from uma.common.types.types_scope import DEFAULT_TENANT_ID
from uma.memory.consolidation.consolidator import Consolidator
from uma.memory.consolidation.feature import ConsolidationFeature


TENANT = "tenant-a"
USER = "user:u1"


def _episode(episode_id: str) -> Episode:
    return Episode(
        id=episode_id,
        user_id=USER,
        timestamp=datetime.now(timezone.utc),
        summary=f"summary for {episode_id}",
        raw=f"raw for {episode_id}",
        tenant_id=TENANT,
        owner_type="user",
        owner_id=USER,
    )


class RecordingEpisodicCore:
    """Records the scope of every call the consolidator makes."""

    def __init__(self, episodes: list[Episode] | None = None) -> None:
        self.episodes = episodes if episodes is not None else []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_recent(self, tenant_id, *, owner_type, owner_id, n):
        self.calls.append(
            ("list_recent", {"tenant_id": tenant_id, "owner_type": owner_type, "owner_id": owner_id})
        )
        return list(self.episodes)

    async def upsert_cluster_summary(self, user_id, *, tenant_id, owner_type, owner_id, **kwargs):
        self.calls.append(
            ("upsert_cluster_summary", {"tenant_id": tenant_id, "owner_type": owner_type, "owner_id": owner_id})
        )
        return True

    async def delete_episode(self, episode_id, *, tenant_id, owner_type, owner_id):
        self.calls.append(
            ("delete_episode", {"tenant_id": tenant_id, "owner_type": owner_type, "owner_id": owner_id})
        )
        return True


class RecordingSemanticCore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.upserted: list[Fact] = []

    async def list_facts_for_owner(self, *, tenant_id, owner_type, owner_id):
        self.calls.append(
            ("list_facts_for_owner", {"tenant_id": tenant_id, "owner_type": owner_type, "owner_id": owner_id})
        )
        return []

    async def upsert_fact(self, fact, embedding):
        self.upserted.append(fact)
        return True

    async def delete_fact(self, fact_id, *, tenant_id, owner_type, owner_id):
        self.calls.append(
            ("delete_fact", {"tenant_id": tenant_id, "owner_type": owner_type, "owner_id": owner_id})
        )
        return True


class StubEmbedder:
    dimension = 3

    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class StubLLM:
    async def generate(self, *args, **kwargs):
        return ""


def _build(episodic, semantic, *, extracted: list[Fact] | None = None) -> Consolidator:
    consolidator = Consolidator(
        episodic_core=episodic,
        semantic_core=semantic,
        llm=StubLLM(),
        embedder=StubEmbedder(),
    )

    class _Summarizer:
        async def summarize_cluster(self, cluster_text):
            return "a distilled summary of the cluster"

    class _Extractor:
        async def extract_user_facts(self, *, tenant_id, **kwargs):
            # Mirrors the real FactExtractor: the tenant it is handed is the
            # one it stamps on every fact it returns.
            facts = list(extracted or [])
            for fact in facts:
                fact.tenant_id = tenant_id
            return facts

    consolidator.summarizer = _Summarizer()
    consolidator.extractor = _Extractor()
    return consolidator


def _fact(fact_id: str) -> Fact:
    now = datetime.now(timezone.utc)
    return Fact(
        id=fact_id,
        subject="user",
        predicate="prefers",
        object="dark mode in the editor",
        created_at=now,
        updated_at=now,
        confidence=0.9,
        salience=0.9,
        owner_type="user",
        owner_id=USER,
    )


@pytest.mark.asyncio
async def test_run_once_scopes_every_store_call_to_the_requested_tenant() -> None:
    """Regression: these call sites read a `self.memory` that never existed."""
    episodic = RecordingEpisodicCore([_episode("ep1"), _episode("ep2")])
    semantic = RecordingSemanticCore()
    consolidator = _build(episodic, semantic)

    await consolidator.run_once(USER, tenant_id=TENANT)

    assert episodic.calls, "consolidation made no episodic calls at all"
    for name, scope in [*episodic.calls, *semantic.calls]:
        assert scope["tenant_id"] == TENANT, f"{name} used tenant {scope['tenant_id']!r}"
        assert scope["owner_type"] == "user", name
        assert scope["owner_id"] == USER, name


@pytest.mark.asyncio
async def test_run_once_defaults_to_the_single_tenant_value() -> None:
    episodic = RecordingEpisodicCore([_episode("ep1")])
    consolidator = _build(episodic, RecordingSemanticCore())

    await consolidator.run_once(USER)

    assert episodic.calls[0][1]["tenant_id"] == DEFAULT_TENANT_ID


@pytest.mark.asyncio
async def test_distilled_facts_are_written_into_the_requested_tenant() -> None:
    """The call's tenant reaches the extractor and lands on what is persisted."""
    semantic = RecordingSemanticCore()
    consolidator = _build(
        RecordingEpisodicCore([_episode("ep1")]),
        semantic,
        extracted=[_fact("fact-1")],
    )

    facts = await consolidator.run_once(USER, tenant_id=TENANT)

    assert [f.id for f in facts] == ["fact-1"]
    assert [f.tenant_id for f in semantic.upserted] == [TENANT]


@pytest.mark.asyncio
async def test_run_once_normalizes_a_raw_user_id() -> None:
    episodic = RecordingEpisodicCore([_episode("ep1")])
    consolidator = _build(episodic, RecordingSemanticCore())

    await consolidator.run_once("u1", tenant_id=TENANT)

    assert episodic.calls[0][1]["owner_id"] == "user:u1"


@pytest.mark.asyncio
async def test_a_failed_episode_fetch_is_raised_not_reported_as_empty() -> None:
    """A swallowed fetch error is indistinguishable from "no episodes"."""

    class BrokenEpisodicCore(RecordingEpisodicCore):
        async def list_recent(self, *args, **kwargs):
            raise RuntimeError("episodic store is down")

    consolidator = _build(BrokenEpisodicCore(), RecordingSemanticCore())

    with pytest.raises(RuntimeError, match="episodic store is down"):
        await consolidator.run_once(USER, tenant_id=TENANT)


@pytest.mark.asyncio
async def test_no_episodes_is_a_clean_empty_cycle() -> None:
    consolidator = _build(RecordingEpisodicCore([]), RecordingSemanticCore())

    assert await consolidator.run_once(USER, tenant_id=TENANT) == []


# ── the public entrypoint: uma.api.management.consolidate ─────────────


class _MemoryStub:
    def __init__(self, feature: ConsolidationFeature) -> None:
        self.features: dict[str, Any] = {"consolidation": feature}

    def _ensure_ingestion_ready(self) -> None:
        """Features are already attached; nothing to warm up in this stub."""


def _memory_for(consolidator: Consolidator) -> _MemoryStub:
    """Attach a ConsolidationFeature wrapping the given Consolidator, without booting a UMAMemory."""
    feature = ConsolidationFeature(
        episodic_core=consolidator.episodic_core,
        semantic_core=consolidator.semantic_core,
        llm=StubLLM(),
        embedder=StubEmbedder(),
    )
    # attach() wires the feature's own Consolidator; swap in the stubbed one.
    feature.consolidator = consolidator

    memory = _MemoryStub(feature)
    feature.attach(
        FeatureContext(
            memory=memory,
            config={},
            services={},
            logger=logging.getLogger("test"),
        )
    )
    return memory


@pytest.mark.asyncio
async def test_consolidation_run_carries_tenant_and_user_per_call() -> None:
    episodic = RecordingEpisodicCore([_episode("ep1")])
    memory = _memory_for(_build(episodic, RecordingSemanticCore()))

    result = await consolidate(memory, user_id=USER, tenant_id=TENANT)

    assert result.ok
    assert episodic.calls[0][1]["tenant_id"] == TENANT


@pytest.mark.asyncio
async def test_consolidation_run_rejects_a_missing_user() -> None:
    memory = _memory_for(_build(RecordingEpisodicCore(), RecordingSemanticCore()))

    result = await consolidate(memory, user_id="")

    assert not result.ok
    assert "invalid user_id" in result.errors


@pytest.mark.asyncio
async def test_consolidation_run_surfaces_a_broken_cycle_as_a_failure() -> None:
    """The feature boundary is where recovery belongs, not the fetch."""

    class BrokenEpisodicCore(RecordingEpisodicCore):
        async def list_recent(self, *args, **kwargs):
            raise RuntimeError("episodic store is down")

    memory = _memory_for(_build(BrokenEpisodicCore(), RecordingSemanticCore()))

    result = await consolidate(memory, user_id=USER, tenant_id=TENANT)

    assert not result.ok
    assert any("episodic store is down" in error for error in result.errors)
