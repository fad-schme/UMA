from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from uma import UMARuntime
from uma.core.retrieval.rlm.context_pack import ContextPack
from uma.core.retrieval.rlm.evidence import expand_evidence_chunks_from_facts
from uma.core.retrieval.rlm.request import RetrievalRequest
from uma.stores.base_sql_store import DEFAULT_TENANT_ID
from uma.types import RuntimeContext


@dataclass
class _Chunk:
    id: str
    owner_type: str
    owner_id: str
    tenant_id: str = "default"
    meta: dict = field(default_factory=dict)


class _EvidenceEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []
        self._memory = type(
            "_MemoryCfg",
            (),
            {
                "retrieval_cfg": type("_Cfg", (), {"max_evidence_chunks": 6})(),
            },
        )()

    async def fetch_chunks(self, request: RetrievalRequest, *, ids, owner_type, owner_id):
        self.calls.append((owner_type, owner_id, list(ids)))
        return [_Chunk(id=f"{owner_type}:{chunk_id}", owner_type=owner_type, owner_id=owner_id) for chunk_id in ids]


@pytest.mark.asyncio
async def test_evidence_expansion_fetches_chunks_by_source_fact_owner_scope() -> None:
    env = _EvidenceEnv()
    request = RetrievalRequest.from_runtime_context(
        RuntimeContext(
            tenant_id="tenant-1",
            agent_id="agent:alpha",
            request_id="req-evidence",
            user_id="user:u1",
        )
    )
    pack = ContextPack(
        user_id="user:u1",
        query_text="hello world",
        owner_type="user",
        owner_id="user:u1",
        facts=[
            {"id": "fact-agent", "owner_type": "agent", "owner_id": "agent:alpha", "source_ids": ["chunk-agent"]},
            {"id": "fact-user", "owner_type": "user", "owner_id": "user:u1", "source_ids": ["chunk-user"]},
        ],
    )

    chunks = await expand_evidence_chunks_from_facts(
        env=env,
        request=request,
        pack=pack,
        max_items_per_type=10,
    )

    assert env.calls == [
        ("agent", "agent:alpha", ["chunk-agent"]),
        ("user", "user:u1", ["chunk-user"]),
    ]
    assert {chunk.owner_type for chunk in chunks} == {"agent", "user"}
    assert {chunk.owner_id for chunk in chunks} == {"agent:alpha", "user:u1"}


@pytest.mark.asyncio
async def test_bound_context_retrieval_is_isolated_across_agents_on_shared_runtime(uma_memory, tmp_path) -> None:
    memory = uma_memory
    runtime = UMARuntime.from_memory(memory)

    doc_a = tmp_path / "agent_a.txt"
    doc_a.write_text(
        (
            "Agent alpha KB document. It mentions shared keyword and alpha-only guidance. "
            "This sentence pads the content so chunking remains valid in CI.\n"
        ),
        encoding="utf-8",
    )
    doc_b = tmp_path / "agent_b.txt"
    doc_b.write_text(
        (
            "Agent beta KB document. It mentions shared keyword and beta-only guidance. "
            "This sentence pads the content so chunking remains valid in CI.\n"
        ),
        encoding="utf-8",
    )
    user_doc = tmp_path / "user_doc.txt"
    user_doc.write_text(
        (
            "User-owned document. It mentions shared keyword for the same user. "
            "This sentence pads the content so chunking remains valid in CI.\n"
        ),
        encoding="utf-8",
    )

    await memory.ingest_document(str(doc_a), owner_type="agent", owner_id="agent:alpha")
    await memory.ingest_document(str(doc_b), owner_type="agent", owner_id="agent:beta")
    await memory.ingest_document(str(user_doc), owner_type="user", owner_id="user:u1")

    handle_a = runtime.bind(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent:alpha",
            request_id="req-alpha",
            user_id="user:u1",
        )
    )
    handle_b = runtime.bind(
        RuntimeContext(
            tenant_id=DEFAULT_TENANT_ID,
            agent_id="agent:beta",
            request_id="req-beta",
            user_id="user:u1",
        )
    )

    ctx_a, ctx_b = await asyncio.gather(
        handle_a.retrieve_structured_context("shared keyword"),
        handle_b.retrieve_structured_context("shared keyword"),
    )

    owner_pairs_a = {(getattr(chunk, "owner_type", None), getattr(chunk, "owner_id", None)) for chunk in ctx_a.get("chunks") or []}
    owner_pairs_b = {(getattr(chunk, "owner_type", None), getattr(chunk, "owner_id", None)) for chunk in ctx_b.get("chunks") or []}
    tenant_ids_a = {getattr(chunk, "tenant_id", None) for chunk in ctx_a.get("chunks") or []}
    tenant_ids_b = {getattr(chunk, "tenant_id", None) for chunk in ctx_b.get("chunks") or []}

    assert ("agent", "agent:alpha") in owner_pairs_a
    assert ("agent", "agent:beta") not in owner_pairs_a
    assert ("agent", "agent:beta") in owner_pairs_b
    assert ("agent", "agent:alpha") not in owner_pairs_b
    assert ("user", "user:u1") in owner_pairs_a
    assert ("user", "user:u1") in owner_pairs_b
    assert tenant_ids_a == {"default"}
    assert tenant_ids_b == {"default"}
