import pytest

from uma.adapters.llm.callable_adapter import CallableEmbedderAdapter, CallableLLMAdapter


def _bad_llm(no_messages):
    return "ok"


async def _good_llm(messages, **kwargs):
    return "ok"


def _bad_embedder(texts):
    return [[0.0, 1.0]]


async def _good_embedder(texts):
    return [[0.0, 1.0, 2.0]]


def test_callable_llm_preflight_rejects_missing_messages():
    with pytest.raises(TypeError):
        CallableLLMAdapter(_bad_llm, preflight=True)


@pytest.mark.asyncio
async def test_callable_llm_returns_string():
    adapter = CallableLLMAdapter(_good_llm, preflight=True)
    out = await adapter.generate(messages=[{"role": "user", "content": "hi"}])
    assert out == "ok"


@pytest.mark.asyncio
async def test_callable_embedder_dimension_mismatch():
    adapter = CallableEmbedderAdapter(_bad_embedder, dimension=3, preflight=True)
    with pytest.raises(RuntimeError):
        await adapter.embed(["hi"])


@pytest.mark.asyncio
async def test_callable_embedder_ok():
    adapter = CallableEmbedderAdapter(_good_embedder, dimension=3, preflight=True)
    out = await adapter.embed(["hi"])
    assert out == [[0.0, 1.0, 2.0]]
