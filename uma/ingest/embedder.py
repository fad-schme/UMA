from __future__ import annotations

import logging
from typing import Dict, List, Any

from uma.adapters.llm.retry_utils import retryable, should_retry_network_only
from .types import DocumentChunk

logger = logging.getLogger(__name__)


def _batched(items: List[Any], batch_size: int) -> List[List[Any]]:
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _validate_embedding(vec: List[float], expected_dim: int | None = None) -> None:
    if not isinstance(vec, list) or not vec:
        raise ValueError("embedder: embedding is empty")
    if not all(isinstance(x, (float, int)) for x in vec):
        raise ValueError("embedder: embedding contains non-numeric values")
    if expected_dim is not None and len(vec) != expected_dim:
        raise ValueError(
            f"embedder: embedding dim mismatch expected={expected_dim} got={len(vec)}"
        )


async def embed_chunks(
    chunks: List[DocumentChunk],
    *,
    embedder: Any,
    batch_size: int = 16,
    expected_dim: int | None = None,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 8.0,
    strict: bool = True,
) -> Dict[str, List[float]]:
    """
    Batch-embed chunk texts with retries.

    Returns mapping: chunk_id -> embedding vector
    """
    if not chunks:
        return {}
    if embedder is None or not hasattr(embedder, "embed"):
        raise ValueError("embed_chunks: embedder with .embed() required")

    results: Dict[str, List[float]] = {}

    @retryable(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        max_delay=max_delay,
        should_retry=should_retry_network_only,
    )
    async def _embed_batch(texts: List[str]) -> List[List[float]]:
        return await embedder.embed(texts)

    for batch_idx, batch in enumerate(_batched(chunks, batch_size)):
        texts = [c.text or "" for c in batch]
        try:
            vectors = await _embed_batch(texts)
        except Exception as exc:
            logger.exception(
                "embed_chunks: embedding batch failed batch_idx=%d size=%d",
                batch_idx,
                len(batch),
            )
            if strict:
                raise RuntimeError(
                    f"embed_chunks strict: batch failed batch_idx={batch_idx} size={len(batch)}"
                ) from exc
            continue

        if not isinstance(vectors, list) or len(vectors) != len(batch):
            logger.error(
                "embed_chunks: invalid embedder result size batch_idx=%d expected=%d got=%r",
                batch_idx,
                len(batch),
                len(vectors) if isinstance(vectors, list) else type(vectors),
            )
            if strict:
                raise RuntimeError(
                    f"embed_chunks strict: result size mismatch batch_idx={batch_idx} expected={len(batch)}"
                )
            continue

        for chunk, vec in zip(batch, vectors):
            try:
                _validate_embedding(vec, expected_dim)
                results[chunk.chunk_id] = [float(x) for x in vec]
            except Exception as exc:
                logger.exception(
                    "embed_chunks: invalid embedding for chunk_id=%s batch_idx=%d",
                    chunk.chunk_id,
                    batch_idx,
                )
                if strict:
                    raise RuntimeError(
                        f"embed_chunks strict: invalid embedding chunk_id={chunk.chunk_id} batch_idx={batch_idx}"
                    ) from exc
                continue

    if strict:
        missing = [c.chunk_id for c in chunks if c.chunk_id not in results]
        if missing:
            raise RuntimeError(
                f"embed_chunks strict: missing embeddings count={len(missing)} sample={missing[:5]}"
            )

    return results
