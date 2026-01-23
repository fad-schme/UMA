from __future__ import annotations

import logging
import os
from datetime import datetime
from uuid import uuid4

from uma.types_fact import Fact
from uma.core.utils.identity import ensure_user_subject

logger = logging.getLogger(__name__)

def _chunk_text(text: str, *, max_chars: int = 800, overlap: int = 80) -> list[str]:
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    overlap = max(0, min(overlap, max_chars - 1))
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + max_chars)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = end - overlap
    return chunks


async def load_documents_folder(folder: str, topic: str, memory, user_id: str) -> int:
    """Load .txt/.md/.pdf documents from `folder` and upsert as semantic facts.

    Each chunk becomes a Fact with subject "user:<id>" and predicate "document".

    Returns number of chunks ingested.
    """
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith((".txt", ".md", ".pdf")):
                files.append(os.path.join(root, fn))

    if not files:
        logger.warning("No supported documents found under %s", folder)
        return 0

    count = 0
    failed = 0
    for path in files:
        text = None
        try:
            if path.lower().endswith(".pdf"):
                # Lazy import to keep deps optional
                try:
                    import PyPDF2

                    with open(path, "rb") as fh:
                        reader = PyPDF2.PdfReader(fh)
                        pages = [p.extract_text() or "" for p in reader.pages]
                        text = "\n".join(pages)
                except Exception:
                    logger.exception("Failed to extract PDF text: %s", path)
                    text = ""
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
        except Exception:
            logger.exception("Failed to read document: %s", path)
            text = ""

        title = os.path.basename(path)
        chunks = _chunk_text(text or "")
        if not chunks:
            continue

        subject = ensure_user_subject(user_id)
        for idx, chunk in enumerate(chunks, start=1):
            obj = {"title": title, "path": path, "text": chunk, "chunk": idx}
            fact = Fact(
                id=str(uuid4()),
                subject=subject,
                predicate="document",
                object=obj,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                source_ids=[path],
                confidence=1.0,
                meta={"source": "local_folder", "topic": topic},
            )

            # Compute embedding and upsert
            try:
                vectors = await memory.embedder.embed([chunk or ""])
                embedding = vectors[0] if vectors else []
                await memory.semantic_store.upsert_fact(fact, embedding)
                count += 1
            except Exception:
                failed += 1
                logger.exception("Failed to embed or upsert chunk for %s", path)
                continue

    if count == 0 and files:
        logger.warning("No documents ingested (files=%d, failed_chunks=%d).", len(files), failed)
    return count
