from __future__ import annotations

import os
from datetime import datetime
from uuid import uuid4

from uma.types_fact import Fact
from uma.core.utils.identity import ensure_user_subject


async def load_documents_folder(folder: str, topic: str, memory, user_id: str) -> int:
    """Load .txt/.md/.pdf documents from `folder` and upsert as semantic facts.

    Each document becomes a Fact with subject "user:<id>" and predicate "document".

    Returns number of documents ingested.
    """
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith((".txt", ".md", ".pdf")):
                files.append(os.path.join(root, fn))

    if not files:
        return 0

    count = 0
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
                    # skip unreadable PDFs
                    text = ""
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
        except Exception:
            text = ""

        title = os.path.basename(path)
        # Trim very large documents for storage in object; full text still embedded
        obj = {"title": title, "path": path, "text": text[:10000]}

        subject = ensure_user_subject(user_id)
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
            vectors = await memory.embedder.embed([text or ""])
            embedding = vectors[0] if vectors else []
            await memory.semantic_store.upsert_fact(fact, embedding)
            count += 1
        except Exception:
            # best-effort: continue on failures
            continue

    return count
