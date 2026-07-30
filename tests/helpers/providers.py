from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Optional


def _extract_user_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            parts.append(str(m.get("content") or ""))
    return "\n".join(parts).strip()


def _first_sentence(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = re.search(r"^(.+?[.!?])(?:\\s|$)", t)
    if m:
        return m.group(1).strip()
    return t.splitlines()[0].strip()


def _ensure_min_words(text: str, min_words: int) -> str:
    t = " ".join((text or "").split()).strip()
    if min_words <= 0:
        return t
    words = t.split()
    if len(words) >= min_words:
        return t
    # Pad deterministically with a neutral filler token.
    words.extend(["detail"] * (min_words - len(words)))
    return " ".join(words).strip()


def _stable_uuid_int(s: str) -> int:
    h = hashlib.sha256((s or "").encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


async def fake_embed(
    *,
    texts: Optional[list[str]] = None,
    dimension: int = 64,
    **_kwargs: Any,
) -> list[list[float]]:
    """
    Deterministic, offline embedding for tests via seeded PRNG.

    Produces stable vectors across runs/CI without external services.
    """
    dim = int(dimension)
    out: list[list[float]] = []
    for t in list(texts or []):
        seed = _stable_uuid_int(f"embed:v1:{dim}:{t}")
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        out.append(vec)
    return out


async def fake_llm(
    *,
    messages: Optional[list[dict[str, str]]] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    **_kwargs: Any,
) -> str:
    """
    Deterministic, offline LLM callable for tests.

    Behavior:
    - If the prompt requests JSON (common in semantic extraction), returns schema-valid JSON.
    - Otherwise returns a short summary-like string.
    """
    _ = int(max_tokens)
    _ = float(temperature)
    msgs = list(messages or [])
    system = ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            system = str(m.get("content") or "")
            break

    user_text = _extract_user_text(msgs)
    wants_json = "Return ONLY valid JSON" in system or "\"facts\"" in system or "\"chunks\"" in system

    if not wants_json:
        # WorkingMemory compaction / consolidation summarizer uses plain text.
        t = _first_sentence(user_text) or "OK."
        return t

    # --- Semantic extractor schemas ---
    # Batch chunk facts: user content is JSON {"chunks":[{"chunk_id":"...","text":"..."}]}
    if "\"chunks\":" in system and "\"chunk_id\"" in system:
        try:
            payload = json.loads(user_text or "{}")
        except Exception:
            payload = {}
        items = payload.get("chunks") if isinstance(payload, dict) else None
        chunks_out: dict[str, Any] = {}
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                chunk_id = str(it.get("chunk_id") or "").strip()
                text = str(it.get("text") or "")
                if not chunk_id:
                    continue
                obj = _ensure_min_words(_first_sentence(text) or "Chunk contains information.", 5)
                chunks_out[chunk_id] = {
                    "facts": [
                        {
                            "subject": "Document",
                            "predicate": "STATES",
                            "object": obj,
                            "confidence": 0.8,
                        }
                    ]
                }
        return json.dumps({"chunks": chunks_out})

    # User facts schema: {"facts":[{"predicate":"likes","object":"sushi",...}]}
    if "LONG-TERM, STABLE facts" in system and "\"predicate\"" in system:
        # Extract the TEXT block if present (FactExtractor formats as "SUBJECT: ...\nTEXT:\n...").
        txt = user_text
        if "TEXT:" in txt:
            txt = txt.split("TEXT:", 1)[1]
        txt = " ".join((txt or "").split()).strip()

        facts: list[dict[str, Any]] = []

        research_match = re.search(r"\bresearch(?:ing)?\b\s+([^.;\n]+)", txt, flags=re.IGNORECASE)
        if research_match:
            obj = research_match.group(1).strip(" .")
            if obj:
                facts.append(
                    {
                        "predicate": "RESEARCHING",
                        "object": obj,
                        "confidence": 0.8,
                        "source_ids": [],
                    }
                )

        interest_match = re.search(r"\binterested in\b\s+([^.;\n]+)", txt, flags=re.IGNORECASE)
        if interest_match:
            tail = interest_match.group(1).strip(" .")
            parts = re.split(r"\b(?:and|or)\b", tail, flags=re.IGNORECASE)
            for part in parts:
                obj = " ".join(part.split()).strip(" .")
                if obj:
                    facts.append(
                        {
                            "predicate": "INTERESTED_IN",
                            "object": obj,
                            "confidence": 0.8,
                            "source_ids": [],
                        }
                    )

        if facts:
            return json.dumps({"facts": facts[:6]})

        # Heuristic: pull multiple likes from patterns like "likes sushi and pizza".
        likes: list[str] = []
        m = re.search(r"\\blikes\\b\\s+([^\\.;\\n]+)", txt, flags=re.IGNORECASE)
        if m:
            tail = m.group(1)
            tail = re.sub(r"\\b(and|or)\\b", ",", tail, flags=re.IGNORECASE)
            parts = [p.strip() for p in tail.split(",") if p.strip()]
            for p in parts:
                # Keep the head noun-ish token for determinism (avoid full clause).
                token = re.sub(r"[^a-zA-Z0-9_-]+", " ", p).strip().split()
                if token:
                    likes.append(token[0].lower())
        else:
            # Fallback: pull obvious tokens from the text itself.
            lowered = txt.lower()
            for candidate in ("sushi", "pizza", "coffee", "tea"):
                if candidate in lowered:
                    likes.append(candidate)
            if not likes:
                tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", lowered)
                likes = tokens[-2:] if tokens else []

        if not likes:
            likes = ["coffee"]

        facts = [
            {"predicate": "LIKES", "object": obj, "confidence": 0.7, "source_ids": []}
            for obj in likes[:6]
        ]
        return json.dumps({"facts": facts})

    # Summary facts schema (document summary to facts)
    if "Summarize the document into KB-grade facts" in system:
        obj = _ensure_min_words(_first_sentence(user_text) or "Document summary.", 5)
        return json.dumps(
            {
                "facts": [
                    {
                        "subject": "Document",
                        "predicate": "SUMMARY",
                        "object": obj,
                        "confidence": 0.7,
                    }
                ]
            }
        )

    # Default: single chunk facts schema {"facts":[...]}
    obj = _ensure_min_words(_first_sentence(user_text) or "Text contains information.", 5)
    return json.dumps(
        {
            "facts": [
                {
                    "subject": "Document",
                    "predicate": "STATES",
                    "object": obj,
                    "confidence": 0.8,
                }
            ]
        }
    )
