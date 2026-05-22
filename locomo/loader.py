"""LoCoMo dataset loader.

Handles the native LoCoMo schema (locomo10.json-style):

  [
    {
      "sample_id": "conv-26",
      "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
          {"speaker": "Caroline", "dia_id": "D1:1", "text": "..."},
          ...
        ],
        "session_2_date_time": "...",
        "session_2": [...],
        ...
      },
      "qa": [
        {"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 2},
        ...
      ],
      "event_summary": {...},     # paper-provided memory artifacts — ignored
      "observation": {...},       # ignored: they encode the answers
      "session_summary": {...}    # ignored
    },
    ...
  ]

Also keeps a permissive fallback for simpler/test-shaped JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
SESSION_DT_KEY_RE = re.compile(r"^session_(\d+)_date_time$")


@dataclass(frozen=True)
class TurnRecord:
    index: int               # global turn index within the conversation
    speaker: str
    text: str
    timestamp: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QARecord:
    question_id: str
    question: str
    expected_answer: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: str
    turns: list[TurnRecord]
    qa_items: list[QARecord]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_locomo_dataset(path: str) -> list[ConversationRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_items = _extract_items(data)
    out: list[ConversationRecord] = []
    for index, item in enumerate(raw_items):
        record = _normalize_item(item, index=index)
        if record.turns and record.qa_items:
            out.append(record)
    return out


# --------------------------------------------------------------- shape detection


def _extract_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("conversations", "dialogs", "dialogues", "data", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        # single conversation object
        return [data]
    raise ValueError("LoCoMo dataset must be a JSON object or array.")


def _normalize_item(item: dict[str, Any], *, index: int) -> ConversationRecord:
    conversation_id = str(
        item.get("sample_id")
        or item.get("conversation_id")
        or item.get("id")
        or item.get("dialog_id")
        or f"conversation_{index}"
    )

    conv = item.get("conversation")
    if isinstance(conv, dict) and _looks_like_locomo_native(conv):
        turns = _flatten_native_sessions(conv)
    else:
        turns = _flatten_generic_turns(item)

    qa_items = _normalize_qa_items(item, conversation_id=conversation_id)

    metadata = {
        "sample_id": conversation_id,
        # Carry speaker hints up — useful for sanity checks and reporting.
        "speakers": _native_speakers(conv) if isinstance(conv, dict) else None,
    }
    return ConversationRecord(
        conversation_id=conversation_id,
        turns=turns,
        qa_items=qa_items,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


def _looks_like_locomo_native(conv: dict[str, Any]) -> bool:
    has_speakers = "speaker_a" in conv or "speaker_b" in conv
    has_session_payload = any(SESSION_KEY_RE.match(k) for k in conv.keys())
    return has_speakers or has_session_payload


def _native_speakers(conv: dict[str, Any]) -> list[str] | None:
    speakers = [conv.get("speaker_a"), conv.get("speaker_b")]
    speakers = [s for s in speakers if isinstance(s, str) and s.strip()]
    return speakers or None


# --------------------------------------------------------------- turn flattening


def _flatten_native_sessions(conv: dict[str, Any]) -> list[TurnRecord]:
    """Walk session_1, session_2, ... in numeric order and emit TurnRecords.

    Each turn carries its session number and date-time as metadata so the
    adapter can group by session and surface timestamps in the document.
    """
    session_payloads: dict[int, list[dict[str, Any]]] = {}
    session_dates: dict[int, str | None] = {}

    for key, value in conv.items():
        m = SESSION_KEY_RE.match(key)
        if m and isinstance(value, list):
            session_payloads[int(m.group(1))] = [t for t in value if isinstance(t, dict)]
            continue
        m = SESSION_DT_KEY_RE.match(key)
        if m and isinstance(value, str):
            session_dates[int(m.group(1))] = value.strip() or None

    turns: list[TurnRecord] = []
    global_index = 0
    for session_n in sorted(session_payloads.keys()):
        session_dt = session_dates.get(session_n)
        for turn in session_payloads[session_n]:
            text = _first_str(turn, ("text", "content", "message", "utterance"))
            if not text:
                continue
            speaker = _first_str(turn, ("speaker", "role", "name")) or "Unknown"
            # Preserve dia_id and session_n in raw so the adapter can group by
            # session and downstream scorers can match against gold evidence.
            raw_with_session = dict(turn)
            raw_with_session.setdefault("session_n", session_n)
            raw_with_session.setdefault("session_date_time", session_dt)
            turns.append(
                TurnRecord(
                    index=global_index,
                    speaker=speaker,
                    text=text,
                    timestamp=session_dt,
                    raw=raw_with_session,
                )
            )
            global_index += 1
    return turns


def _flatten_generic_turns(item: dict[str, Any]) -> list[TurnRecord]:
    """Permissive fallback for simpler/test-shaped JSON."""
    raw_turns: list[Any] | None = None
    for key in ("turns", "conversation", "messages", "dialogue"):
        value = item.get(key)
        if isinstance(value, list):
            raw_turns = value
            break
    if raw_turns is None:
        return []

    turns: list[TurnRecord] = []
    for index, turn in enumerate(raw_turns):
        if not isinstance(turn, dict):
            continue
        text = _first_str(turn, ("text", "content", "message", "utterance", "value"))
        if not text:
            continue
        speaker = _first_str(turn, ("speaker", "role", "name", "participant")) or f"speaker_{index}"
        timestamp = _first_str(turn, ("timestamp", "time", "datetime", "date"))
        turns.append(
            TurnRecord(
                index=index,
                speaker=speaker,
                text=text,
                timestamp=timestamp,
                raw=turn,
            )
        )
    return turns


# ----------------------------------------------------------------------- QA


def _normalize_qa_items(item: dict[str, Any], *, conversation_id: str) -> list[QARecord]:
    raw_items: list[Any] | None = None
    for key in ("qa", "qas", "questions"):
        value = item.get(key)
        if isinstance(value, list):
            raw_items = value
            break
    if raw_items is None:
        return []

    records: list[QARecord] = []
    for index, qa in enumerate(raw_items):
        if not isinstance(qa, dict):
            continue
        question = _first_str(qa, ("question", "query", "prompt"))
        if not question:
            continue
        question_id = str(qa.get("question_id") or qa.get("id") or f"{conversation_id}_q{index}")
        # LoCoMo adversarial answers come through as None (the correct
        # behavior is "I don't know"); preserve that explicitly.
        expected_answer = qa.get("answer")
        if expected_answer is None:
            expected_answer = qa.get("expected_answer") or qa.get("gold_answer") or qa.get("target")
        if isinstance(expected_answer, str):
            expected_answer = expected_answer.strip() or None
        records.append(
            QARecord(
                question_id=question_id,
                question=question,
                expected_answer=expected_answer,
                raw=qa,
            )
        )
    return records


# --------------------------------------------------------------------- utility


def _first_str(item: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
