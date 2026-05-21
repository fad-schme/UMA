from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TurnRecord:
    index: int
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
    conversations = _extract_conversations(data)
    normalized: list[ConversationRecord] = []
    for index, item in enumerate(conversations):
        conversation = _normalize_conversation(item, index=index)
        if conversation.turns and conversation.qa_items:
            normalized.append(conversation)
    return normalized


def _extract_conversations(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        raise ValueError("LoCoMo dataset must be a JSON object or array.")
    for key in ("conversations", "dialogs", "dialogues", "data", "records"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if _looks_like_conversation(data):
        return [data]
    raise ValueError("Could not find conversation records in dataset JSON.")


def _looks_like_conversation(item: dict[str, Any]) -> bool:
    return any(isinstance(item.get(key), list) for key in ("turns", "conversation", "messages", "dialogue"))


def _normalize_conversation(item: dict[str, Any], *, index: int) -> ConversationRecord:
    conversation_id = str(
        item.get("conversation_id")
        or item.get("id")
        or item.get("dialog_id")
        or item.get("name")
        or f"conversation_{index}"
    )
    turns = _normalize_turns(item)
    qa_items = _normalize_qa_items(item, conversation_id=conversation_id)
    metadata = {
        key: value
        for key, value in item.items()
        if key not in {"turns", "conversation", "messages", "dialogue", "qa", "qas", "questions"}
    }
    return ConversationRecord(
        conversation_id=conversation_id,
        turns=turns,
        qa_items=qa_items,
        metadata=metadata,
    )


def _normalize_turns(item: dict[str, Any]) -> list[TurnRecord]:
    raw_turns = None
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


def _normalize_qa_items(item: dict[str, Any], *, conversation_id: str) -> list[QARecord]:
    raw_items = None
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
        expected_answer = _first_str(qa, ("answer", "expected_answer", "gold_answer", "target"))
        records.append(
            QARecord(
                question_id=question_id,
                question=question,
                expected_answer=expected_answer,
                raw=qa,
            )
        )
    return records


def _first_str(item: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
