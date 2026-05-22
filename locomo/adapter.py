"""LoCoMo adapter for UMA-RLM.

Design
------
LoCoMo conversations are between two peers (e.g. Caroline & Melanie). UMA's
tenant model expects a single user, so each conversation gets one synthetic
user that owns the whole dialogue:

    user_id    = f"locomo_user_{conversation_id}"
    session_id = f"locomo_{conversation_id}"

Subject attribution is handled in the *text itself* via speaker prefixes,
not by ownership. Every turn carries its speaker as the first token after
the timestamp:

    "[<timestamp>] <Speaker>: <text>"

The speaker name is part of the indexed text, so retrieval ("What did
Caroline say about X") hits the right turns and any fact extractor sees
the subject up front. dia_id and session_n are also passed in extra_meta
so a future scorer can map retrieved chunks back to LoCoMo gold-evidence
pointers without parsing strings.

Ingest pairing (sliding window)
-------------------------------
LoCoMo is peer-to-peer, not user→assistant. Pairing adjacent turns is a
pragmatic compromise that keeps assistant_reply non-empty (avoiding any
empty-string guards in the derive stage) and gives working memory a
coherent transcript shape. Each utterance appears exactly once as
user_msg — that determines long-term storage. It also appears once as
assistant_reply in the previous pair, which feeds working memory only.

    turn 1: user=Caroline_1, assistant=Melanie_1
    turn 2: user=Melanie_1,  assistant=Caroline_2
    turn 3: user=Caroline_2, assistant=Melanie_2
    ...
    turn N: user=last_turn,  assistant=""   (no successor)

UMA is a Python package, not an agent. This adapter does not generate
answers; it returns the raw retrieve_memory result for an external scorer.
"""

from __future__ import annotations

import time
from typing import Any

from uma import UMAMemory

from locomo.loader import ConversationRecord, QARecord, TurnRecord


AGENT_ID = "locomo_agent"


def user_id_for(conversation_id: str) -> str:
    return f"locomo_user_{conversation_id}"


def session_id_for(conversation_id: str) -> str:
    return f"locomo_{conversation_id}"


class LocomoUMAAdapter:
    """Drives UMAMemory for LoCoMo benchmark runs."""

    def __init__(self, config_path: str) -> None:
        self.memory = UMAMemory.from_yaml(config_path).set_context(agent_id=AGENT_ID)

    def close(self) -> None:
        self.memory.shutdown()

    # ------------------------------------------------------------------ ingest

    async def ingest_conversation(self, conversation: ConversationRecord) -> list[str]:
        """Ingest turns via sliding-window pairs through process_turn.

        Each turn is the user_msg of exactly one process_turn call. The next
        turn (if any) is its assistant_reply. Every utterance therefore hits
        long-term storage exactly once via the user_msg path.
        """
        user_id = user_id_for(conversation.conversation_id)
        session_id = session_id_for(conversation.conversation_id)
        turns = [t for t in conversation.turns if (t.text or "").strip()]
        warnings: list[str] = []

        for i, turn in enumerate(turns):
            next_turn = turns[i + 1] if i + 1 < len(turns) else None
            user_msg = _format_turn(turn)
            assistant_reply = _format_turn(next_turn) if next_turn is not None else ""
            extra_meta = {
                "locomo": {
                    "conversation_id": conversation.conversation_id,
                    "dia_id": turn.raw.get("dia_id"),
                    "session_n": turn.raw.get("session_n"),
                    "speaker": turn.speaker,
                    "timestamp": turn.timestamp,
                    "turn_index": turn.index,
                    "paired_with_dia_id": next_turn.raw.get("dia_id") if next_turn else None,
                }
            }
            try:
                await self.memory.process_turn(
                    user_id=user_id,
                    user_msg=user_msg,
                    assistant_reply=assistant_reply,
                    session_id=session_id,
                    extra_meta=extra_meta,
                )
                print(f"process_turn: {i}")

            except Exception as exc:
                warnings.append(
                    f"ingest failed conversation_id={conversation.conversation_id} "
                    f"turn_index={turn.index} dia_id={turn.raw.get('dia_id')!r} "
                    f"error={type(exc).__name__}: {exc}"
                )
        return warnings

    # ----------------------------------------------------------------- retrieve

    async def run_question(
        self,
        conversation: ConversationRecord,
        qa: QARecord,
    ) -> dict[str, Any]:
        """Retrieve memory for one QA item. No answer generation."""
        user_id = user_id_for(conversation.conversation_id)
        session_id = session_id_for(conversation.conversation_id)
        record: dict[str, Any] = {
            "conversation_id": conversation.conversation_id,
            "question_id": qa.question_id,
            "question": qa.question,
            "expected_answer": qa.expected_answer,
            "memory_result": None,
            "latency_ms": {},
            "error": None,
        }

        try:
            t0 = time.perf_counter()
            memory_result = await self.memory.retrieve_memory(
                query_text=qa.question,
                user_id=user_id,
                session_id=session_id,
                include_debug=True,
            )
            record["latency_ms"]["retrieve_memory"] = round((time.perf_counter() - t0) * 1000, 3)
            record["memory_result"] = _jsonable(memory_result)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record


# ---------------------------------------------------------------------- helpers


def _format_turn(turn: TurnRecord) -> str:
    """Render a single turn as 'speaker: text', with an optional timestamp."""
    speaker = turn.speaker or "Unknown"
    text = (turn.text or "").strip()
    if turn.timestamp:
        return f"[{turn.timestamp}] {speaker}: {text}"
    return f"{speaker}: {text}"


def _jsonable(value: Any) -> Any:
    """Shallow coercion to JSON-safe types. No field renaming, no stripping."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)
