from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomo.adapter import LocomoUMAAdapter
from locomo.loader import ConversationRecord, load_locomo_dataset
from locomo.writer import JsonlWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal LoCoMo benchmark harness against UMA.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on total QA records written across all conversations.")
    parser.add_argument("--conversation-id", default=None,
                        help="If set, only run this conversation (sample_id).")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Truncate each conversation to its first N turns before ingest.")
    parser.add_argument("--max-qa-per-conversation", type=int, default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    conversations = _filter_conversations(
        load_locomo_dataset(args.dataset),
        conversation_id=args.conversation_id,
    )
    if args.max_conversations is not None:
        conversations = conversations[: max(args.max_conversations, 0)]
    if not conversations:
        raise SystemExit("No conversations matched the requested dataset filter.")

    adapter = LocomoUMAAdapter(args.config)
    remaining = args.limit
    written = 0

    try:
        with JsonlWriter(args.output) as writer:
            for conversation in conversations:
                if remaining is not None and remaining <= 0:
                    break
                selected = _slice_conversation(
                    conversation,
                    max_turns=args.max_turns,
                    max_qa_per_conversation=args.max_qa_per_conversation,
                )
                print(f"conversation_id={selected.conversation_id}")
                print(f"  ingest_turns={len(selected.turns)}/{len(conversation.turns)}")
                print(f"  qa_items={len(selected.qa_items)}/{len(conversation.qa_items)}")

                warnings = await adapter.ingest_conversation(selected)
                for warning in warnings:
                    print(f"  warn: {warning}")

                for qa in selected.qa_items:
                    if remaining is not None and remaining <= 0:
                        break
                    record = await adapter.run_question(selected, qa)
                    if warnings:
                        record.setdefault("warnings", []).extend(warnings)
                    writer.write(record)
                    written += 1
                    print(
                        f"  wrote_record={written} "
                        f"question_id={qa.question_id}"
                    )
                    if remaining is not None:
                        remaining -= 1
    finally:
        adapter.close()

    print(f"Wrote {written} record(s) to {args.output}")


def _filter_conversations(
    conversations: list[ConversationRecord],
    *,
    conversation_id: str | None,
) -> list[ConversationRecord]:
    if conversation_id is None:
        return conversations
    return [item for item in conversations if item.conversation_id == conversation_id]


def _slice_conversation(
    conversation: ConversationRecord,
    *,
    max_turns: int | None,
    max_qa_per_conversation: int | None,
) -> ConversationRecord:
    turns = conversation.turns
    qa_items = conversation.qa_items
    if max_turns is not None:
        turns = turns[: max(max_turns, 0)]
    if max_qa_per_conversation is not None:
        qa_items = qa_items[: max(max_qa_per_conversation, 0)]
    return replace(conversation, turns=turns, qa_items=qa_items)


if __name__ == "__main__":
    asyncio.run(main())
