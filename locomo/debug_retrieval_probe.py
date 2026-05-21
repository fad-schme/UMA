from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace

from locomo.adapter import LocomoUMAAdapter
from locomo.loader import load_locomo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact retrieval probes against a truncated LoCoMo conversation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--query", action="append", required=True)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    conversation = next(
        item for item in load_locomo_dataset(args.dataset) if item.conversation_id == args.conversation_id
    )
    if args.max_turns is not None:
        conversation = replace(conversation, turns=conversation.turns[: max(args.max_turns, 0)])

    adapter = LocomoUMAAdapter(args.config, disable_llm=True)
    try:
        warnings = await adapter.ingest_conversation(conversation)
        print("ingest_warnings:", warnings)
        user_id = f"locomo_user_{conversation.conversation_id}"
        session_id = f"locomo_{conversation.conversation_id}"

        for query in args.query:
            mem = await adapter.memory.retrieve_memory(
                query_text=query,
                user_id=user_id,
                session_id=session_id,
                include_debug=True,
            )
            ctx = await adapter.memory.retrieve_context(
                query_text=query,
                user_id=user_id,
                session_id=session_id,
            )
            debug = mem.get("debug") or {}
            debug_memories = debug.get("memories") or []
            print(f"QUERY: {query}")
            print(
                "  memory facts=",
                len(mem.get("facts") or []),
                "evidence=",
                len(mem.get("evidence") or []),
                "debug_memories=",
                len(debug_memories),
            )
            if mem.get("facts"):
                print("  fact_texts:", [item.get("text") for item in (mem.get("facts") or [])[:3]])
            if mem.get("evidence"):
                print("  evidence_texts:", [item.get("text") for item in (mem.get("evidence") or [])[:2]])
            if debug_memories:
                first = debug_memories[0]
                print("  debug_memory_summary:", first.get("summary"))
                print("  debug_memory_text:", first.get("text"))
            episodic = ctx.get("episodic") or []
            print(
                "  context episodic=",
                len(episodic),
                "facts=",
                len(ctx.get("facts") or []),
                "chunks=",
                len(ctx.get("chunks") or []),
            )
            if episodic:
                summaries = []
                for item in episodic[:2]:
                    if isinstance(item, dict):
                        summaries.append(item.get("summary"))
                    else:
                        summaries.append(getattr(item, "summary", None))
                print("  episodic_summaries:", summaries)
            rendered = (ctx.get("working_memory") or [])
            if rendered:
                print("  working_memory_messages:", len(rendered))
            print()
    finally:
        adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
