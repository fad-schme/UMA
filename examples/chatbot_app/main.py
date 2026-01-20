from __future__ import annotations

import asyncio
import logging
from typing import Optional

from uma.core.uma3_memory import UMAMemory
from uma.core.pipeline import MemoryPipeline
from uma.core.utils.identity import ensure_user_subject

from .loader import load_documents_folder

logging.basicConfig(level=logging.INFO)


SYSTEM_PROMPT_DEFAULT = (
    "You are an assistant that answers questions using available memory. "
    "Be concise and cite memory where helpful."
)


async def interactive_chat(
    config_path: str = "config/uma3.yaml",
    user_id: str = "user:local",
    system_prompt: Optional[str] = None,
):
    system_prompt = system_prompt or SYSTEM_PROMPT_DEFAULT
    user_subject = ensure_user_subject(user_id)

    # Initialize UMA memory runtime
    memory = UMAMemory.from_yaml(config_path)
    memory.initialize()

    # Pipeline for turn processing
    pipeline = MemoryPipeline(memory_client=memory, hooks=memory.hooks)

    print("UMA-RLM chatbot ready. Commands: /load <folder> <topic>, /setprompt, /quit")

    while True:
        try:
            user = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user:
            continue

        if user.lower().startswith("/quit"):
            break

        if user.lower().startswith("/setprompt"):
            print("Enter new system prompt (empty to cancel):")
            p = input().strip()
            if p:
                system_prompt = p
                print("System prompt updated.")
            continue

        if user.lower().startswith("/load "):
            parts = user.split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: /load <folder> <topic>")
                continue
            folder, topic = parts[1], parts[2]
            print(f"Loading documents from {folder} as topic '{topic}'...")
            n = await load_documents_folder(folder, topic, memory, user_subject)
            print(f"Ingested {n} documents into semantic memory.")
            continue

        # Normal chat: build context and call LLM
        try:
            ctx = await memory.get_user_context(user_id=user_subject, query_text=user)
            pack = await memory.build_context_pack(user_id=user_subject, query_text=user)
            memory_snippet = _format_context_pack(pack)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user}\n\nRelevant memory:\n{memory_snippet}"},
            ]

            reply = await memory.llm.generate(messages=messages, max_tokens=256)
            print("Assistant>", reply)

            # Update UMA memory with the turn
            await pipeline.process_turn(user_id=user_subject, user_msg=user, assistant_reply=reply)

        except Exception as exc:
            logging.exception("Chat turn failed: %s", exc)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Example UMA-RLM interactive chatbot")
    parser.add_argument("--config", default="config/uma3.yaml")
    parser.add_argument("--user", default="user:local")
    parser.add_argument("--system-prompt", default=None)
    args = parser.parse_args()

    asyncio.run(interactive_chat(config_path=args.config, user_id=args.user, system_prompt=args.system_prompt))


def _format_context_pack(pack: dict) -> str:
    """Render UMA-RLM context pack into a compact, readable prompt snippet."""
    lines = []

    wm = pack.get("working_memory", [])
    if wm:
        lines.append("Working memory:")
        for msg in wm[-8:]:
            role = msg.get("role")
            text = (msg.get("text") or "").strip()
            if text:
                lines.append(f"- {role}: {text}")

    episodic = pack.get("episodic", [])
    if episodic:
        lines.append("\nEpisodic:")
        for ep in episodic[:5]:
            summary = (ep.get("summary") or "").strip()
            if summary:
                lines.append(f"- {summary}")

    semantic = pack.get("semantic", [])
    if semantic:
        lines.append("\nSemantic facts:")
        for fact in semantic[:8]:
            subject = fact.get("subject", "unknown")
            predicate = fact.get("predicate", "related_to")
            obj = fact.get("object")
            if isinstance(obj, dict):
                title = obj.get("title") or obj.get("text") or str(obj)
                snippet = (obj.get("text") or "").strip()
                snippet = snippet[:400] if snippet else ""
                lines.append(f"- {subject} {predicate} {title}")
                if snippet:
                    lines.append(f"  excerpt: {snippet}")
            else:
                lines.append(f"- {subject} {predicate} {obj}")

    procedural = pack.get("procedural", [])
    if procedural:
        lines.append("\nProcedural skills:")
        for skill in procedural[:5]:
            name = skill.get("name") or "Unnamed"
            desc = (skill.get("description") or "").strip()
            lines.append(f"- {name}: {desc}")

    graph = pack.get("graph", [])
    if graph:
        lines.append("\nGraph:")
        for node in graph[:5]:
            lines.append(f"- {node}")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    main()
