from __future__ import annotations

import asyncio
import logging
from typing import Optional

from uma.core.uma_memory import UMAMemory
from uma.core.utils.identity import ensure_user_subject

from .loader import load_documents_folder

logging.basicConfig(level=logging.INFO)


SYSTEM_PROMPT_DEFAULT = (
    "You are an assistant that answers questions using available memory. "
    "Be concise and cite memory where helpful."
)


async def agent_generate(messages: list) -> str:
    """
    Placeholder for the developer's agent LLM call.
    """
    raise NotImplementedError("Integrate your agent LLM here.")


async def interactive_chat(
    config_path: str = "config/uma.yaml",
    user_id: str = "user:local",
    system_prompt: Optional[str] = None,
):
    system_prompt = system_prompt or SYSTEM_PROMPT_DEFAULT
    user_subject = ensure_user_subject(user_id)

    # Initialize UMA memory runtime
    memory = UMAMemory.from_yaml(config_path)
    memory.initialize()
    if memory.rlm_controller is None:
        logging.warning("RLM is disabled in config; enable retrieval.rlm.enabled for full UMA-RLM behavior.")

    # Pipeline for turn processing
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

        # Normal chat: retrieve context only; agent behavior is developer-owned
        try:
            user_message = user
            context_messages = await memory.build_prompt_messages(
                user_id=user_subject,
                query_text=user_message,
            )
            messages = [{"role": "system", "content": system_prompt}] + context_messages
            reply = await agent_generate(messages=messages)
            print("Assistant>", reply)

            # Update UMA memory with the turn
            await memory.process_turn(user_id=user_subject, user_msg=user_message, assistant_reply=reply)

        except Exception as exc:
            logging.exception("Chat turn failed: %s", exc)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Example UMA-RLM interactive chatbot")
    parser.add_argument("--config", default="config/uma.yaml")
    parser.add_argument("--user", default="user:local")
    parser.add_argument("--system-prompt", default=None)
    args = parser.parse_args()

    asyncio.run(interactive_chat(config_path=args.config, user_id=args.user, system_prompt=args.system_prompt))


if __name__ == "__main__":
    main()
