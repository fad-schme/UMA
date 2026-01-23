from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from uma.core.uma_memory import UMAMemory
from uma.core.utils.identity import ensure_user_subject
from uma.adapters.llm.base import LLMInterface

from .loader import load_documents_folder

logging.basicConfig(level=logging.INFO)


SYSTEM_PROMPT_DEFAULT = (
    "You are an assistant that answers questions using available memory. "
    "Be concise and cite memory where helpful."
)


async def agent_generate(messages: list, llm: Optional[LLMInterface] = None) -> str:
    """
    Generate a response using UMA's configured LLM.
    """
    if llm is None:
        raise RuntimeError("No LLM configured; set llm.provider in config/uma.yaml.")
    return await llm.generate(messages=messages, max_tokens=128, temperature=0.2)


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
    vector_backend = getattr(memory.raw_config.storage, "vector_backend", "")
    if vector_backend in ("faiss", "inmemory"):
        logging.info("Rebuilding vector indexes from SQL for user=%s", user_subject)
        try:
            await memory.rebuild_vector_indexes(user_id=user_subject)
        except Exception:
            logging.exception("Vector index rebuild failed; continuing with empty index.")

    # Pipeline for turn processing
    print("UMA-RLM chatbot ready. Commands: /load <folder> <topic>, /setprompt, /quit")

    config_dir = os.path.dirname(os.path.abspath(config_path))

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
            resolved = folder
            if not os.path.isabs(resolved) and not os.path.exists(resolved):
                alt = os.path.join(config_dir, resolved)
                if os.path.exists(alt):
                    resolved = alt
            print(f"Loading documents from {resolved} as topic '{topic}'...")
            n = await load_documents_folder(resolved, topic, memory, user_subject)
            print(f"Ingested {n} documents into semantic memory.")
            continue

        # Normal chat: retrieve context only; agent behavior is developer-owned
        try:
            user_message = user
            pack = await memory.build_context_pack(user_id=user_subject, query_text=user_message)
            from uma.core.utils.context_pack_builder import ContextPackBuilder

            snippet = ContextPackBuilder.render_snippet(pack)
            if not snippet:
                context_messages = [{"role": "user", "content": user_message}]
                reply = (
                    "I don't have relevant memory stored for that question. "
                    "Please load documents or provide more context."
                )
            else:
                user_content = f"{user_message}\n\nRelevant memory:\n{snippet}"
                context_messages = [{"role": "user", "content": user_content}]
                reply = await agent_generate(
                    messages=[{"role": "system", "content": system_prompt}] + context_messages,
                    llm=memory.llm,
                )

            print("***** context:", context_messages)
            print("***** context/")
            print("Assistant>", reply)

            # Update UMA memory with the turn
            await memory.process_turn(user_id=user_subject, user_msg=user_message, assistant_reply=reply)

        except Exception as exc:
            logging.exception("Chat turn failed: %s", exc)


def main():
    import argparse
    import os
    import shutil

    parser = argparse.ArgumentParser(description="Example UMA-RLM interactive chatbot")
    parser.add_argument("--config", default="config/uma.yaml")
    parser.add_argument("--user", default="user:local")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Delete all UMA SQL stores under storage.db_root and exit.",
    )
    args = parser.parse_args()

    if args.clear_all:
        from uma.core.memory_config import UMAConfig

        cfg = UMAConfig.load_yaml(args.config)
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        db_root = cfg.storage.db_root
        abs_root = os.path.abspath(db_root if os.path.isabs(db_root) else os.path.join(cfg_dir, db_root))
        if abs_root in {"/", ""}:
            raise RuntimeError(f"Refusing to clear unsafe db_root path: {abs_root}")
        if os.path.exists(abs_root):
            shutil.rmtree(abs_root)
        os.makedirs(abs_root, exist_ok=True)
        print(f"Cleared UMA storage at {abs_root}")
        return

    asyncio.run(interactive_chat(config_path=args.config, user_id=args.user, system_prompt=args.system_prompt))


if __name__ == "__main__":
    main()
