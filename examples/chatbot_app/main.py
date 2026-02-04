from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from uma.core.uma_memory import UMAMemory
from uma.adapters.llm.base import LLMInterface
from uma.core.ingest.parser import FileContentParser
logger = logging.getLogger(__name__)


SYSTEM_PROMPT_DEFAULT = (
    "You are an assistant that answers questions using available memory. "
    "Be concise and cite memory where helpful."
)


async def agent_generate(messages: list, llm: Optional[LLMInterface] = None) -> str:
    """
    Generate a response using UMA's configured LLM.
    """
    if llm is None:
        raise RuntimeError("No LLM configured; set llms.agent in config/uma.yaml.")
    reply = await llm.generate(messages=messages, max_tokens=128, temperature=0.2)
    if not isinstance(reply, str) or not reply.strip():
        logger.warning("agent_generate: LLM returned empty reply.")
    return reply


async def interactive_chat(
    config_path: str = "config/uma.yaml",
    user_id: str = "user:local",
    agent_id: str = "agent-default",
    system_prompt: Optional[str] = None,
    auto_load_material: bool = False,
):
    system_prompt = system_prompt or SYSTEM_PROMPT_DEFAULT

    # Initialize UMA memory runtime
    memory = UMAMemory.from_yaml(config_path)
    memory.agent_id = agent_id
    memory._lazy_init()
    try:
        vector_backend = getattr(memory.raw_config.storage, "vector_backend", "")
        if vector_backend in ("faiss", "inmemory"):
            logging.info("Rebuilding vector indexes from SQL for user=%s", user_id)
            try:
                await memory.rebuild_vector_indexes(user_id=user_id)
            except Exception:
                logging.exception("Vector index rebuild failed; continuing with empty index.")

        # Pipeline for turn processing
        print("UMA-RLM chatbot ready. Commands: /load, /setprompt, /quit")

        config_dir = os.path.dirname(os.path.abspath(config_path))
        project_root = os.path.dirname(config_dir)
        material_dir = os.path.join(project_root, "material")

        async def _load_material() -> int:
            if not os.path.isdir(material_dir):
                logger.warning("Material folder not found: %s", material_dir)
                return 0
            parser = FileContentParser()
            supported = set(parser.supported_ext())
            count = 0
            for root, _, filenames in os.walk(material_dir):
                for fn in filenames:
                    path = os.path.join(root, fn)
                    ext = os.path.splitext(path)[1].lower()
                    if ext not in supported:
                        continue
                    try:
                        print(f"Ingesting {fn} ...")
                        await memory.ingest_document(
                            path,
                            owner_type="agent",
                            owner_id=agent_id,
                        )
                        count += 1
                    except Exception:
                        logger.exception("Failed to ingest %s", path)
                        continue
            return count

        if auto_load_material:
            print(f"Loading documents from {material_dir} ...")
            n = await _load_material()
            print(f"Ingested {n} documents from /material.")

        while True:
            try:
                user = input("You> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not user:
                continue

            if user.lower().startswith("/q"):
                break

            if user.lower().startswith("/load"):
                print(f"Loading documents from {material_dir} ...")
                n = await _load_material()
                print(f"Ingested {n} documents from /material.")
                continue

            # Normal chat: retrieve context only; agent behavior is developer-owned
            try:
                user_message = user
                # One-liner to get a rendered snippet
                snippet = await memory.build_context_snippet_for_query(
                    user_id=user_id, query_text=user_message
                )
                if not snippet:
                    context_messages = [{"role": "user", "content": user_message}]
                    reply = (
                        "I don't have relevant memory stored for that question. "
                        "Please load documents or provide more context."
                    )
                else:
                    user_content = f"{user_message}\n\nRelevant memory:\n{snippet}"
                    context_messages = [{"role": "user", "content": user_content}]

                    print("**************************** Context snippet:")
                    print(snippet)
                    print("**************************** End of snippet")

                reply = await agent_generate(
                    messages=[{"role": "system", "content": system_prompt}] + context_messages,
                    llm=getattr(memory, "agent_llm", None) or memory.llm,
                )
                if not isinstance(reply, str) or not reply.strip():
                    reply = "I don't have enough information to answer that yet."

                print("Assistant>", reply)
                # Update UMA memory with the turn
                await memory.process_turn(
                    user_id=user_id, user_msg=user_message, assistant_reply=reply
                )

            except Exception as exc:
                logging.exception("Chat turn failed: %s", exc)
    finally:
        # Ensure graph driver (and other resources) are closed to avoid driver warnings.
        try:
            memory.shutdown()
        except Exception:
            logger.exception("Failed to shut down UMA memory.")


def main():
    import argparse
    import os
    import shutil

    parser = argparse.ArgumentParser(description="Example UMA-RLM interactive chatbot")
    parser.add_argument("--config", default="config/uma.yaml")
    parser.add_argument("--user", default="user:local")
    parser.add_argument("--agent", default="agent-default")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Delete all UMA SQL stores under storage.db_root and exit.",
    )
    args = parser.parse_args()

    if args.clear_all:
        cfg_path = os.path.abspath(args.config)
        cfg_dir = os.path.dirname(cfg_path)
        project_root = os.path.dirname(cfg_dir)  # sibling of config/
        abs_root = os.path.join(project_root, "data")
        if abs_root in {"/", ""}:
            raise RuntimeError(f"Refusing to clear unsafe db_root path: {abs_root}")
        if os.path.exists(abs_root):
            shutil.rmtree(abs_root)
        os.makedirs(abs_root, exist_ok=True)
        print(f"Cleared UMA storage at {abs_root}")
        asyncio.run(
            interactive_chat(
                config_path=args.config,
                user_id=args.user,
                agent_id=args.agent,
                system_prompt=args.system_prompt,
                auto_load_material=True,
            )
        )
        return

    asyncio.run(
        interactive_chat(
            config_path=args.config,
            user_id=args.user,
            agent_id=args.agent,
            system_prompt=args.system_prompt,
            auto_load_material=False,
        )
    )


if __name__ == "__main__":
    main()
