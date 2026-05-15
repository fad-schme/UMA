from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
from typing import Any

from uma import UMAMemory


def _set_memory_context(memory: UMAMemory, config_path: str, user_id: str, agent_id: str) -> UMAMemory:
    params = inspect.signature(memory.set_context).parameters
    if "user_id" in params:
        return memory.set_context(
            user_id=user_id,
            agent_id=agent_id,
            tenant_id="default",
            request_id=f"memory:{user_id}",
            session_id=f"memory:{user_id}",
        )
    return memory.set_context(agent_id=agent_id)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _extract_memory_texts(result: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    candidate_keys = ("text", "summary", "content", "snippet", "body")

    def add_text(value: Any) -> None:
        if not isinstance(value, str):
            return
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        texts.append(normalized)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key in candidate_keys:
                add_text(node.get(key))
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(result.get("memories") or [])
    visit(result.get("compiled_answer") or {})
    visit(result.get("evidence") or [])
    visit(result.get("supporting_evidence") or [])
    return texts


def _render_memory_result(result: dict[str, Any]) -> str:
    memory_texts = _extract_memory_texts(result)
    lines = ["", "==================== Retrieved Memory Texts ===================="]
    if memory_texts:
        for index, text in enumerate(memory_texts, start=1):
            lines.append(f"{index}. {text}")
    else:
        lines.append("No memory texts found in the retrieval result.")

    raw_result = json.dumps(_to_jsonable(result), indent=2, ensure_ascii=True, sort_keys=True)
    lines.extend(
        [
            "",
            "==================== Raw Retrieval Result ====================",
            raw_result,
            "============================================================",
            "",
        ]
    )
    return "\n".join(lines)


async def interactive_memory(
    load_bootstrap: bool = False,
) -> None:
    config_path = "config/uma.yaml"
    user_id = "user:local"
    agent_id = "agent-default"

    memory = _set_memory_context(
        UMAMemory.from_yaml(config_path),
        config_path,
        user_id,
        agent_id,
    )

    session_id = f"memory:{user_id}"
    memory_md = "examples/MEMORY.md"
    user_md = "examples/USER.md"
    soul_md = "examples/SOUL.md"
    diary_md = "examples/DAILY_DIARY.md"

    async def load_files() -> None:

        await memory.load_memory_bootstrap(
            memory_md,
            user_id=user_id,
        )
        await memory.load_daily_diary_bootstrap(
            diary_md,
            user_id=user_id,
        )

        memory.load_userprofile(user_md)
        memory.load_agentprofile(soul_md)

    if load_bootstrap:
        await load_files()

    try:
        print("UMA memory example ready. Commands: /load, /quit")
        while True:
            try:
                user_query = input("ask memory > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not user_query:
                continue
            if user_query.lower().startswith("/q"):
                break
            if user_query.lower().startswith("/load"):
                await load_files()
                print("Bootstrap load complete.")
                continue

            result = await memory.retrieve_memory(
                query_text=user_query,
                user_id=user_id,
            )

            print(_render_memory_result(result))
    finally:
        memory.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Example UMA memory retrieval loop")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    if not os.path.exists("config/uma.yaml"):
        raise SystemExit("Config file not found: config/uma.yaml")

    asyncio.run(
        interactive_memory(
            load_bootstrap=args.load,
        )
    )


if __name__ == "__main__":
    main()
