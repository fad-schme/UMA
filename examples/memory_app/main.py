"""Interactive `retrieve_memory` loop.

Ask questions against UMA's compiled, evidence-backed memory. Optionally
bootstrap the store first from the markdown files in `examples/`.

Run from the repository root:

    python examples/memory_app/main.py --config path/to/uma.yaml
    python examples/memory_app/main.py --config path/to/uma.yaml --load
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from uma import MemoryResult, UMAMemory

USER_ID = "user:local"
AGENT_ID = "agent-default"
EXAMPLES = Path(__file__).resolve().parent.parent


def render(result: MemoryResult) -> str:
    """Format a MemoryResult for the terminal.

    `MemoryResult` is a Pydantic model — read it by attribute. `facts` and
    `evidence` are deliberately narrower dict projections of the domain
    types, so those are read by key.
    """
    lines: list[str] = []

    compiled = result.compiled_memory
    if compiled is not None:
        lines.append(f"intent: {compiled.memory_intent}")
    else:
        lines.append("intent: (evidence-only fallback - no compiled answer)")

    lines.append(f"provenance_valid: {result.provenance_valid}")
    if result.provenance_error:
        lines.append(f"provenance_error: {result.provenance_error}")

    lines.append("")
    lines.append(f"facts ({len(result.facts)})")
    for index, fact in enumerate(result.facts, start=1):
        text = fact.get("text", "")
        confidence = fact.get("confidence")
        suffix = f"  [confidence={confidence:.2f}]" if isinstance(confidence, float) else ""
        lines.append(f"  {index}. {text}{suffix}")
    if not result.facts:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"evidence ({len(result.evidence)})")
    for index, chunk in enumerate(result.evidence, start=1):
        text = " ".join(str(chunk.get("text", "")).split())
        source = chunk.get("source") or chunk.get("source_document_id") or "unknown"
        lines.append(f"  {index}. [{source}] {text[:200]}")
    if not result.evidence:
        lines.append("  (none)")

    if result.gaps:
        lines.append("")
        lines.append(f"gaps: {result.gaps}")

    return "\n".join(lines)


async def bootstrap(memory: UMAMemory) -> None:
    """Seed the store from the markdown files shipped alongside this example."""
    await memory.load_memory_bootstrap(str(EXAMPLES / "MEMORY.md"), user_id=USER_ID, agent_id=AGENT_ID)
    await memory.load_daily_diary_bootstrap(
        str(EXAMPLES / "DAILY_DIARY.md"), user_id=USER_ID,
        agent_id=AGENT_ID,
    )


async def run(config_path: str, load_bootstrap: bool) -> None:
    memory = UMAMemory.from_yaml(config_path)

    try:
        if load_bootstrap:
            await bootstrap(memory)
            print("Bootstrap load complete.")

        print("UMA memory example ready. Commands: /load, /quit")
        while True:
            try:
                query = input("ask memory > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not query:
                continue
            if query.lower().startswith("/q"):
                break
            if query.lower().startswith("/load"):
                await bootstrap(memory)
                print("Bootstrap load complete.")
                continue

            result = await memory.retrieve_memory(
                query_text=query,
                user_id=USER_ID,
                session_id=f"memory:{USER_ID}",
                agent_id=AGENT_ID,
            )
            print()
            print(render(result))
            print()
    finally:
        memory.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to your uma.yaml. UMA never ships one; you author it.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Bootstrap the store from examples/*.md before the first question.",
    )
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config file not found: {args.config}")

    asyncio.run(run(args.config, args.load))


if __name__ == "__main__":
    main()
