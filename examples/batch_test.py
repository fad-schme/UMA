"""Batch retrieval smoke test.

Runs a fixed question set against a loaded UMA store and prints what each query
retrieves. Useful for eyeballing recall after a bootstrap or an ingest.

Load the store first with `python examples/memory_app/main.py --config <path> --load`,
then:

    python examples/batch_test.py --config path/to/uma.yaml
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from uma import ContextBundle, UMAMemory

USER_ID = "user:local"
AGENT_ID = "agent-default"
SESSION_ID = "batch-test"

QUESTIONS = [
    # Project status
    "What is the status of the current project and when is the v1.0 launch targeted?",
    "What tech stack is being used?",
    "What are the current blockers or risks?",
    "What milestones have been completed so far?",
    # Daily diary
    "What happened on May 10 in the daily diary?",
    "What did Anna work on during the week of May 12-14?",
    "Were there any production issues or incidents logged in the diary?",
    "What were the main tasks completed in the diary on May 13?",
    # Profile and preferences
    "What is Anna's core availability window and when should I avoid scheduling meetings?",
    "Are there any dietary or personal restrictions to know about when planning team activities?",
    "Who does Anna work closely with and what is her reporting structure?",
    "How does Anna typically evaluate architectural proposals?",
    # Communication and work style
    "How should I format documentation and code examples for Anna?",
    "What does Anna look for when reviewing code and what is her approval workflow?",
    "If there is a production issue, how should I communicate the problem to Anna?",
    # Long-term memory
    "When is Anna on vacation and what communication should be avoided during that time?",
    "What are Anna's long-term career goals or aspirations?",
    "What recurring meetings or commitments does Anna have each week?",
    # Cross-file
    "What engineering principles does Anna follow when making technical decisions?",
    "Summarize Anna's current projects and professional priorities.",
]


def summarize(bundle: ContextBundle) -> str:
    """Print the retrieved lanes of a ContextBundle.

    Read the bundle by attribute — it is a Pydantic model. `facts` are `Fact`
    domain objects (subject-predicate-object), `chunks` and `episodic` are
    `Chunk` and `Episode`.
    """
    lines = [
        f"  lanes: {len(bundle.facts)} facts | {len(bundle.chunks)} chunks | "
        f"{len(bundle.episodic)} episodic | {len(bundle.skills)} skills"
    ]

    for fact in bundle.facts:
        triple = " ".join(
            part for part in (fact.subject, fact.predicate, fact.object) if part
        )
        lines.append(f"    fact:  {triple}")

    for chunk in bundle.chunks:
        text = " ".join(str(chunk.text).split())
        lines.append(f"    chunk: {text[:160]}")

    for episode in bundle.episodic:
        summary = " ".join(str(getattr(episode, "summary", "")).split())
        if summary:
            lines.append(f"    epi:   {summary[:160]}")

    if len(lines) == 1:
        lines.append("    (nothing retrieved)")
    return "\n".join(lines)


async def run(config_path: str) -> None:
    memory = UMAMemory.from_yaml(config_path)

    empty = 0
    try:
        for index, question in enumerate(QUESTIONS, start=1):
            print(f"\n{'=' * 70}")
            print(f"Q{index:02d}: {question}")
            print("-" * 70)
            bundle = await memory.retrieve_context(
                query_text=question,
                user_id=USER_ID,
                session_id=SESSION_ID,
                agent_id=AGENT_ID,
            )
            if not (bundle.facts or bundle.chunks or bundle.episodic):
                empty += 1
            print(summarize(bundle))

        print(f"\n{'=' * 70}")
        print(f"Queries returning nothing: {empty}/{len(QUESTIONS)}")
    finally:
        memory.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to your uma.yaml.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config file not found: {args.config}")

    asyncio.run(run(args.config))


if __name__ == "__main__":
    main()
