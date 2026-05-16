"""
Batch retrieval test — runs 20 questions against the loaded UMA memory and prints results.
Usage: python examples/batch_test.py
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys

from uma import UMAMemory
from uma.retrieve.context_pack_builder import ContextPackBuilder


QUESTIONS = [
    # Project Animus
    "What is the status of Project Animus and when is the v1.0 launch targeted?",
    "What tech stack is being used for Project Animus?",
    "What are the current blockers or risks for Project Animus?",
    "What milestones has Project Animus completed so far?",
    # Daily diary
    "What happened on May 10 in the daily diary?",
    "What did Anna work on during the week of May 12-14?",
    "Were there any production issues or incidents logged in the diary?",
    "What were the main tasks completed in the diary on May 13?",
    # Anna's profile and preferences
    "What is Anna's core availability window and when should I avoid scheduling meetings?",
    "Are there any dietary or personal restrictions I should know about when planning team activities?",
    "Who does Anna work closely with and what is her reporting structure?",
    "How does Anna typically evaluate architectural proposals?",
    # Communication and work style
    "How should I format documentation and code examples for Anna?",
    "What does Anna look for when reviewing code and what is her approval workflow?",
    "If there is a production issue, how should I communicate the problem to Anna?",
    # Memory / long-term
    "When is Anna on vacation and what communication should be avoided during that time?",
    "What are Anna's long-term career goals or aspirations?",
    "What recurring meetings or commitments does Anna have each week?",
    # Cross-file
    "What engineering principles does Anna follow when making technical decisions?",
    "Summarize Anna's current projects and professional priorities.",
]


def _set_context(memory: UMAMemory, user_id: str, agent_id: str) -> UMAMemory:
    params = inspect.signature(memory.set_context).parameters
    if "user_id" in params:
        return memory.set_context(
            user_id=user_id,
            agent_id=agent_id,
            tenant_id="default",
            request_id="batch-test",
            session_id="batch-test",
        )
    return memory.set_context(agent_id=agent_id)


async def run() -> None:
    config_path = "config/uma.yaml"
    user_id = "user:local"
    agent_id = "agent-default"

    memory = _set_context(UMAMemory.from_yaml(config_path), user_id, agent_id)

    try:
        for i, question in enumerate(QUESTIONS, start=1):
            print(f"\n{'='*70}")
            print(f"Q{i:02d}: {question}")
            print("-" * 70)
            try:
                ctx = await memory.retrieve_context(
                    query_text=question,
                    user_id=user_id,
                )
                pack = ContextPackBuilder.build(question, ctx)
                snippet = await ContextPackBuilder.render_snippet_async(
                    pack,
                    context_cfg=getattr(memory, "retrieval_cfg", None) and
                                 getattr(memory.retrieval_cfg, "context", None),
                    llm=getattr(memory, "llm", None),
                )
                print(snippet if snippet.strip() else "(no context retrieved)")
            except Exception as exc:
                print(f"ERROR: {exc}")
    finally:
        memory.shutdown()


if __name__ == "__main__":
    if not os.path.exists("config/uma.yaml"):
        sys.exit("Config not found: config/uma.yaml — run from the repo root")
    asyncio.run(run())
