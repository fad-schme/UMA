"""
Simulated conversation covering both GitHub manual (ingested document)
and Anna's existing memory/diary data.
"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from uma import UMAMemory
from uma.retrieve import ContextPackBuilder

QUESTIONS = [
    # GitHub manual — new ingested document
    "How do I create a new branch in GitHub?",
    "What is a pull request and how do I open one?",
    "How does the GitHub fork workflow work?",
    # Anna profile — existing memory data
    "What is Anna's availability window for scheduling meetings?",
    "What projects is Anna currently working on?",
    # Cross-source — requires both GitHub knowledge and Anna's code review style
    "Anna is reviewing a PR. What does she look for in code reviews?",
    # History / episodic
    "What did Anna work on last week in the diary?",
]


async def run() -> None:
    config_path = "config/uma.yaml"
    user_id = "user:local"
    agent_id = "agent-default"
    session_id = "chat:sim"

    memory = UMAMemory.from_yaml(config_path).set_context(agent_id=agent_id)
    ctx_cfg = getattr(getattr(memory, "retrieval_cfg", None), "context", None)
    llm = getattr(memory, "agent_llm", None) or memory.llm

    sep = "=" * 70

    try:
        for i, question in enumerate(QUESTIONS, 1):
            print(f"\n{sep}")
            print(f"Q{i:02d}: {question}")
            print("-" * 70)

            context = await memory.retrieve_context(query_text=question, user_id=user_id)
            pack = ContextPackBuilder.build(question, context)
            snippet = await ContextPackBuilder.render_snippet_async(
                pack, context_cfg=ctx_cfg, llm=llm
            )

            print("--- memory context ---")
            print(snippet if snippet.strip() else "(no context retrieved)")
            print("--- end context ---\n")

            if snippet.strip():
                user_content = f"Context:\n{snippet}\n\nQuestion: {question}"
            else:
                user_content = question

            reply = await llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Answer concisely using the provided context. "
                            "If the context does not contain enough information, say so clearly."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                max_tokens=256,
                temperature=0.2,
            )
            print(f"Assistant: {reply}")

            await memory.process_turn(
                user_id=user_id,
                user_msg=question,
                assistant_reply=reply,
                session_id=session_id,
            )
    finally:
        memory.shutdown()


if __name__ == "__main__":
    if not os.path.exists("config/uma.yaml"):
        sys.exit("Run from the repo root: python examples/chatbot_app/sim.py")
    asyncio.run(run())
