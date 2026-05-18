"""
10-round GitHub documentation chat evaluation.
Runs an interactive conversation against the ingested github-manual.pdf,
showing retrieved context and LLM reply per turn.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uma import UMAMemory
from uma.retrieve.context_pack_builder import ContextPackBuilder

QUESTIONS = [
    "How do I create a new branch in GitHub?",
    "What is a pull request and how do I open one?",
    "How do I fork a repository on GitHub?",
    "What is the difference between git clone and git fork?",
    "How do I resolve a merge conflict in GitHub?",
    "How do I review and approve a pull request?",
    "How do I revert a merged pull request?",
    "What are GitHub Actions and how do I create a workflow?",
    "How do I protect a branch from direct pushes?",
    "How do I tag a release and publish it on GitHub?",
]


async def run() -> None:
    config_path = "config/uma.yaml"
    user_id = "user:local"
    agent_id = "agent-default"
    session_id = "chat:github-eval"

    memory = UMAMemory.from_yaml(config_path).set_context(agent_id=agent_id)
    ctx_cfg = getattr(getattr(memory, "retrieval_cfg", None), "context", None)
    llm = getattr(memory, "agent_llm", None) or memory.llm
    sep = "=" * 72

    facts_hit = 0
    chunks_hit = 0

    try:
        for i, question in enumerate(QUESTIONS, 1):
            print(f"\n{sep}")
            print(f"Q{i:02d}: {question}")
            print("-" * 72)

            ctx = await memory.retrieve_context(query_text=question, user_id=user_id, session_id=session_id)

            n_facts = len(ctx.get("facts", []))
            n_chunks = len(ctx.get("chunks", []))
            n_episodic = len(ctx.get("episodic", []))
            if n_facts > 0:
                facts_hit += 1
            if n_chunks > 0:
                chunks_hit += 1

            print(f"  Retrieved: {n_facts} facts | {n_chunks} chunks | {n_episodic} episodic")

            pack = ContextPackBuilder.build(question, ctx)
            snippet = await ContextPackBuilder.render_snippet_async(
                pack, context_cfg=ctx_cfg, llm=llm
            )

            if snippet.strip():
                print(f"  Context snippet ({len(snippet)} chars):")
                for line in snippet.strip().splitlines()[:12]:
                    print(f"    {line}")
                if snippet.strip().count("\n") >= 12:
                    print("    [...]")
            else:
                print("  Context: (none)")

            user_content = f"Context:\n{snippet}\n\nQuestion: {question}" if snippet.strip() else question
            reply = await llm.generate(
                messages=[
                    {"role": "system", "content": (
                        "You are a helpful assistant. Answer using the provided context. "
                        "Be concise. If context lacks the answer, say so."
                    )},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=200,
                temperature=0.1,
            )
            print(f"\n  Reply: {reply.strip()}")

            await memory.process_turn(
                user_id=user_id,
                user_msg=question,
                assistant_reply=reply,
                session_id=session_id,
            )

        print(f"\n{sep}")
        print("EVALUATION SUMMARY")
        print(f"  Turns with facts retrieved  : {facts_hit}/{len(QUESTIONS)}")
        print(f"  Turns with chunks retrieved : {chunks_hit}/{len(QUESTIONS)}")
        print(sep)

    finally:
        memory.shutdown()


if __name__ == "__main__":
    if not os.path.exists("config/uma.yaml"):
        sys.exit("Run from the repo root: python examples/github_chat_eval.py")
    asyncio.run(run())
