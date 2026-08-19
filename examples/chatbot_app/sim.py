"""Scripted cross-source conversation.

Drives a fixed question set that spans an ingested document, the user profile,
and the diary, so you can see UMA pull from several lanes in one session.

Ingest and bootstrap first (see `README.md`), then:

    python examples/chatbot_app/sim.py --config path/to/uma.yaml

UMA manages memory only — it does not generate replies. This example owns its
own LLM client, built from the same `llms.uma` block in `uma.yaml`. Requires the
`openai` package (`pip install 'uma-mem[openai]'` or `'uma-mem[ollama]'`).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from uma import ContextBundle, UMAMemory

USER_ID = "user:local"
AGENT_ID = "agent-default"
SESSION_ID = "chat:sim"
SEP = "=" * 70

QUESTIONS = [
    # Ingested document
    "How do I create a new branch in GitHub?",
    "What is a pull request and how do I open one?",
    "How does the GitHub fork workflow work?",
    # Profile lane
    "What is Anna's availability window for scheduling meetings?",
    "What projects is Anna currently working on?",
    # Cross-source: document knowledge plus profile preferences
    "Anna is reviewing a PR. What does she look for in code reviews?",
    # Episodic lane
    "What did Anna work on last week in the diary?",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely using the provided context. "
    "If the context does not contain enough information, say so clearly."
)


def build_llm(config_path: str):
    """Create an OpenAI-compatible client from the config's `llms.uma` block."""
    from openai import AsyncOpenAI

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    llm_cfg = cfg["llms"]["uma"]
    provider_cfg = llm_cfg.get("config") or {}
    host = provider_cfg.get("host", "http://localhost:11434")
    client = AsyncOpenAI(
        base_url=f"{host.rstrip('/')}/v1",
        api_key=provider_cfg.get("api_key", "not-needed-for-ollama"),
        timeout=provider_cfg.get("timeout", 120.0),
    )
    return client, llm_cfg["model"]


def render_context(bundle: ContextBundle, max_chars: int = 4000) -> str:
    """Flatten a ContextBundle into prompt text.

    Read the bundle by attribute. `facts` are `Fact` domain objects carrying a
    subject-predicate-object triple, not dicts with a `text` key.
    """
    parts: list[str] = []

    if bundle.facts:
        parts.append("Known facts:")
        parts.extend(
            "- " + " ".join(
                part for part in (fact.subject, fact.predicate, fact.object) if part
            )
            for fact in bundle.facts
        )

    if bundle.episodic:
        summaries = [
            " ".join(str(getattr(ep, "summary", "")).split()) for ep in bundle.episodic
        ]
        summaries = [s for s in summaries if s]
        if summaries:
            parts.append("\nEarlier sessions:")
            parts.extend(f"- {s}" for s in summaries)

    if bundle.chunks:
        parts.append("\nSupporting excerpts:")
        parts.extend(f"- {' '.join(str(c.text).split())}" for c in bundle.chunks)

    return "\n".join(parts)[:max_chars]


async def run(config_path: str) -> None:
    memory = UMAMemory.from_yaml(config_path)
    client, model = build_llm(config_path)

    try:
        for index, question in enumerate(QUESTIONS, start=1):
            print(f"\n{SEP}")
            print(f"Q{index:02d}: {question}")
            print("-" * 70)

            bundle = await memory.retrieve_context(
                query_text=question,
                user_id=USER_ID,
                session_id=SESSION_ID,
                agent_id=AGENT_ID,
            )
            context = render_context(bundle)

            print("--- memory context ---")
            print(context if context.strip() else "(no context retrieved)")
            print("--- end context ---\n")

            user_content = (
                f"Context:\n{context}\n\nQuestion: {question}"
                if context.strip()
                else question
            )
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=256,
                temperature=0.2,
            )
            reply = (response.choices[0].message.content or "").strip()
            print(f"Assistant: {reply}")

            await memory.process_turn(
                user_id=USER_ID,
                user_msg=question,
                assistant_reply=reply,
                session_id=SESSION_ID,
                agent_id=AGENT_ID,
            )
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
