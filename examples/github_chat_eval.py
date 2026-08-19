"""10-round retrieval evaluation against an ingested document.

Ingest `examples/chatbot_app/github-manual.pdf` first, then run this to see what
UMA retrieves per turn and how a model answers with that context.

    python examples/chatbot_app/main.py --config path/to/uma.yaml --ingest ...
    python examples/github_chat_eval.py --config path/to/uma.yaml

UMA manages memory only — it does not generate replies. This example therefore
owns its own LLM client, built from the same `llms.uma` block in `uma.yaml`, to
keep the boundary visible. Requires the `openai` package (`pip install
'uma-mem[openai]'` or `'uma-mem[ollama]'`).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from uma import ContextBundle, UMAMemory

USER_ID = "user:local"
AGENT_ID = "agent-default"
SESSION_ID = "chat:github-eval"
SEP = "=" * 72

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

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using the provided context. "
    "Be concise. If the context lacks the answer, say so."
)


def build_llm(config_path: str):
    """Create an OpenAI-compatible client from the config's `llms.uma` block.

    Your application owns the model call; UMA owns the memory. Reading the same
    config here is a convenience for the example, not a UMA API.
    """
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

    `ContextBundle` is a Pydantic model: read it by attribute. Its `facts` are
    `Fact` domain objects carrying a subject-predicate-object triple — there is
    no `.text` field. (`retrieve_memory` is the API that returns pre-flattened
    fact dicts with a `text` key; the two products differ deliberately.)
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

    if bundle.chunks:
        parts.append("\nSupporting excerpts:")
        for chunk in bundle.chunks:
            text = " ".join(str(chunk.text).split())
            parts.append(f"- {text}")

    rendered = "\n".join(parts)
    return rendered[:max_chars]


async def run(config_path: str) -> None:
    memory = UMAMemory.from_yaml(config_path)
    client, model = build_llm(config_path)

    facts_hit = 0
    chunks_hit = 0

    try:
        for index, question in enumerate(QUESTIONS, start=1):
            print(f"\n{SEP}")
            print(f"Q{index:02d}: {question}")
            print("-" * 72)

            bundle = await memory.retrieve_context(
                query_text=question,
                user_id=USER_ID,
                session_id=SESSION_ID,
                agent_id=AGENT_ID,
            )

            facts_hit += bool(bundle.facts)
            chunks_hit += bool(bundle.chunks)
            print(
                f"  Retrieved: {len(bundle.facts)} facts | "
                f"{len(bundle.chunks)} chunks | {len(bundle.episodic)} episodic"
            )

            context = render_context(bundle)
            if context.strip():
                print(f"  Context ({len(context)} chars):")
                for line in context.splitlines()[:12]:
                    print(f"    {line}")
            else:
                print("  Context: (none)")

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
                max_tokens=200,
                temperature=0.1,
            )
            reply = (response.choices[0].message.content or "").strip()
            print(f"\n  Reply: {reply}")

            await memory.process_turn(
                user_id=USER_ID,
                user_msg=question,
                assistant_reply=reply,
                session_id=SESSION_ID,
                agent_id=AGENT_ID,
            )

        print(f"\n{SEP}")
        print("EVALUATION SUMMARY")
        print(f"  Turns with facts retrieved  : {facts_hit}/{len(QUESTIONS)}")
        print(f"  Turns with chunks retrieved : {chunks_hit}/{len(QUESTIONS)}")
        print(SEP)
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
