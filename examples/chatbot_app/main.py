from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Any, Dict, Optional, Tuple

from uma import UMAMemory
from uma.adapters.llm.base import LLMInterface
from uma.ingest.parser import FileContentParser
from uma.retrieve import ContextPackBuilder

logger = logging.getLogger(__name__)

# Directory that contains this file — used to resolve sibling assets.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT_DEFAULT = (
    "You are a helpful assistant. "
    "Answer the user's question using the provided context. "
    "Be concise and direct. "
    "If the context does not contain enough information, say so clearly."
)


def _format_startup_error(config_path: str, exc: Exception) -> str:
    text = str(exc).strip()
    lines = [
        f"Failed to start the UMA chatbot with config '{config_path}'.",
        f"Cause: {text or type(exc).__name__}",
    ]
    lowered = text.lower()
    if isinstance(exc, ModuleNotFoundError):
        lines.append("The example must be run as a module from the repo root.")
    if "qdrant-client is not installed" in lowered or "no module named 'qdrant_client'" in lowered:
        lines.append(
            "Install vector dependencies with `pip install '.[vector]'` or switch "
            "`storage.vector_backend` to a backend available in your environment."
        )
    elif "neo4j" in lowered and ("not installed" in lowered or "no module named" in lowered):
        lines.append(
            "Install graph dependencies with `pip install '.[graph]'` or set "
            "`storage.graph_backend: disabled`."
        )
    elif "ollama" in lowered and ("not installed" in lowered or "no module named" in lowered):
        lines.append(
            "Install the Ollama client with `pip install '.[ollama]'` or configure a "
            "different LLM/embedder provider."
        )
    elif "failed to initialize client" in lowered or "connection" in lowered or "connectivity" in lowered:
        lines.append(
            "Verify that the backends referenced by your config are installed, reachable, and running."
        )
    lines.append(
        "The supported invocation is: "
        "`python -m examples.chatbot_app.main [--config config/uma.yaml] [--load]`"
    )
    return "\n".join(lines)


def _load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML config format in {path}")
    return data


def _find_neo4j_config(cfg: Any) -> Optional[Dict[str, Any]]:
    if isinstance(cfg, dict):
        keys = {k.lower() for k in cfg.keys() if isinstance(k, str)}
        if ("uri" in keys or "url" in keys) and ("user" in keys or "username" in keys) and "password" in keys:
            return cfg
        for v in cfg.values():
            found = _find_neo4j_config(v)
            if found:
                return found
    elif isinstance(cfg, list):
        for item in cfg:
            found = _find_neo4j_config(item)
            if found:
                return found
    return None


def _reset_qdrant_from_config(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
        vector_backend = storage.get("vector_backend")
        vcfg = storage.get("vector_config") if isinstance(storage.get("vector_config"), dict) else {}
        if "qdrant" not in str(vector_backend or "").lower():
            return False, "Qdrant backend not configured; skipping."
        collection = vcfg.get("collection")
        if not collection:
            return False, "Qdrant collection not set; skipping."
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:
            return False, f"qdrant-client not installed; cannot reset Qdrant ({exc})."
        url = vcfg.get("url")
        api_key = vcfg.get("api_key")
        path = vcfg.get("path")
        client = None
        if url:
            client = QdrantClient(url=url, api_key=api_key, timeout=10.0)
        elif path:
            client = QdrantClient(path=path, timeout=10.0)
        else:
            return False, "Qdrant vector_config must include either url or path; skipping."
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection_name=str(collection))
                return True, f"Deleted Qdrant collection '{collection}'."
            return True, f"Qdrant collection '{collection}' does not exist; nothing to delete."
        finally:
            try:
                client.close()
            except Exception:
                pass
    except Exception as exc:
        return False, f"Failed to reset Qdrant (best-effort): {exc}"


def _reset_neo4j_from_config(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        neo = _find_neo4j_config(cfg)
        if not neo:
            return False, "Neo4j config not found; skipping graph reset."
        uri = neo.get("uri") or neo.get("url")
        user = neo.get("user") or neo.get("username")
        password = neo.get("password")
        database = neo.get("database") or neo.get("db") or "neo4j"
        if not (uri and user and password):
            return False, "Neo4j config incomplete (need uri/url, user/username, password); skipping."
        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            return False, f"neo4j driver not installed; cannot reset graph DB ({exc})."
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session(database=database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            return True, "Cleared Neo4j graph (MATCH (n) DETACH DELETE n)."
        finally:
            try:
                driver.close()
            except Exception:
                pass
    except Exception as exc:
        return False, f"Failed to reset Neo4j (best-effort): {exc}"


async def agent_generate(messages: list, llm: Optional[LLMInterface] = None) -> str:
    if llm is None:
        raise RuntimeError(
            "No LLM configured. Add an `llms.agent` (or `llms.uma`) entry to your config."
        )
    reply = await llm.generate(messages=messages, max_tokens=512, temperature=0.2)
    if not isinstance(reply, str) or not reply.strip():
        logger.warning("agent_generate: LLM returned empty reply.")
    return reply


async def interactive_chat(
    config_path: str = "config/uma.yaml",
    user_id: str = "user:local",
    agent_id: str = "agent-default",
    system_prompt: Optional[str] = None,
    load_bootstrap: bool = False,
):
    system_prompt = system_prompt or SYSTEM_PROMPT_DEFAULT
    session_id = f"chat:{user_id}"

    try:
        memory = UMAMemory.from_yaml(config_path).set_context(agent_id=agent_id)
    except Exception as exc:
        raise RuntimeError(_format_startup_error(config_path, exc)) from exc

    async def _load_all() -> None:
        # Memory bootstrap (facts, user profile, agent soul, diary)
        await memory.load_memory_bootstrap(
            os.path.join(_APP_DIR, "..", "MEMORY.md"),
            user_id=user_id,
        )
        await memory.load_daily_diary_bootstrap(
            os.path.join(_APP_DIR, "..", "DAILY_DIARY.md"),
            user_id=user_id,
        )
        memory.load_userprofile(os.path.join(_APP_DIR, "..", "USER.md"))
        memory.load_agentprofile(os.path.join(_APP_DIR, "..", "SOUL.md"))

        # Ingest documents from the chatbot_app directory
        pdf_path = os.path.join(_APP_DIR, "github-manual.pdf")
        if os.path.isfile(pdf_path):
            print(f"Ingesting {os.path.basename(pdf_path)} ...")
            try:
                await memory.ingest_document(
                    pdf_path,
                    owner_type="agent",
                    owner_id=agent_id,
                )
                print("  Done.")
            except Exception:
                logger.exception("Failed to ingest %s", pdf_path)
        else:
            logger.warning("github-manual.pdf not found at %s", pdf_path)

    try:
        vector_backend = getattr(memory.raw_config.storage, "vector_backend", "")
        if vector_backend in ("faiss", "inmemory"):
            try:
                await memory.rebuild_vector_indexes()
            except Exception:
                logger.exception("Vector index rebuild failed; continuing with empty index.")

        if load_bootstrap:
            print("Loading memory bootstrap and documents ...")
            await _load_all()
            print("Bootstrap load complete.\n")

        ctx_cfg = getattr(getattr(memory, "retrieval_cfg", None), "context", None)
        llm = getattr(memory, "agent_llm", None) or memory.llm

        print("UMA chatbot ready. Commands: /load, /setprompt <text>, /quit")

        while True:
            try:
                user_input = input("You> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if not user_input:
                continue

            if user_input.lower().startswith("/q"):
                break

            if user_input.lower().startswith("/load"):
                print("Loading memory bootstrap and documents ...")
                await _load_all()
                print("Bootstrap load complete.")
                continue

            if user_input.lower().startswith("/setprompt "):
                system_prompt = user_input[len("/setprompt "):].strip() or system_prompt
                print(f"System prompt updated.")
                continue

            try:
                context = await memory.retrieve_context(
                    query_text=user_input,
                    user_id=user_id,
                )
                pack = ContextPackBuilder.build(user_input, context)
                snippet = await ContextPackBuilder.render_snippet_async(
                    pack,
                    context_cfg=ctx_cfg,
                    llm=llm,
                )

                print("\n--- memory context ---")
                print(snippet if snippet.strip() else "(no context retrieved)")
                print("---------------------\n")

                if snippet.strip():
                    user_content = (
                        f"Context:\n{snippet}\n\n"
                        f"Question: {user_input}"
                    )
                else:
                    user_content = user_input

                reply = await agent_generate(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    llm=llm,
                )
                if not isinstance(reply, str) or not reply.strip():
                    reply = "(no reply)"

                print("Assistant>", reply)

                await memory.process_turn(
                    user_id=user_id,
                    user_msg=user_input,
                    assistant_reply=reply,
                    session_id=session_id,
                )

            except Exception as exc:
                logger.exception("Chat turn failed: %s", exc)
    finally:
        try:
            memory.shutdown()
        except Exception:
            logger.exception("Failed to shut down UMA memory.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="UMA interactive chatbot example")
    parser.add_argument("--config", default="config/uma.yaml")
    parser.add_argument("--user", default="user:local")
    parser.add_argument("--agent", default="agent-default")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--load",
        action="store_true",
        help="Ingest github-manual.pdf and load the memory bootstrap on startup.",
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Delete all UMA SQL and vector stores, then start fresh with --load.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"Config not found: {args.config} — run from the repo root")

    if args.clear_all:
        try:
            cfg = _load_yaml_config(args.config)
        except Exception as exc:
            logger.warning("Failed to load YAML config for --clear-all: %s", exc)
            cfg = {}

        ok, msg = _reset_qdrant_from_config(cfg)
        if msg:
            print(f"Qdrant: {msg}")
        ok2, msg2 = _reset_neo4j_from_config(cfg)
        if msg2:
            print(f"Graph: {msg2}")

        storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
        db_root = str(storage.get("db_root") or ".uma/db").rstrip("/")
        db_root_base = str(storage.get("db_root_base") or "cwd").strip().lower()
        if db_root_base in ("cwd", "workdir", "auto"):
            abs_db_root = os.path.abspath(db_root)
        else:
            abs_db_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.config)), db_root))

        vector_config = cfg.get("storage", {}).get("vector_config") or {}
        vector_path = str(vector_config.get("path") or ".uma/vectors")
        abs_vector_path = os.path.abspath(vector_path)

        for path in (abs_db_root, abs_vector_path):
            if path in {"/", ""}:
                raise SystemExit(f"Refusing to clear unsafe path: {path}")
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
                print(f"Cleared {path}")

        args_load = True
    else:
        args_load = args.load

    try:
        asyncio.run(
            interactive_chat(
                config_path=args.config,
                user_id=args.user,
                agent_id=args.agent,
                system_prompt=args.system_prompt,
                load_bootstrap=args_load,
            )
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
