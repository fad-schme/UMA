from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

from uma.core.uma_memory import UMAMemory
from uma.adapters.llm.base import LLMInterface
from uma.core.ingest.parser import FileContentParser
logger = logging.getLogger(__name__)


SYSTEM_PROMPT_DEFAULT = (
    "You are a memory quality evaluator. "
    "Do not answer user questions. "
    "Only evaluate the snippet for relevance, completeness, and gaps."
)


# ==== External store reset helpers (Qdrant, Neo4j) ====

def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load UMA YAML config as a plain dict.

    We intentionally avoid constructing UMAMemory here so that `--clear-all` can
    still work even if runtime adapters fail to initialize.
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for --clear-all to reset external stores. Install with `pip install pyyaml`."
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML config format in {path}")
    return data


def _find_neo4j_config(cfg: Any) -> Optional[Dict[str, Any]]:
    """Best-effort recursive search for a Neo4j connection block."""
    if isinstance(cfg, dict):
        # Common shapes: {uri,user,password} or {url,username,password}
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
    """Delete the configured Qdrant collection if present.

    Returns (success, message). This is best-effort and will never raise.
    """
    try:
        storage = cfg.get("storage") if isinstance(cfg.get("storage"), dict) else {}
        vector_backend = storage.get("vector_backend")
        vcfg = storage.get("vector_config") if isinstance(storage.get("vector_config"), dict) else {}

        # Only attempt if the backend looks like Qdrant
        vb = str(vector_backend or "").lower()
        if "qdrant" not in vb:
            return False, "Qdrant backend not configured; skipping."

        collection = vcfg.get("collection")
        url = vcfg.get("url")
        api_key = vcfg.get("api_key")
        path = vcfg.get("path")

        if not collection:
            return False, "Qdrant collection not set; skipping."

        try:
            from qdrant_client import QdrantClient  # type: ignore
        except Exception as exc:
            return False, f"qdrant-client not installed; cannot reset Qdrant ({exc})."

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
    """Wipe all nodes/relationships from configured Neo4j database (best-effort).

    Returns (success, message). This is best-effort and will never raise.
    """
    try:
        neo = _find_neo4j_config(cfg)
        if not neo:
            return False, "Neo4j config not found; skipping graph reset."

        # Normalize common keys
        uri = neo.get("uri") or neo.get("url")
        user = neo.get("user") or neo.get("username")
        password = neo.get("password")
        database = neo.get("database") or neo.get("db") or "neo4j"

        if not (uri and user and password):
            return False, "Neo4j config incomplete (need uri/url, user/username, password); skipping."

        try:
            from neo4j import GraphDatabase  # type: ignore
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
    
    try:
        vector_backend = getattr(memory.raw_config.storage, "vector_backend", "")
        if vector_backend in ("faiss", "inmemory"):
            logging.info("Rebuilding vector indexes from SQL")
            try:
                await memory.rebuild_vector_indexes()
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
                snippet = await memory.get_rendered_context(
                    user_id=user_id, query_text=user_message
                )
                #snippet = await memory.get_structured_context(user_id=user_id, query_text=user_message)
                if not snippet:
                    context_messages = [{"role": "user", "content": user_message}]
                    reply = "No memory snippet available to evaluate."
                else:
                    user_content = (

                        "You are evaluating a retrieved UMA memory snippet for use as LLM context.\n"
                        "Do NOT answer the user’s question.\n"
                        "Do NOT add new facts or suggestions beyond what is in the snippet.\n"
                        "Return ONLY a brief evaluation of whether the snippet is good supporting context for answering the question.\n"
                        "Focus on:\n"
                        "- Relevance to the question\n"
                        "- Coverage/completeness (what important info is missing)\n"
                        "- Specificity/grounding (is it concrete, attributable, unambiguous?)\n"
                        "- Noise/irrelevance (what should be removed)\n"
                        "- Risks (stale info, contradictions, PII/sensitive data)\n\n"

                        f"User question:\n{user_message}\n\n"
                        f"Snippet to evaluate:\n{snippet}\n"
                    )
                    context_messages = [{"role": "user", "content": user_content}]
                    print("\n**************************** Snippet to evaluate:")
                    print(snippet)
                    print("**************************** End of snippet\n\n")

                reply = await agent_generate(
                    messages=[{"role": "system", "content": system_prompt}] + context_messages,
                    llm=getattr(memory, "agent_llm", None) or memory.llm,
                )
                if not isinstance(reply, str) or not reply.strip():
                    reply = "Snippet evaluation unavailable."

                print("Assistant>", reply)
                # Update UMA memory with the turn
                # await memory.process_turn(
                #     user_id=user_id, user_msg=user_message, assistant_reply=reply
                # )

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

        # Best-effort reset of external/vector/graph stores based on config.
        try:
            cfg = _load_yaml_config(cfg_path)
        except Exception as exc:
            logging.warning("Failed to load YAML config for external reset: %s", exc)
            cfg = {}

        ok, msg = _reset_qdrant_from_config(cfg)
        logging.info("Qdrant reset: %s", msg)
        ok2, msg2 = _reset_neo4j_from_config(cfg)
        logging.info("Graph reset: %s", msg2)

        if abs_root in {"/", ""}:
            raise RuntimeError(f"Refusing to clear unsafe db_root path: {abs_root}")
        if os.path.exists(abs_root):
            shutil.rmtree(abs_root)
        os.makedirs(abs_root, exist_ok=True)
        print(f"Cleared UMA storage at {abs_root}")
        if msg:
            print(f"External reset: {msg}")
        if msg2:
            print(f"Graph reset: {msg2}")
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
