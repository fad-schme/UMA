from __future__ import annotations

import inspect
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from uma import UMAMemory
from uma.retrieve.context_pack_builder import ContextPackBuilder

from locomo.loader import ConversationRecord, QARecord, TurnRecord


class LocomoUMAAdapter:
    def __init__(self, config_path: str, *, disable_llm: bool = False) -> None:
        self.config_path = config_path
        self.disable_llm = disable_llm
        self.agent_id = "locomo_agent"
        self.memory = UMAMemory.from_yaml(config_path).set_context(agent_id=self.agent_id)
        self.context_cfg = getattr(getattr(self.memory, "retrieval_cfg", None), "context", None)
        self.llm = None if disable_llm else (getattr(self.memory, "agent_llm", None) or getattr(self.memory, "llm", None))

    def close(self) -> None:
        self.memory.shutdown()

    async def ingest_conversation(self, conversation: ConversationRecord) -> list[str]:
        warnings: list[str] = []
        user_id = _user_id(conversation.conversation_id)
        session_id = _session_id(conversation.conversation_id, conversation.metadata)
        turns = conversation.turns
        distinct_speakers = sorted({turn.speaker for turn in turns})
        if len(distinct_speakers) > 2:
            warnings.append(
                "Conversation has more than two speakers; UMA ingestion uses adjacent-turn pairing and preserves raw labels in metadata."
            )
        for start in range(0, len(turns), 2):
            pair = turns[start : start + 2]
            if len(pair) < 2:
                warnings.append(
                    f"Skipped trailing unpaired turn index={pair[0].index} conversation_id={conversation.conversation_id}."
                )
                continue
            first, second = pair
            extra_meta = {
                "locomo": {
                    "conversation_id": conversation.conversation_id,
                    "source_turn_indices": [first.index, second.index],
                    "source_speakers": [first.speaker, second.speaker],
                    "source_timestamps": [first.timestamp, second.timestamp],
                    "original_order": [first.index, second.index],
                    "original_turn_text": [first.text, second.text],
                    "raw_turn_count": len(pair),
                    "conversation_speakers": distinct_speakers,
                }
            }
            if len(distinct_speakers) > 2:
                extra_meta["locomo"]["speaker_mapping_warning"] = "more_than_two_distinct_speakers_in_pair"
            try:
                await self.memory.process_turn(
                    user_id=user_id,
                    user_msg=first.text,
                    assistant_reply=second.text,
                    session_id=session_id,
                    extra_meta=extra_meta,
                )
            except Exception as exc:
                warnings.append(
                    "Turn ingest failed "
                    f"conversation_id={conversation.conversation_id} "
                    f"turn_indices={[first.index, second.index]} "
                    f"error={type(exc).__name__}: {exc}"
                )
        return warnings

    async def run_question(self, conversation: ConversationRecord, qa: QARecord) -> dict[str, Any]:
        user_id = _user_id(conversation.conversation_id)
        session_id = _session_id(conversation.conversation_id, conversation.metadata)
        record: dict[str, Any] = {
            "conversation_id": conversation.conversation_id,
            "question_id": qa.question_id,
            "question": qa.question,
            "expected_answer": qa.expected_answer,
            "predicted_answer": None,
            "memory_results": [],
            "context_result": {
                "rendered_context": "",
                "facts": [],
                "chunks": [],
                "episodic": [],
                "documents": [],
                "trace": {},
                "provenance": {},
            },
            "active_lanes": [],
            "latency_ms": {
                "retrieve_memory": None,
                "retrieve_context": None,
                "answer_generation": None,
            },
            "token_counts": {},
            "warnings": [],
            "error": None,
        }

        memory_result: dict[str, Any] = {}
        context_result: dict[str, Any] = {}

        try:
            started = time.perf_counter()
            memory_result = await self.memory.retrieve_memory(
                query_text=qa.question,
                user_id=user_id,
                session_id=session_id,
                include_debug=True,
            )
            record["latency_ms"]["retrieve_memory"] = round((time.perf_counter() - started) * 1000, 3)
            record["memory_results"] = _normalize_memory_results(memory_result)

            started = time.perf_counter()
            context_result = await self.memory.retrieve_context(
                query_text=qa.question,
                user_id=user_id,
                session_id=session_id,
            )
            record["latency_ms"]["retrieve_context"] = round((time.perf_counter() - started) * 1000, 3)
            record["active_lanes"] = list(context_result.get("active_lanes") or [])
            record["context_result"] = await _normalize_context_result(
                qa.question,
                context_result,
                context_cfg=self.context_cfg,
                llm=self.llm,
            )

            if self.disable_llm:
                record["warnings"].append("LLM answer generation disabled by --no-llm.")
            elif self.llm is None:
                record["warnings"].append("No configured LLM available for answer generation.")
            else:
                started = time.perf_counter()
                record["predicted_answer"] = await self._generate_answer(
                    question=qa.question,
                    rendered_context=record["context_result"]["rendered_context"],
                )
                record["latency_ms"]["answer_generation"] = round((time.perf_counter() - started) * 1000, 3)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if not record["memory_results"] and memory_result:
                record["memory_results"] = _normalize_memory_results(memory_result)
            if context_result and not record["context_result"]["rendered_context"]:
                record["context_result"] = await _normalize_context_result(
                    qa.question,
                    context_result,
                    context_cfg=self.context_cfg,
                    llm=None,
                )
        return record

    async def _generate_answer(self, *, question: str, rendered_context: str) -> str:
        user_content = question
        if rendered_context.strip():
            user_content = f"Context:\n{rendered_context}\n\nQuestion: {question}"
        reply = await self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using the provided context. "
                        "If the context is insufficient, say so explicitly."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=256,
            temperature=0.1,
        )
        return reply.strip() if isinstance(reply, str) else repr(reply)


def safe_reset_from_config(config_path: str) -> tuple[bool, str]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        return False, f"Invalid config format in {config_path}."
    storage = cfg.get("storage")
    if not isinstance(storage, dict):
        return False, "Config does not contain a storage section."
    db_root = storage.get("db_root")
    if not isinstance(db_root, str) or not db_root.strip():
        return False, "storage.db_root is missing or empty."

    root = _resolve_storage_path(
        db_root=db_root,
        db_root_base=str(storage.get("db_root_base") or "auto"),
        config_path=config_path,
    )
    repo_root = Path.cwd().resolve()
    targets = [root]
    vector_cfg = storage.get("vector_config")
    if isinstance(vector_cfg, dict):
        vector_path = vector_cfg.get("path")
        if isinstance(vector_path, str) and vector_path.strip():
            targets.append(
                _resolve_storage_path(
                    db_root=vector_path,
                    db_root_base=str(storage.get("db_root_base") or "auto"),
                    config_path=config_path,
                )
            )

    deleted: list[str] = []
    skipped: list[str] = []
    for target in targets:
        if not _is_safe_local_reset_target(target, repo_root):
            skipped.append(f"{target} (outside repo boundary)")
            continue
        if not target.exists():
            continue
        _remove_tree(target)
        deleted.append(str(target))

    if deleted:
        suffix = f" Skipped: {', '.join(skipped)}." if skipped else ""
        return True, f"Deleted local UMA storage paths: {', '.join(deleted)}.{suffix}"
    if skipped:
        return False, f"No storage paths deleted. Skipped: {', '.join(skipped)}."
    return True, "No local UMA storage paths existed; nothing to delete."


def _resolve_storage_path(*, db_root: str, db_root_base: str, config_path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(db_root)))
    if expanded.is_absolute():
        return expanded.resolve()

    cwd_root = (Path.cwd() / expanded).resolve()
    config_root = (Path(config_path).resolve().parent / expanded).resolve()
    base = db_root_base.strip().lower() or "auto"
    if base in {"config", "config_dir"}:
        return config_root
    if base in {"cwd", "workdir"}:
        return cwd_root
    if base == "auto":
        return cwd_root if cwd_root.exists() or not config_root.exists() else config_root
    return cwd_root


def _is_safe_local_reset_target(path: Path, repo_root: Path) -> bool:
    try:
        path.relative_to(repo_root)
        return True
    except ValueError:
        return False


def _remove_tree(path: Path) -> None:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    if last_error is not None:
        raise last_error


def _user_id(conversation_id: str) -> str:
    return f"locomo_user_{conversation_id}"


def _session_id(conversation_id: str, metadata: dict[str, Any]) -> str:
    session_value = metadata.get("session_id") or metadata.get("session") or metadata.get("dialogue_id")
    if isinstance(session_value, str) and session_value.strip():
        return f"locomo_{session_value.strip()}"
    return f"locomo_{conversation_id}"


async def _normalize_context_result(
    question: str,
    context_result: dict[str, Any],
    *,
    context_cfg: Any,
    llm: Any,
) -> dict[str, Any]:
    pack = ContextPackBuilder.build(question, context_result)
    rendered_context = await ContextPackBuilder.render_snippet_async(pack, context_cfg=context_cfg, llm=llm)
    return {
        "rendered_context": rendered_context,
        "facts": [_sanitize_for_benchmark(_jsonable(item)) for item in context_result.get("facts") or []],
        "chunks": [_sanitize_for_benchmark(_jsonable(item)) for item in context_result.get("chunks") or []],
        "episodic": [_sanitize_for_benchmark(_jsonable(item)) for item in context_result.get("episodic") or []],
        "documents": [_sanitize_for_benchmark(_jsonable(item)) for item in context_result.get("documents") or []],
        "trace": _sanitize_for_benchmark(_jsonable(context_result.get("trace") or [])),
        "provenance": _sanitize_for_benchmark(_jsonable(context_result.get("provenance") or {})),
    }


def _normalize_memory_results(memory_result: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    debug_payload = memory_result.get("debug")
    debug_memories: list[Any] = []
    if isinstance(debug_payload, dict):
        debug_memories = list(debug_payload.get("memories") or [])

    memories = debug_memories
    if not memories:
        facts = list(memory_result.get("facts") or [])
        evidence = list(memory_result.get("evidence") or [])
        for rank, fact in enumerate(facts, start=1):
            obj = _jsonable(fact)
            if not isinstance(obj, dict):
                obj = {"text": obj}
            normalized.append(
                {
                    "id": obj.get("id"),
                    "lane": "semantic",
                    "type": "fact",
                    "text": obj.get("text"),
                    "score": obj.get("confidence"),
                    "rank": rank,
                    "source_ids": list(obj.get("source_chunk_ids") or []),
                    "provenance": {},
                    "timestamp": None,
                    "metadata": {},
                }
            )
        start_rank = len(normalized) + 1
        for offset, item in enumerate(evidence, start=0):
            obj = _jsonable(item)
            if not isinstance(obj, dict):
                obj = {"text": obj}
            normalized.append(
                {
                    "id": obj.get("id"),
                    "lane": "raw",
                    "type": "evidence",
                    "text": obj.get("text"),
                    "score": None,
                    "rank": start_rank + offset,
                    "source_ids": [obj.get("source_document_id")] if obj.get("source_document_id") else [],
                    "provenance": {},
                    "timestamp": None,
                    "metadata": {"source": obj.get("source")} if obj.get("source") else {},
                }
            )
        return normalized

    for rank, item in enumerate(memories, start=1):
        obj = _jsonable(item)
        if not isinstance(obj, dict):
            obj = {"value": obj}
        source_ids = _collect_source_ids(obj)
        normalized.append(
            {
                "id": obj.get("id") or obj.get("artifact_id") or obj.get("doc_id"),
                "lane": obj.get("kb_lane") or obj.get("lane") or obj.get("artifact_lane"),
                "type": obj.get("artifact_kind") or obj.get("artifact_type") or obj.get("kind") or obj.get("type"),
                "text": obj.get("text") or obj.get("summary") or obj.get("content") or obj.get("snippet"),
                "score": obj.get("score") or obj.get("final_score") or obj.get("confidence"),
                "rank": rank,
                "source_ids": source_ids,
                "provenance": _sanitize_for_benchmark(_jsonable(obj.get("provenance") or {})),
                "timestamp": obj.get("timestamp") or obj.get("derived_at") or obj.get("created_at"),
                "metadata": _sanitize_for_benchmark(_jsonable(obj.get("metadata") or obj.get("meta") or {})),
            }
        )
    return normalized


def _collect_source_ids(obj: dict[str, Any]) -> list[Any]:
    source_ids: list[Any] = []
    for key in (
        "source_ids",
        "chunk_ids",
        "direct_source_chunk_ids",
        "direct_source_document_ids",
        "parent_artifact_ids",
        "related_artifact_ids",
    ):
        value = obj.get(key)
        if isinstance(value, list):
            source_ids.extend(item for item in value if item is not None)
    return source_ids


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return _jsonable(value.dict())
    for attr in ("__dict__",):
        raw = getattr(value, attr, None)
        if isinstance(raw, dict):
            return _jsonable(raw)
    candidate: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        member = getattr(value, name, None)
        if inspect.ismethod(member) or inspect.isfunction(member):
            continue
        if isinstance(member, (str, int, float, bool)) or member is None:
            candidate[name] = member
        elif isinstance(member, (list, tuple, set, dict)):
            candidate[name] = _jsonable(member)
    return candidate or repr(value)


def _sanitize_for_benchmark(value: Any) -> Any:
    blocked = {"vector", "embedding", "embeddings", "dense_vector", "sparse_vector", "query_vector"}
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        embedding_removed = False
        embedding_dim: int | None = None
        for key, item in value.items():
            key_str = str(key)
            if key_str.lower() in blocked:
                embedding_removed = True
                if isinstance(item, list) and item and all(isinstance(x, (int, float)) for x in item):
                    embedding_dim = len(item)
                continue
            cleaned[key_str] = _sanitize_for_benchmark(item)
        if embedding_removed:
            cleaned["embedding_removed"] = True
            if embedding_dim is not None:
                cleaned["embedding_dim"] = embedding_dim
        return cleaned
    if isinstance(value, list):
        return [_sanitize_for_benchmark(item) for item in value]
    return value
