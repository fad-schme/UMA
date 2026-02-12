from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

Message = Dict[str, str]  # {"role": "...", "content": "..."}


@dataclass(frozen=True)
class LLMCallConfig:
    max_retries: int = 1
    temperature: float = 0.0
    timeout_s: Optional[float] = None
    strict_json: bool = True
    max_json_chars: int = 50_000


@dataclass(frozen=True)
class LLMCallContext:
    trace_id: Optional[str] = None
    op: str = "llm_call"
    user_id: Optional[str] = None
    owner_type: Optional[str] = None
    owner_id: Optional[str] = None


def extract_json_object(raw: str, *, max_chars: int = 50_000) -> Dict[str, Any]:
    """
    Extract and parse a JSON object from a possibly noisy LLM response.

    Accepts either:
    - a clean JSON object string
    - text containing a JSON object (we take the outermost {...})
    """
    cleaned = (raw or "").strip()
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        logger.exception("extract_json_object: failed to parse JSON response")
        raise

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.error("extract_json_object: no JSON object found in response")
        raise ValueError("No JSON object found in response")

    obj = cleaned[start : end + 1]
    parsed = json.loads(obj)
    if not isinstance(parsed, dict):
        logger.error("extract_json_object: extracted JSON was not an object")
        raise ValueError("Extracted JSON was not an object")
    return parsed


async def generate_text(
    *,
    llm: Any,
    messages: Sequence[Message],
    max_tokens: int,
    cfg: Optional[LLMCallConfig] = None,
    ctx: Optional[LLMCallContext] = None,
) -> str:
    cfg = cfg or LLMCallConfig()
    ctx = ctx or LLMCallContext()

    if llm is None or not hasattr(llm, "generate"):
        logger.error("generate_text: llm with .generate() required")
        raise ValueError("generate_text: llm with .generate() required")

    async def _call() -> str:
        return await llm.generate(
            messages=list(messages),
            max_tokens=int(max_tokens),
            temperature=float(cfg.temperature),
        )

    attempt = 0
    last_err: Optional[BaseException] = None
    retries = max(0, int(cfg.max_retries))
    for attempt in range(retries + 1):
        try:
            if cfg.timeout_s:
                return await asyncio.wait_for(_call(), timeout=float(cfg.timeout_s))
            return await _call()
        except Exception as e:
            last_err = e
            logger.warning(
                "LLM generate failed op=%s attempt=%d/%d trace_id=%s: %s",
                ctx.op,
                attempt + 1,
                retries + 1,
                ctx.trace_id,
                type(e).__name__,
            )
            await asyncio.sleep(0)
    assert last_err is not None
    raise last_err


async def generate_json(
    *,
    llm: Any,
    messages: Sequence[Message],
    max_tokens: int,
    cfg: Optional[LLMCallConfig] = None,
    ctx: Optional[LLMCallContext] = None,
    repair_messages_fn: Optional[Callable[[str], Sequence[Message]]] = None,
) -> Dict[str, Any]:
    cfg = cfg or LLMCallConfig()
    ctx = ctx or LLMCallContext()

    raw = await generate_text(llm=llm, messages=messages, max_tokens=max_tokens, cfg=cfg, ctx=ctx)
    try:
        return extract_json_object(raw, max_chars=int(cfg.max_json_chars))
    except Exception as e:
        if repair_messages_fn:
            try:
                repair_msgs = repair_messages_fn(raw)
                repaired = await generate_text(
                    llm=llm,
                    messages=repair_msgs,
                    max_tokens=max_tokens,
                    cfg=cfg,
                    ctx=LLMCallContext(**{**ctx.__dict__, "op": f"{ctx.op}_repair"}),
                )
                return extract_json_object(repaired, max_chars=int(cfg.max_json_chars))
            except Exception as exc:
                if cfg.strict_json:
                    logger.exception("generate_json repair failed (strict) op=%s trace_id=%s", ctx.op, ctx.trace_id)
                    raise
                logger.warning(
                    "generate_json repair failed op=%s trace_id=%s: %s",
                    ctx.op,
                    ctx.trace_id,
                    exc,
                )
                return {}

        if cfg.strict_json:
            logger.exception("generate_json parse failed (strict) op=%s trace_id=%s", ctx.op, ctx.trace_id)
            raise
        logger.warning("generate_json parse failed op=%s trace_id=%s: %s", ctx.op, ctx.trace_id, e)
        return {}


async def generate_model(
    *,
    llm: Any,
    messages: Sequence[Message],
    max_tokens: int,
    model_validate: Callable[[Dict[str, Any]], T],
    cfg: Optional[LLMCallConfig] = None,
    ctx: Optional[LLMCallContext] = None,
    repair_messages_fn: Optional[Callable[[str], Sequence[Message]]] = None,
) -> T:
    """
    generate_json(...) + validate via provided callable (e.g. Pydantic .model_validate()).
    """
    cfg = cfg or LLMCallConfig()
    ctx = ctx or LLMCallContext()

    repaired_attempted = False
    raw = await generate_text(llm=llm, messages=messages, max_tokens=max_tokens, cfg=cfg, ctx=ctx)
    try:
        parsed = extract_json_object(raw, max_chars=int(cfg.max_json_chars))
    except Exception:
        if repair_messages_fn:
            repaired_attempted = True
            try:
                repair_msgs = repair_messages_fn(raw)
                repaired = await generate_text(
                    llm=llm,
                    messages=repair_msgs,
                    max_tokens=max_tokens,
                    cfg=cfg,
                    ctx=LLMCallContext(**{**ctx.__dict__, "op": f"{ctx.op}_repair"}),
                )
                parsed = extract_json_object(repaired, max_chars=int(cfg.max_json_chars))
            except Exception:
                if cfg.strict_json:
                    logger.exception("generate_model repair failed (strict) op=%s trace_id=%s", ctx.op, ctx.trace_id)
                    raise
                parsed = {}
        else:
            if cfg.strict_json:
                logger.exception("generate_model parse failed (strict) op=%s trace_id=%s", ctx.op, ctx.trace_id)
                raise
            parsed = {}

    def _log_validation_failure(stage: str, payload: Dict[str, Any]) -> None:
        preview = json.dumps(payload, default=str) if isinstance(payload, dict) else str(payload)
        if len(preview) > 2000:
            preview = preview[:2000] + "...(truncated)"
        logger.exception(
            "generate_model validation failed stage=%s op=%s trace_id=%s payload_preview=%s",
            stage,
            ctx.op,
            ctx.trace_id,
            preview,
        )

    try:
        return model_validate(parsed)
    except Exception:
        _log_validation_failure("initial", parsed)
        if repair_messages_fn and not repaired_attempted:
            # One repair attempt on validation failure (no extra retries).
            try:
                repair_msgs = repair_messages_fn(raw)
                repaired = await generate_text(
                    llm=llm,
                    messages=repair_msgs,
                    max_tokens=max_tokens,
                    cfg=cfg,
                    ctx=LLMCallContext(**{**ctx.__dict__, "op": f"{ctx.op}_repair"}),
                )
                repaired_parsed = extract_json_object(repaired, max_chars=int(cfg.max_json_chars))
                return model_validate(repaired_parsed)
            except Exception:
                _log_validation_failure("repair", repaired_parsed if "repaired_parsed" in locals() else {})
                if cfg.strict_json:
                    logger.exception("generate_model validation repair failed (strict) op=%s trace_id=%s", ctx.op, ctx.trace_id)
                    raise

        if cfg.strict_json:
            logger.exception("generate_model validation failed (strict) op=%s trace_id=%s", ctx.op, ctx.trace_id)
            raise

        # Best-effort fallback for non-strict mode.
        try:
            return model_validate({})
        except Exception:
            _log_validation_failure("fallback_empty", {})
            raise
