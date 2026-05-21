from __future__ import annotations

import json

import pytest

from uma.memory.semantic.extractor import FactExtractor


class _PromptSensitiveLLM:
    async def generate(self, messages, max_tokens: int = 0, temperature: float = 0.0):
        _ = max_tokens
        _ = temperature
        system = ""
        user = ""
        for message in list(messages or []):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                system = str(message.get("content") or "")
            elif message.get("role") == "user":
                user = str(message.get("content") or "")

        required_markers = (
            "user goals",
            "current projects or research topics",
            "identity statements self-declared by the user",
            "community affiliation",
            "career or education plans",
            "important life context",
        )
        if not all(marker in system.lower() for marker in required_markers):
            return json.dumps({"facts": []})

        text = user.split("TEXT:\n", 1)[-1].strip().lower()
        if "education" in text and "mental health" in text:
            return json.dumps(
                {
                    "facts": [
                        {
                            "predicate": "INTERESTED_IN",
                            "object": "counseling or mental health work",
                            "confidence": 0.88,
                            "source_ids": [],
                        },
                        {
                            "predicate": "PLANS",
                            "object": "continue education and explore career options",
                            "confidence": 0.82,
                            "source_ids": [],
                        },
                    ]
                }
            )
        if "adoption agencies" in text:
            return json.dumps(
                {
                    "facts": [
                        {
                            "predicate": "RESEARCHING",
                            "object": "adoption agencies",
                            "confidence": 0.9,
                            "source_ids": [],
                        }
                    ]
                }
            )
        if "transgender journey" in text and "trans community" in text:
            return json.dumps(
                {
                    "facts": [
                        {
                            "predicate": "IDENTIFIES_WITH",
                            "object": "trans community",
                            "confidence": 0.86,
                            "source_ids": [],
                        },
                        {
                            "predicate": "DISCUSSES",
                            "object": "transgender journey",
                            "confidence": 0.85,
                            "source_ids": [],
                        },
                    ]
                }
            )
        return json.dumps({"facts": []})


def _fact_objects(facts) -> set[str]:
    return {str(getattr(fact, "object", "")).lower() for fact in list(facts or [])}


@pytest.mark.asyncio
async def test_extract_user_facts_captures_durable_self_declared_context() -> None:
    extractor = FactExtractor(llm=_PromptSensitiveLLM())

    education_facts = await extractor.extract_user_facts(
        subject="user",
        text="I want to continue my education and check out career options. I am keen on counseling or working in mental health.",
        owner_type="user",
        owner_id="user:u1",
    )
    adoption_facts = await extractor.extract_user_facts(
        subject="user",
        text="I am researching adoption agencies and one of the adoption agencies I am looking into seems promising.",
        owner_type="user",
        owner_id="user:u1",
    )
    identity_facts = await extractor.extract_user_facts(
        subject="user",
        text="I want to talk about my transgender journey and give a voice to the trans community.",
        owner_type="user",
        owner_id="user:u1",
    )

    education_objects = _fact_objects(education_facts)
    adoption_objects = _fact_objects(adoption_facts)
    identity_objects = _fact_objects(identity_facts)

    assert any("education" in obj or "career" in obj for obj in education_objects)
    assert any("counseling" in obj or "mental health" in obj for obj in education_objects)
    assert any("adoption agenc" in obj for obj in adoption_objects)
    assert any("transgender journey" in obj or "trans community" in obj for obj in identity_objects)
