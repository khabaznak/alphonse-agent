from __future__ import annotations

import json

from alphonse.agent.cognition.intent_catalog import IntentSpec, SlotSpec
from alphonse.agent.cognition.intent_detector_llm import _build_prompt, serialize_intents


class FakePromptStore:
    def __init__(self) -> None:
        self.rules = "RULES_BLOCK"
        self.prompt = "RULES:{rules_block}\nCAT:{catalog_json}\nUSER:{user_message}"

    def get_template(self, key, context):  # type: ignore[no-untyped-def]
        class Match:
            def __init__(self, template: str) -> None:
                self.template = template

        if key == "intent_detector.rules.v1":
            return Match(self.rules)
        if key == "intent_detector.catalog.prompt.v1":
            return Match(self.prompt)
        return Match("")


def _sample_intents() -> list[IntentSpec]:
    return [
        IntentSpec(
            intent_name="b.intent",
            category="task_plane",
            description="B",
            examples=["b1"],
            required_slots=[],
            optional_slots=[
                SlotSpec(
                    name="slot_b",
                    type="string",
                    required=False,
                    prompt_key="x",
                    critical=False,
                )
            ],
            default_mode="aventurizacion",
            risk_level="low",
            handler="b",
            enabled=True,
        ),
        IntentSpec(
            intent_name="a.intent",
            category="task_plane",
            description="A",
            examples=["a1", "a2"],
            required_slots=[
                SlotSpec(
                    name="slot_a",
                    type="string",
                    required=True,
                    prompt_key="y",
                    critical=True,
                )
            ],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="a",
            enabled=True,
        ),
    ]


def test_prompt_uses_rules_and_catalog_json() -> None:
    store = FakePromptStore()
    intents = _sample_intents()
    prompt = _build_prompt(
        intents,
        prompt_store=store,  # type: ignore[arg-type]
        user_message="Hi",
        locale=None,
    )
    assert "RULES_BLOCK" in prompt
    payload = serialize_intents(intents)
    assert json.dumps(payload, ensure_ascii=False, indent=2) in prompt


def test_prompt_is_deterministic() -> None:
    store = FakePromptStore()
    intents = _sample_intents()
    prompt1 = _build_prompt(
        intents,
        prompt_store=store,  # type: ignore[arg-type]
        user_message="Hi",
        locale=None,
    )
    prompt2 = _build_prompt(
        intents,
        prompt_store=store,  # type: ignore[arg-type]
        user_message="Hi",
        locale=None,
    )
    assert prompt1 == prompt2
