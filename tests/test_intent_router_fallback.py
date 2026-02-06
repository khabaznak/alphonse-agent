from __future__ import annotations

from alphonse.agent.cognition.intent_registry import IntentCategory, get_registry
from alphonse.agent.cognition.intent_router import route_message


class StubLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self._payload


def test_llm_fallback_routes_greeting_variant() -> None:
    llm = StubLLM(
        '{"category":"CORE_CONVERSATIONAL","intent_guess":"greeting","confidence":0.8,"needs_clarification":false}'
    )
    routed = route_message("Quiubo, mi GPT!?", registry=get_registry(), llm_client=llm)
    assert routed.intent == "greeting"
    assert routed.category == IntentCategory.CORE_CONVERSATIONAL
    assert routed.needs_clarification is False


def test_llm_low_confidence_clarifies() -> None:
    llm = StubLLM(
        '{"category":"TASK_PLANE","intent_guess":null,"confidence":0.4,"needs_clarification":true}'
    )
    routed = route_message("Necesito algo raro", registry=get_registry(), llm_client=llm)
    assert routed.intent == "unknown"
    assert routed.needs_clarification is True
    assert routed.rationale == "llm_clarify"


def test_llm_meta_capabilities_routes() -> None:
    llm = StubLLM(
        '{"category":"DEBUG_META","intent_guess":"meta.capabilities","confidence":0.9,"needs_clarification":false}'
    )
    routed = route_message("What else can you do?", registry=get_registry(), llm_client=llm)
    assert routed.intent == "meta.capabilities"
    assert routed.category == IntentCategory.DEBUG_META
