from __future__ import annotations

from alphonse.agent.cognition.plans import PlanType
from alphonse.agent.cortex import graph


class _FakeLlm:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self._response


def test_plan_critic_repairs_unknown_tool_to_ask_question() -> None:
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "last_user_message": "remind me in 1 min",
        "correlation_id": "corr-1",
    }
    loop_state = {
        "kind": "discovery_loop",
        "steps": [
            {
                "tool": "notification",
                "parameters": {"text": "Go shower"},
                "status": "ready",
                "executed": False,
                "chunk_index": 0,
                "sequence": 0,
            }
        ],
    }
    llm = _FakeLlm('{"tool":"askQuestion","parameters":{"question":"When should I remind you?"}}')

    result = graph._run_discovery_loop_step(state, loop_state, llm)

    assert "plans" not in result
    assert result.get("response_text") == "When should I remind you?"
    pending = result.get("pending_interaction")
    assert isinstance(pending, dict)
    assert pending.get("type") == "SLOT_FILL"


def test_plan_critic_unrepaired_unknown_tool_creates_gap() -> None:
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "last_user_message": "remind me in 1 min",
        "correlation_id": "corr-1",
    }
    loop_state = {
        "kind": "discovery_loop",
        "steps": [
            {
                "tool": "notification",
                "parameters": {"text": "Go shower"},
                "status": "ready",
                "executed": False,
                "chunk_index": 0,
                "sequence": 0,
            }
        ],
    }
    llm = _FakeLlm('{"tool":"still_fake","parameters":{"x":1}}')

    result = graph._run_discovery_loop_step(state, loop_state, llm)

    plans = result.get("plans")
    assert isinstance(plans, list)
    assert plans
    assert plans[0]["plan_type"] == PlanType.CAPABILITY_GAP
    assert plans[0]["payload"]["reason"] == "unknown_tool_in_plan"
