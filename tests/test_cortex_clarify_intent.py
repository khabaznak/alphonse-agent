from __future__ import annotations

from alphonse.agent.cortex.graph import invoke_cortex


class StubLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self._payload


def test_unknown_routes_to_clarify_response() -> None:
    llm = StubLLM(
        '{"category":"TASK_PLANE","intent_guess":null,"confidence":0.3,"needs_clarification":true}'
    )
    state = {
        "chat_id": "cli",
        "channel_type": "cli",
        "channel_target": "cli",
        "timezone": "America/Mexico_City",
    }
    result = invoke_cortex(state, "Necesito algo raro", llm_client=llm)
    assert result.meta.get("response_key") == "clarify.intent"
    assert result.reply_text is not None
