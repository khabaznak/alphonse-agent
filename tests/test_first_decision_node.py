from __future__ import annotations

from alphonse.agent.cortex.graph import CortexGraph


class _FakeLlm:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self._response


def test_first_decision_direct_reply_bypasses_planning() -> None:
    llm = _FakeLlm(
        '{"route":"direct_reply","intent":"conversation.language","confidence":0.96,'
        '"reply_text":"Si, puedo hablar espanol.","clarify_question":""}'
    )
    runner = CortexGraph().build().compile()
    result = runner.invoke(
        {
            "chat_id": "123",
            "channel_type": "telegram",
            "channel_target": "123",
            "last_user_message": "Can you speak Spanish?",
            "correlation_id": "corr-first-decision",
            "_llm_client": llm,
        }
    )
    assert result.get("response_text") == "Si, puedo hablar espanol."
    assert not result.get("plans")


def test_first_decision_clarify_sets_pending_interaction() -> None:
    llm = _FakeLlm(
        '{"route":"clarify","intent":"task.reminder.create","confidence":0.72,'
        '"reply_text":"","clarify_question":"When do you want the reminder?"}'
    )
    runner = CortexGraph().build().compile()
    result = runner.invoke(
        {
            "chat_id": "123",
            "channel_type": "telegram",
            "channel_target": "123",
            "last_user_message": "Set a reminder",
            "correlation_id": "corr-first-decision-clarify",
            "_llm_client": llm,
        }
    )
    assert result.get("response_text") == "When do you want the reminder?"
    pending = result.get("pending_interaction")
    assert isinstance(pending, dict)
    assert pending.get("key") == "answer"
