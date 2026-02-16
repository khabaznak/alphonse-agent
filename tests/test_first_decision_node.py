from __future__ import annotations

from alphonse.agent.cortex.graph import CortexGraph


class _FakeLlm:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self._response


class _CapturePromptLlm:
    def __init__(self, response: str) -> None:
        self._response = response
        self.user_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        self.user_prompt = user_prompt
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


def test_first_decision_prompt_uses_recent_conversation_not_session_state() -> None:
    llm = _CapturePromptLlm(
        '{"route":"direct_reply","intent":"conversation","confidence":0.9,'
        '"reply_text":"ok","clarify_question":""}'
    )
    runner = CortexGraph().build().compile()
    _ = runner.invoke(
        {
            "chat_id": "123",
            "channel_type": "telegram",
            "channel_target": "123",
            "last_user_message": "How many dogs do I have?",
            "recent_conversation_block": "## RECENT CONVERSATION (last 10 turns)\n- User: I have 3 dogs.",
            "correlation_id": "corr-first-decision-context",
            "_llm_client": llm,
        }
    )
    assert "## RECENT CONVERSATION (last 10 turns)" in llm.user_prompt
    assert "SESSION_STATE (" not in llm.user_prompt


def test_first_decision_prompt_falls_back_to_session_state_recent_conversation() -> None:
    llm = _CapturePromptLlm(
        '{"route":"direct_reply","intent":"conversation","confidence":0.9,'
        '"reply_text":"ok","clarify_question":""}'
    )
    runner = CortexGraph().build().compile()
    _ = runner.invoke(
        {
            "chat_id": "123",
            "channel_type": "telegram",
            "channel_target": "123",
            "last_user_message": "How many dogs do I have?",
            "session_state": {
                "session_id": "u1|2026-02-16",
                "user_id": "u1",
                "date": "2026-02-16",
                "rev": 1,
                "channels_seen": ["telegram"],
                "recent_conversation": [{"user": "I have 3 dogs", "assistant": "Noted."}],
                "working_set": [],
                "open_loops": [],
                "last_action": None,
            },
            "correlation_id": "corr-first-decision-context-fallback",
            "_llm_client": llm,
        }
    )
    assert "- User: I have 3 dogs" in llm.user_prompt
