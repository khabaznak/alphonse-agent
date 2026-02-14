from __future__ import annotations

from alphonse.agent.cortex.graph import CortexGraph


class _FakeLlm:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self._response


def test_task_route_initializes_task_state() -> None:
    llm = _FakeLlm(
        '{"route":"tool_plan","intent":"meta.query","confidence":0.82,'
        '"reply_text":"","clarify_question":""}'
    )
    runner = CortexGraph().build().compile()
    result = runner.invoke(
        {
            "chat_id": "123",
            "channel_type": "telegram",
            "channel_target": "123",
            "last_user_message": "What can you do?",
            "correlation_id": "corr-task-entry",
            "_llm_client": llm,
        }
    )

    task_state = result.get("task_state")
    assert isinstance(task_state, dict)
    assert task_state.get("mode") == "task"
    assert str(task_state.get("pdca_phase") or "") in {"plan", "check", "do", "act"}
    assert int(task_state.get("cycle_index") or 0) >= 0
    assert task_state.get("initialized") is True


def test_chat_route_skips_task_state_entry() -> None:
    llm = _FakeLlm(
        '{"route":"direct_reply","intent":"conversation.generic","confidence":0.96,'
        '"reply_text":"Hi!","clarify_question":""}'
    )
    runner = CortexGraph().build().compile()
    result = runner.invoke(
        {
            "chat_id": "123",
            "channel_type": "telegram",
            "channel_target": "123",
            "last_user_message": "Hi",
            "correlation_id": "corr-chat-route",
            "_llm_client": llm,
        }
    )

    assert result.get("response_text") == "Hi!"
    assert "task_state" not in result
