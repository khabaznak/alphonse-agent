from __future__ import annotations

from alphonse.agent.cortex.nodes.task_mode import task_mode_entry_node
from alphonse.agent.cortex.graph import CortexGraph
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.tools.registry import build_default_tool_registry


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


def test_task_mode_entry_extracts_goal_from_incoming_payload_not_raw_blob() -> None:
    state = {
        "incoming_raw_message": {
            "text": "Please set a recurring USD to MXN reminder at 7am.",
            "provider_event": {
                "message": {
                    "text": "Please set a recurring USD to MXN reminder at 7am.",
                }
            },
        },
        "last_user_message": (
            "## RAW MESSAGE\n\n"
            "- channel: telegram\n\n"
            "## RAW JSON\n\n"
            "```json\n"
            "{\"message\":{\"text\":\"Please set a recurring USD to MXN reminder at 7am.\"}}\n"
            "```"
        ),
    }

    update = task_mode_entry_node(state)
    task_state = update.get("task_state")
    assert isinstance(task_state, dict)
    assert task_state.get("goal") == "Please set a recurring USD to MXN reminder at 7am."


def test_acceptance_criteria_pending_context_uses_clean_goal_text() -> None:
    tool_registry = build_default_tool_registry()
    next_step = build_next_step_node(tool_registry=tool_registry)
    state = {
        "correlation_id": "corr-clean-goal",
        "incoming_raw_message": {
            "text": "Schedule my daily FX update at 7am.",
            "provider_event": {
                "message": {"text": "Schedule my daily FX update at 7am."}
            },
        },
        "last_user_message": (
            "## RAW MESSAGE\n\n"
            "```json\n"
            "{\"message\":{\"text\":\"Schedule my daily FX update at 7am.\"}}\n"
            "```"
        ),
    }
    state.update(task_mode_entry_node(state))
    out = next_step(state)
    pending = out.get("pending_interaction")
    assert isinstance(pending, dict)
    context = pending.get("context")
    assert isinstance(context, dict)
    goal = str(context.get("goal") or "")
    assert goal == "Schedule my daily FX update at 7am."
    assert "## RAW MESSAGE" not in goal
