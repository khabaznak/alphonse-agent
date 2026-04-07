from __future__ import annotations

import pytest

import alphonse.agent.cortex.task_mode.pdca as pdca_module
from alphonse.agent.cognition.providers.contracts import require_text_completion_provider
from alphonse.agent.cognition.providers.contracts import require_tool_calling_provider
from alphonse.agent.cortex.nodes.task_mode import task_mode_entry_node
from alphonse.agent.cortex.graph import check_node_state_adapter
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.tools.registry import build_default_tool_registry

_CURRENT_TEST_PROVIDER = None


def _set_current_test_provider(provider):
    global _CURRENT_TEST_PROVIDER
    _CURRENT_TEST_PROVIDER = provider
    return provider


@pytest.fixture(autouse=True)
def _patch_pdca_providers(monkeypatch: pytest.MonkeyPatch):
    global _CURRENT_TEST_PROVIDER
    _CURRENT_TEST_PROVIDER = None
    monkeypatch.setattr(
        pdca_module,
        "build_text_completion_provider",
        lambda: require_text_completion_provider(_CURRENT_TEST_PROVIDER, source="tests.test_task_mode_integration"),
    )
    monkeypatch.setattr(
        pdca_module,
        "build_tool_calling_provider",
        lambda: require_tool_calling_provider(_CURRENT_TEST_PROVIDER, source="tests.test_task_mode_integration"),
    )
    yield
    _CURRENT_TEST_PROVIDER = None


class _FakeLlm:
    def __init__(self, response: str) -> None:
        self._response = response
        _set_current_test_provider(self)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return self._response

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str = "auto",
    ) -> str:
        _ = (messages, tools, tool_choice)
        return self._response


def test_task_route_initializes_task_state() -> None:
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "last_user_message": "What can you do?",
        "correlation_id": "corr-task-entry",
    }
    result = task_mode_entry_node(state)
    task_state = result.get("task_state")
    assert isinstance(task_state, dict)
    assert task_state.get("mode") == "task"
    assert str(task_state.get("pdca_phase") or "") in {"plan", "check", "do", "act"}
    assert int(task_state.get("cycle_index") or 0) >= 0
    assert task_state.get("initialized") is True


def test_chat_route_now_starts_plan_first_even_for_greetings() -> None:
    llm = _FakeLlm(
        '{"kind":"plan","case_type":"new_request","reason":"baseline plan first",'
        '"confidence":0.8,"criteria_updates":[{"op":"append","text":"Maintain general conversation context"}],'
        '"evidence_refs":[],"failure_class":null}'
    )
    tool_registry = build_default_tool_registry()
    state = {
        "chat_id": "123",
        "channel_type": "telegram",
        "channel_target": "123",
        "last_user_message": "Hi",
        "correlation_id": "corr-chat-route",
        "_llm_client": llm,
    }
    state.update(task_mode_entry_node(state))
    state.update(check_node_state_adapter(state))
    task_state = state.get("task_state")
    assert isinstance(task_state, dict)
    verdict = task_state.get("judge_verdict")
    assert isinstance(verdict, dict)
    assert verdict.get("kind") == "plan"


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
    llm = _FakeLlm(
        '{"kind":"call_tool","tool_name":"jobs.list","args":{"limit":10},'
        '"acceptance_criteria":["Return the scheduled jobs list."]}'
    )
    state = {
        "correlation_id": "corr-clean-goal",
        "_llm_client": llm,
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
    task_state = state.get("task_state")
    assert isinstance(task_state, dict)
    out = next_step(state)
    goal = str(task_state.get("goal") or "")
    assert goal == "Schedule my daily FX update at 7am."
    assert "## RAW MESSAGE" not in goal
    assert not isinstance(out.get("pending_interaction"), dict)
