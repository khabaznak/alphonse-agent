from __future__ import annotations

import pytest

import alphonse.agent.cortex.task_mode.pdca as pdca_module
from alphonse.agent.cognition.providers.contracts import require_text_completion_provider
from alphonse.agent.cognition.providers.contracts import require_tool_calling_provider
from alphonse.agent.cortex.graph import check_node_state_adapter
from alphonse.agent.cortex.graph import task_record_entry_node
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
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
    ) -> dict[str, object]:
        _ = (messages, tools, tool_choice)
        return {"tool_call": {"kind": "call_tool", "tool_name": "jobs.list", "args": {"limit": 10}}}


def _task_record_state(
    *,
    goal: str,
    correlation_id: str = "",
    recent_conversation_md: str = "- (none)",
) -> dict[str, object]:
    return {
        "task_record": TaskRecord(
            goal=goal,
            correlation_id=correlation_id,
            recent_conversation_md=recent_conversation_md,
            status="running",
        )
    }


def test_task_route_initializes_task_record() -> None:
    state = _task_record_state(goal="What can you do?", correlation_id="corr-task-entry")
    result = task_record_entry_node(state)
    task_record = result.get("task_record")
    assert isinstance(task_record, TaskRecord)
    assert task_record.goal == "What can you do?"
    assert task_record.correlation_id == "corr-task-entry"
    assert result.get("check_provenance") == "entry"


def test_chat_route_now_starts_check_first_even_for_greetings() -> None:
    _FakeLlm(
        '{"kind":"plan","case_type":"new_request","reason":"baseline plan first",'
        '"confidence":0.8,"criteria_updates":[{"op":"append","text":"Maintain general conversation context"}],'
        '"evidence_refs":[],"failure_class":null}'
    )
    state = _task_record_state(goal="Hi", correlation_id="corr-chat-route")
    state.update(task_record_entry_node(state))
    state.update(check_node_state_adapter(state))
    check_result = state.get("check_result")
    assert isinstance(check_result, dict)
    assert check_result.get("verdict") == "plan"


def test_task_mode_entry_extracts_goal_from_incoming_payload_not_raw_blob() -> None:
    state = _task_record_state(
        goal="Please set a recurring USD to MXN reminder at 7am.",
    )

    update = task_record_entry_node(state)
    task_record = update.get("task_record")
    assert isinstance(task_record, TaskRecord)
    assert task_record.goal == "Please set a recurring USD to MXN reminder at 7am."


def test_planner_uses_clean_goal_text_from_task_record() -> None:
    next_step = build_next_step_node(tool_registry=build_default_tool_registry())
    _FakeLlm('unused')
    state = _task_record_state(
        goal="Schedule my daily FX update at 7am.",
        correlation_id="corr-clean-goal",
    )
    state.update(task_record_entry_node(state))
    task_record = state.get("task_record")
    assert isinstance(task_record, TaskRecord)
    out = next_step(task_record)
    assert task_record.goal == "Schedule my daily FX update at 7am."
    assert "## RAW MESSAGE" not in task_record.goal
    assert isinstance(out, dict)
