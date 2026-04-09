from __future__ import annotations

import logging
from typing import Any

import alphonse.agent.cortex.task_mode.execute_step as execute_step_module
from alphonse.agent.cortex.task_mode.execute_step import execute_step_node_impl
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.tools.base import ToolDefinition
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.agent.tools.spec import ToolSpec


class _EchoTool:
    def execute(self, *, value: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
        _ = state
        return {
            "output": {"value": value},
            "exception": None,
            "metadata": {"tool": "echo_tool"},
        }


class _InternalMessageTool:
    def execute(self, *, message: str, recipient: str, state: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        _ = state
        _ = recipient
        return {
            "output": {"sent": True, "message": message},
            "exception": None,
            "metadata": {"tool": "communication.send_message"},
        }


class _AskQuestionTool:
    def execute(self, *, question: str, state: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        _ = state
        _ = question
        return {
            "output": {"queued": True},
            "exception": None,
            "metadata": {"tool": "askQuestion"},
        }


def _register_tool(registry: ToolRegistry, key: str, executor: object) -> None:
    spec = ToolSpec(
        canonical_name=key,
        summary=f"{key} summary",
        description=f"{key} description",
        input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        output_schema={"type": "object", "additionalProperties": True},
    )
    registry.register(ToolDefinition(spec=spec, executor=executor))  # type: ignore[arg-type]


def test_execute_step_emits_memory_hooks_after_tool_call(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(execute_step_module, "record_after_tool_call", lambda **_: calls.append("after_tool_call"))
    monkeypatch.setattr(execute_step_module, "record_plan_step_completion", lambda **_: calls.append("plan_step_completed"))

    registry = ToolRegistry()
    _register_tool(registry, "echo_tool", _EchoTool())
    monkeypatch.setattr(execute_step_module, "_tool_registry", lambda: registry)
    task_record = TaskRecord(
        task_id="task-memory-hook-1",
        user_id="alex",
        correlation_id="corr-memory-hook-1",
        goal="echo data",
    )

    updated = execute_step_node_impl(
        task_record,
        {
            "tool_call": {"kind": "call_tool", "tool_name": "echo_tool", "args": {"value": "hello"}},
            "planner_intent": "Echo the provided data.",
        },
        logger=logging.getLogger("test.execute_step.memory"),
        log_task_event=lambda **_kwargs: None,
    )
    assert updated["provenance"] == "do"
    assert isinstance(updated.get("task_record"), TaskRecord)
    assert "echo_tool" in updated["task_record"].tool_call_history_md
    assert "after_tool_call" in calls
    assert "plan_step_completed" in calls


def test_execute_ask_question_tool_emits_plan_step_completion_memory(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(execute_step_module, "record_after_tool_call", lambda **_: calls.append("after_tool_call"))
    monkeypatch.setattr(execute_step_module, "record_plan_step_completion", lambda **_: calls.append("plan_step_completed"))

    registry = ToolRegistry()
    _register_tool(registry, "askQuestion", _AskQuestionTool())
    monkeypatch.setattr(execute_step_module, "_tool_registry", lambda: registry)
    task_record = TaskRecord(
        task_id="task-memory-hook-2",
        user_id="alex",
        correlation_id="corr-memory-hook-2",
        goal="finish task",
    )
    updated = execute_step_node_impl(
        task_record,
        {
            "tool_call": {"kind": "call_tool", "tool_name": "askQuestion", "args": {"question": "Can you clarify?"}},
            "planner_intent": "Ask the user for clarification.",
        },
        logger=logging.getLogger("test.execute_step.memory"),
        log_task_event=lambda **_kwargs: None,
    )
    assert updated["provenance"] == "do"
    assert "askQuestion" in updated["task_record"].tool_call_history_md
    assert "plan_step_completed" in calls


def test_send_message_call_writes_facts_and_memory(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(execute_step_module, "record_after_tool_call", lambda **_: calls.append("after_tool_call"))
    monkeypatch.setattr(execute_step_module, "record_plan_step_completion", lambda **_: calls.append("plan_step_completed"))

    registry = ToolRegistry()
    _register_tool(registry, "communication.send_message", _InternalMessageTool())
    monkeypatch.setattr(execute_step_module, "_tool_registry", lambda: registry)
    task_record = TaskRecord(
        task_id="task-memory-hook-internal",
        user_id="alex",
        correlation_id="corr-memory-hook-internal",
        goal="internal progress",
    )
    updated = execute_step_node_impl(
        task_record,
        {
            "tool_call": {
                "kind": "call_tool",
                "tool_name": "communication.send_message",
                "args": {
                    "message": "Estoy avanzando",
                    "recipient": "8553589429",
                    "internal_progress": True,
                },
            },
            "planner_intent": "Send an internal progress update.",
        },
        logger=logging.getLogger("test.execute_step.memory"),
        log_task_event=lambda **_kwargs: None,
    )
    assert "communication.send_message" in updated["task_record"].tool_call_history_md
    assert "after_tool_call" in calls
    assert "plan_step_completed" in calls
