from __future__ import annotations

from typing import Any

import alphonse.agent.cortex.task_mode.execute_step as execute_step_module
from alphonse.agent.cortex.task_mode.execute_step import execute_step_node_impl
from alphonse.agent.tools.registry import ToolRegistry


class _EchoTool:
    def execute(self, *, value: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
        _ = state
        return {
            "status": "ok",
            "result": {"value": value},
            "error": None,
            "metadata": {"tool": "echo_tool"},
        }


def _task_state_with_defaults(state: dict[str, Any]) -> dict[str, Any]:
    task_state = state.get("task_state")
    assert isinstance(task_state, dict)
    return task_state


def _current_step(task_state: dict[str, Any]) -> dict[str, Any] | None:
    plan = task_state.get("plan")
    if not isinstance(plan, dict):
        return None
    step_id = str(plan.get("current_step_id") or "").strip()
    for step in plan.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if str(step.get("step_id") or "").strip() == step_id:
            return step
    return None


def test_execute_step_emits_memory_hooks_after_tool_call(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _capture_memory(**kwargs: Any) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(execute_step_module, "_memory_append_episode", _capture_memory)
    monkeypatch.setattr(execute_step_module, "_memory_artifacts_from_result", lambda **_: [])

    registry = ToolRegistry()
    registry.register("echo_tool", _EchoTool())
    state: dict[str, Any] = {
        "correlation_id": "corr-memory-hook-1",
        "incoming_user_id": "alex",
        "task_state": {
            "status": "running",
            "goal": "echo data",
            "plan": {
                "current_step_id": "step_1",
                "steps": [
                    {
                        "step_id": "step_1",
                        "status": "proposed",
                        "proposal": {
                            "kind": "call_tool",
                            "tool_name": "echo_tool",
                            "args": {"value": "hello"},
                        },
                    }
                ],
            },
        },
    }

    updated = execute_step_node_impl(
        state,
        tool_registry=registry,
        task_state_with_defaults=_task_state_with_defaults,
        correlation_id=lambda s: str(s.get("correlation_id") or ""),
        current_step=_current_step,
        append_trace_event=lambda *_args, **_kwargs: None,
        logger=execute_step_module.logging.getLogger("test.execute_step.memory"),
        log_task_event=lambda **_kwargs: None,
    )
    assert isinstance(updated.get("task_state"), dict)
    event_types = [str(item.get("event_type") or "") for item in calls]
    assert "after_tool_call" in event_types
    assert "plan_step_completed" in event_types


def test_execute_finish_emits_plan_step_completion_memory(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _capture_memory(**kwargs: Any) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(execute_step_module, "_memory_append_episode", _capture_memory)

    state: dict[str, Any] = {
        "correlation_id": "corr-memory-hook-2",
        "incoming_user_id": "alex",
        "task_state": {
            "status": "running",
            "goal": "finish task",
            "plan": {
                "current_step_id": "step_2",
                "steps": [
                    {
                        "step_id": "step_2",
                        "status": "proposed",
                        "proposal": {
                            "kind": "finish",
                            "final_text": "done",
                        },
                    }
                ],
            },
        },
    }
    updated = execute_step_node_impl(
        state,
        tool_registry=ToolRegistry(),
        task_state_with_defaults=_task_state_with_defaults,
        correlation_id=lambda s: str(s.get("correlation_id") or ""),
        current_step=_current_step,
        append_trace_event=lambda *_args, **_kwargs: None,
        logger=execute_step_module.logging.getLogger("test.execute_step.memory"),
        log_task_event=lambda **_kwargs: None,
    )
    assert isinstance(updated.get("task_state"), dict)
    assert any(str(item.get("event_type") or "") == "plan_step_completed" for item in calls)
