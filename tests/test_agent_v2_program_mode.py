from __future__ import annotations

import json
from typing import Any

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.programs import ProgramRunner
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService








def test_programmatic_parallel_call_rejects_non_read_only_tool() -> None:
    registry = _Registry()
    service = ToolInvocationService(context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry), task=TaskState())
    result = service.invoke("native.write", {}, parallel=True)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "tool_not_parallel_safe"






def test_docker_runner_reports_unavailability_without_local_fallback(monkeypatch) -> None:
    runner = ProgramRunner(docker_bin="missing-docker")
    monkeypatch.setattr(runner, "available", lambda: False)
    outcome = runner.run(source="async def main(tools):\n    return {}", invocation_service=object())
    assert outcome["status"] == "failed"
    assert outcome["error"]["code"] == "docker_unavailable"


def _program_task() -> TaskState:
    task = TaskState(goal="Program", acceptance_criteria_md="1.- [ ] Done")
    task.append_plan_call(
        {
            "id": "program-1",
            "execution_mode": "program",
            "program": {"language": "python", "source": "async def main(tools):\n    return {}"},
            "internal_state": "Running a bounded program.",
        }
    )
    return task


class _Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.descriptors = {
            "native.read": ToolDescriptor("native.read", "read", ToolKind.NATIVE, read_only=True),
            "native.write": ToolDescriptor("native.write", "write", ToolKind.NATIVE),
        }

    def get(self, tool_id: str) -> ToolDescriptor | None:
        return self.descriptors.get(tool_id)

    def list(self) -> tuple[ToolDescriptor, ...]:
        return tuple(self.descriptors.values())

    def execute(self, tool_id: str, arguments: dict[str, Any], execution_context=None) -> dict[str, Any]:
        _ = execution_context
        self.calls.append((tool_id, dict(arguments)))
        return {"value": arguments.get("value")}


class _Runner:
    def __init__(self, status: str) -> None:
        self.status = status

    def run(self, *, source: str, invocation_service: ToolInvocationService) -> dict[str, Any]:
        assert "async def main" in source
        if self.status == "waiting":
            return {"status": "waiting", "program_result": {"question": "q"}, "tool_calls": []}
        child = invocation_service.invoke("native.read", {"value": 42})
        return {"status": "success", "program_result": {"answer": 42}, "tool_calls": [child]}

    def available(self) -> bool:
        return True


class _UnavailableRunner:
    def available(self) -> bool:
        return False


def _program_value() -> dict[str, Any]:
    return {
        "execution_mode": "program",
        "program": {"language": "python", "source": "async def main(tools):\n    return {}"},
        "internal_state": "Compare options.",
    }
