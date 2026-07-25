from __future__ import annotations

import json
from typing import Any

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.intelligence.pdca.nodes import do_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_do_node_executes_latest_unexecuted_planned_call() -> None:
    task = TaskState()
    task.append_plan_call(_planned_call("plan-call-1", tool_id="tool-1", path="first.txt"))
    task.append_plan_call(_planned_call("plan-call-2", tool_id="tool-1", path="second.txt"))
    tools = _ToolRegistry(result={"written": True})

    do_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools))

    assert tools.calls == [("tool-1", {"path": "second.txt"})]
    calls = json.loads(task.plan_json)
    assert "execution" not in calls[0]
    assert calls[1]["execution"]["status"] == "success"
    assert calls[1]["execution"]["result"] == {"written": True}
    assert task.metadata["do_executed_since_last_act"] is True


def test_do_node_skips_already_executed_calls() -> None:
    task = TaskState()
    task.append_plan_call(_planned_call("plan-call-1", tool_id="tool-1", path="first.txt"))
    task.append_plan_call(_planned_call("plan-call-2", tool_id="tool-1", path="second.txt"))
    task.record_plan_call_success("plan-call-2", {"done": True})
    tools = _ToolRegistry(result={"written": True})

    do_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools))

    assert tools.calls == [("tool-1", {"path": "first.txt"})]
    calls = json.loads(task.plan_json)
    assert calls[0]["execution"]["status"] == "success"
    assert calls[1]["execution"]["status"] == "success"


def test_do_node_records_tool_exception() -> None:
    task = TaskState()
    task.append_plan_call(_planned_call("plan-call-1"))
    tools = _ToolRegistry(exception=RuntimeError("tool failed"))

    do_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools))

    execution = json.loads(task.plan_json)[0]["execution"]
    assert execution["status"] == "exception"
    assert execution["result"] is None
    assert execution["exception"] == "RuntimeError: tool failed"


def test_do_node_records_missing_registry_as_exception() -> None:
    task = TaskState()
    task.append_plan_call(_planned_call("plan-call-1"))

    do_node(task)

    execution = json.loads(task.plan_json)[0]["execution"]
    assert execution["status"] == "exception"
    assert "Tool registry is not available" in execution["exception"]


def test_do_node_without_unexecuted_call_does_not_execute() -> None:
    task = TaskState()
    tools = _ToolRegistry(result={"ok": True})

    do_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools))

    assert tools.calls == []
    assert task.plan_json == "- (none)"
    assert "Do found no unexecuted planned tool call" in task.updates_md


def test_do_node_does_not_judge_acceptance_criteria() -> None:
    task = TaskState(acceptance_criteria_md="1.- [ ] File exists", check_verdict="wip")
    task.append_plan_call(_planned_call("plan-call-1"))

    do_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry(result={"ok": True})))

    assert task.acceptance_criteria_md == "1.- [ ] File exists"
    assert task.check_verdict == "wip"


def test_silent_successful_bash_requires_model_authored_response_before_completion() -> None:
    task = TaskState(acceptance_criteria_md="1.- [x] TODO item is added", check_verdict="wip")
    task.append_plan_call(_planned_call("bash-call", tool_id="native.bash", path="ignored"))
    tools = _ToolRegistry(result={"exit_code": 0, "stdout": "", "stderr": ""})
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools)

    do_node(task, context=context)

    assert task.metadata["pending_silent_bash_confirmation"]["tool_call_id"] == "bash-call"

    from alphonse.agent_v2.core.intelligence.pdca.nodes import act_node
    act_node(task)
    assert task.metadata["act_route"] == "plan"
    assert task.status != "completed"

    task.append_plan_call(_planned_call("respond-call", tool_id="native.respond", path="ignored"))
    tools.result = {"message": "Añadí llevar lentes para el sol a la lista."}
    do_node(task, context=context)

    assert "pending_silent_bash_confirmation" not in task.metadata
    act_node(task)
    assert task.status == "completed"


def _planned_call(call_id: str, *, tool_id: str = "tool-1", path: str = "a.txt") -> dict[str, Any]:
    return {
        "id": call_id,
        "tool_id": tool_id,
        "tool_name": tool_id,
        "arguments": {"command": path} if tool_id == "native.bash" else {"path": path},
        "internal_state": "Writing the requested file.",
    }


class _ToolRegistry:
    def __init__(self, *, result: Any = None, exception: Exception | None = None) -> None:
        self.result = result
        self.exception = exception
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def register(self, tool: ToolDescriptor) -> None:
        _ = tool

    def get(self, name: str) -> ToolDescriptor | None:
        _ = name
        return None

    def list(self) -> tuple[ToolDescriptor, ...]:
        return (ToolDescriptor(tool_id="tool-1", name="write_file", kind=ToolKind.NATIVE),)

    def execute(self, tool_id: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((tool_id, dict(arguments)))
        if self.exception is not None:
            raise self.exception
        return self.result
