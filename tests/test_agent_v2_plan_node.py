from __future__ import annotations

import json
from importlib import import_module

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.intelligence.pdca.nodes import plan_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue

plan_node_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node")


def test_plan_node_renders_tool_call_prompt_with_available_tools() -> None:
    task = TaskState(goal="Write the file", acceptance_criteria_md="1.- [ ] File exists")

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()))

    prompt = task.metadata["tool_call_plan_prompt"]
    assert "exactly one next tool call" in prompt
    assert "write_file" in prompt
    assert "tool-1" in prompt
    assert "1.- [ ] File exists" in prompt
    assert task.metadata["tool_call_planning_llm_stubbed"] is True


def test_plan_node_stubbed_llm_leaves_plan_md_unchanged_and_executes_no_tools() -> None:
    task = TaskState(goal="Write the file", plan_md="- existing", acceptance_criteria_md="1.- [ ] File exists")
    tools = _ToolRegistry()

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools))

    assert task.plan_md == "- existing"
    assert tools.executed is False
    assert "planned_tool_call" not in task.metadata
    assert "Plan prepared one-tool-call prompt" in task.updates_md


def test_plan_node_records_valid_tool_call_when_llm_returns_result(monkeypatch) -> None:
    planned = {"tool_id": "tool-1", "tool_name": "write_file", "arguments": {"path": "a.txt"}}
    task = TaskState(goal="Write the file", acceptance_criteria_md="1.- [ ] File exists")
    monkeypatch.setattr(plan_node_module, "_call_tool_planning_llm", lambda prompt: planned)

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()))

    assert task.metadata["tool_call_planning_llm_stubbed"] is False
    assert task.metadata["planned_tool_call"] == planned
    assert json.loads(task.plan_md) == planned


def test_plan_node_ignores_invalid_tool_call_result(monkeypatch) -> None:
    task = TaskState(goal="Write the file", acceptance_criteria_md="1.- [ ] File exists")
    monkeypatch.setattr(
        plan_node_module,
        "_call_tool_planning_llm",
        lambda prompt: {"tool_id": "missing", "tool_name": "unknown", "arguments": {}},
    )

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()))

    assert "planned_tool_call" not in task.metadata
    assert task.metadata["tool_call_planning_llm_stubbed"] is True


class _ToolRegistry:
    def __init__(self) -> None:
        self.executed = False

    def list(self) -> tuple[ToolDescriptor, ...]:
        return (
            ToolDescriptor(
                tool_id="tool-1",
                name="write_file",
                kind=ToolKind.NATIVE,
                description="Writes files",
            ),
        )
