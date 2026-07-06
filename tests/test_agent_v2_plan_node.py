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


def test_plan_node_stubbed_llm_leaves_plan_json_unchanged_and_executes_no_tools() -> None:
    existing_plan = '[{"id":"prior","tool_id":"tool-1","tool_name":"write_file","arguments":{},"internal_state":"Prior call."}]'
    task = TaskState(goal="Write the file", plan_json=existing_plan, acceptance_criteria_md="1.- [ ] File exists")
    tools = _ToolRegistry()

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=tools))

    assert task.plan_json == existing_plan
    assert tools.executed is False
    assert "planned_tool_call" not in task.metadata
    assert "Plan prepared one-tool-call prompt" in task.updates_md


def test_plan_node_records_valid_tool_call_when_llm_returns_result(monkeypatch) -> None:
    planned = {
        "id": "plan-call-1",
        "tool_id": "tool-1",
        "tool_name": "write_file",
        "arguments": {"path": "a.txt"},
        "internal_state": "Writing the requested file.",
    }
    task = TaskState(goal="Write the file", acceptance_criteria_md="1.- [ ] File exists")
    monkeypatch.setattr(plan_node_module, "_call_tool_planning_llm", lambda prompt: planned)

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()))

    assert task.metadata["tool_call_planning_llm_stubbed"] is False
    assert task.metadata["planned_tool_call"] == planned
    assert json.loads(task.plan_json) == [planned]


def test_plan_node_appends_multiple_tool_calls_without_replacing_existing(monkeypatch) -> None:
    existing = {
        "id": "plan-call-prior",
        "tool_id": "tool-1",
        "tool_name": "write_file",
        "arguments": {"path": "prior.txt"},
        "internal_state": "Prior call.",
    }
    planned = {
        "id": "plan-call-next",
        "tool_id": "tool-1",
        "tool_name": "write_file",
        "arguments": {"path": "next.txt"},
        "internal_state": "Adding the next file.",
    }
    task = TaskState(
        goal="Write the file",
        plan_json=json.dumps([existing]),
        acceptance_criteria_md="1.- [ ] File exists",
    )
    monkeypatch.setattr(plan_node_module, "_call_tool_planning_llm", lambda prompt: planned)

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()))

    assert json.loads(task.plan_json) == [existing, planned]


def test_plan_node_generates_missing_tool_call_id(monkeypatch) -> None:
    planned = {
        "tool_id": "tool-1",
        "tool_name": "write_file",
        "arguments": {"path": "a.txt"},
        "internal_state": "Writing the requested file.",
    }
    task = TaskState(goal="Write the file", acceptance_criteria_md="1.- [ ] File exists")
    monkeypatch.setattr(plan_node_module, "_call_tool_planning_llm", lambda prompt: planned)

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()))

    latest = task.metadata["planned_tool_call"]
    assert latest["id"].startswith("plan-call-")
    assert json.loads(task.plan_json)[0]["id"] == latest["id"]


def test_plan_node_requires_internal_state(monkeypatch) -> None:
    task = TaskState(goal="Write the file", acceptance_criteria_md="1.- [ ] File exists")
    monkeypatch.setattr(
        plan_node_module,
        "_call_tool_planning_llm",
        lambda prompt: {"tool_id": "tool-1", "tool_name": "write_file", "arguments": {}},
    )

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()))

    assert "planned_tool_call" not in task.metadata
    assert task.metadata["tool_call_planning_llm_stubbed"] is True
    assert task.plan_json == "- (none)"


def test_plan_node_caps_internal_state_at_256_chars(monkeypatch) -> None:
    task = TaskState(goal="Write the file", acceptance_criteria_md="1.- [ ] File exists")
    monkeypatch.setattr(
        plan_node_module,
        "_call_tool_planning_llm",
        lambda prompt: {
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {},
            "internal_state": "x" * 300,
        },
    )

    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_ToolRegistry()))

    latest = task.metadata["planned_tool_call"]
    assert len(latest["internal_state"]) == 256
    assert len(json.loads(task.plan_json)[0]["internal_state"]) == 256


def test_plan_node_ignores_invalid_tool_call_result(monkeypatch) -> None:
    task = TaskState(goal="Write the file", acceptance_criteria_md="1.- [ ] File exists")
    monkeypatch.setattr(
        plan_node_module,
        "_call_tool_planning_llm",
        lambda prompt: {
            "tool_id": "missing",
            "tool_name": "unknown",
            "arguments": {},
            "internal_state": "Trying an unavailable tool.",
        },
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
