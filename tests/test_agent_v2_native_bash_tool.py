from __future__ import annotations

import json

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.pdca.nodes import do_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import _render_tool_call_plan_prompt
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native import execute_bash
from alphonse.agent_v2.core.tools.registry.native.bash import MAX_OUTPUT_CHARS


def test_native_registry_registers_bash_tool() -> None:
    registry = build_native_tool_registry()

    descriptor = registry.get(BASH_TOOL_NAME)

    assert descriptor is not None
    assert descriptor.tool_id == BASH_TOOL_ID
    assert descriptor.name == BASH_TOOL_NAME
    assert descriptor.argument_schema["required"] == ["command"]
    assert descriptor.capabilities == ("shell", "local_execution")


def test_bash_descriptor_and_schema_are_visible_to_plan_prompt() -> None:
    task = TaskState(goal="List files", acceptance_criteria_md="1.- [ ] Files are listed")
    descriptor = build_native_tool_registry().list()[0]

    prompt = _render_tool_call_plan_prompt(task, (descriptor,))

    assert BASH_TOOL_ID in prompt
    assert BASH_TOOL_NAME in prompt
    assert "native" in prompt


def test_bash_tool_executes_successful_command_and_captures_stdout() -> None:
    result = execute_bash({"command": "printf hello"})

    assert result["exit_code"] == 0
    assert result["stdout"] == "hello"
    assert result["stderr"] == ""
    assert result["timed_out"] is False
    assert result["cwd"]


def test_bash_tool_captures_stderr_and_nonzero_exit_without_raising() -> None:
    result = execute_bash({"command": "printf problem >&2; exit 7"})

    assert result["exit_code"] == 7
    assert result["stdout"] == ""
    assert result["stderr"] == "problem"
    assert result["timed_out"] is False


def test_bash_tool_enforces_timeout() -> None:
    result = execute_bash({"command": "sleep 2", "timeout_seconds": 0.05})

    assert result["exit_code"] == -1
    assert result["timed_out"] is True
    assert "timed out" in result["stderr"]


def test_bash_tool_rejects_blank_command() -> None:
    with pytest.raises(ValueError, match="bash_command_required"):
        execute_bash({"command": "   "})


def test_bash_tool_truncates_large_output() -> None:
    result = execute_bash({"command": f"printf '%*s' {MAX_OUTPUT_CHARS + 10} '' | tr ' ' x"})

    assert len(result["stdout"]) < MAX_OUTPUT_CHARS + 40
    assert result["stdout"].endswith("... [truncated]")


def test_bash_tool_honors_optional_cwd(tmp_path) -> None:
    (tmp_path / "marker.txt").write_text("ok")

    result = execute_bash({"command": "pwd; ls marker.txt", "cwd": str(tmp_path)})

    assert result["exit_code"] == 0
    assert str(tmp_path) in result["stdout"]
    assert "marker.txt" in result["stdout"]
    assert result["cwd"] == str(tmp_path)


def test_do_node_executes_native_bash_and_records_result() -> None:
    task = TaskState(goal="Print greeting")
    task.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": BASH_TOOL_ID,
            "tool_name": BASH_TOOL_NAME,
            "arguments": {"command": "printf hello"},
            "internal_state": "Printing a greeting.",
        }
    )

    do_node(
        task,
        context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=build_native_tool_registry()),
    )

    execution = json.loads(task.plan_json)[0]["execution"]
    assert execution["status"] == "success"
    assert execution["result"]["exit_code"] == 0
    assert execution["result"]["stdout"] == "hello"


def test_do_node_records_native_bash_invalid_arguments_as_exception() -> None:
    task = TaskState(goal="Run bash")
    task.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": BASH_TOOL_ID,
            "tool_name": BASH_TOOL_NAME,
            "arguments": {"command": ""},
            "internal_state": "Trying bash.",
        }
    )

    do_node(
        task,
        context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=build_native_tool_registry()),
    )

    execution = json.loads(task.plan_json)[0]["execution"]
    assert execution["status"] == "exception"
    assert "bash_command_required" in execution["exception"]


def test_bash_tool_definition_executes_through_registry() -> None:
    registry = build_native_tool_registry()

    result = registry.execute(BASH_TOOL_ID, {"command": "printf registry"})

    assert result["stdout"] == "registry"
    assert result["timed_out"] is False


def test_bash_tool_definition_can_be_built_directly() -> None:
    definition = build_bash_tool_definition()

    assert definition.descriptor.tool_id == BASH_TOOL_ID
    assert definition.argument_schema["properties"]["command"]["type"] == "string"
