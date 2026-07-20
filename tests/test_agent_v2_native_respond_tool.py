from __future__ import annotations

import pytest

from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import _render_tool_call_plan_prompt
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry.native import RESPOND_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import RESPOND_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native import build_respond_tool_definition
from alphonse.agent_v2.core.tools.registry.native import execute_respond


def test_native_registry_registers_respond_tool() -> None:
    registry = build_native_tool_registry()

    descriptor = registry.get(RESPOND_TOOL_NAME)

    assert descriptor is not None
    assert descriptor.tool_id == RESPOND_TOOL_ID
    assert descriptor.name == RESPOND_TOOL_NAME
    assert descriptor.argument_schema["required"] == ["message"]
    assert descriptor.capabilities == ("conversation", "user_response")


def test_native_registry_exposes_native_tools() -> None:
    names = {descriptor.name for descriptor in build_native_tool_registry().list()}

    assert names == {"respond", "bash", "ask_question", "deliver_message", "scheduled_task"}


def test_respond_descriptor_and_schema_are_visible_to_plan_prompt() -> None:
    task = TaskState(goal="Say hello", acceptance_criteria_md="1.- [ ] User is greeted")
    descriptors = build_native_tool_registry().list()

    prompt = _render_tool_call_plan_prompt(task, descriptors)

    assert RESPOND_TOOL_ID in prompt
    assert RESPOND_TOOL_NAME in prompt
    assert "Use `native.respond` for greetings" in prompt
    assert "Use `native.bash` only" in prompt


def test_respond_tool_executes_normal_response() -> None:
    result = execute_respond({"message": "Hello, Alex.", "tone": "warm"})

    assert result == {"message": "Hello, Alex.", "tone": "warm"}


def test_respond_tool_defaults_tone() -> None:
    result = execute_respond({"message": "Hello."})

    assert result == {"message": "Hello.", "tone": "neutral"}


def test_respond_tool_rejects_blank_message() -> None:
    with pytest.raises(ValueError, match="respond_message_required"):
        execute_respond({"message": "   "})


def test_respond_tool_definition_executes_through_registry() -> None:
    registry = build_native_tool_registry()

    result = registry.execute(RESPOND_TOOL_ID, {"message": "Ready.", "tone": "concise"})

    assert result == {"message": "Ready.", "tone": "concise"}


def test_respond_tool_definition_can_be_built_directly() -> None:
    definition = build_respond_tool_definition()

    assert definition.descriptor.tool_id == RESPOND_TOOL_ID
    assert definition.argument_schema["properties"]["message"]["type"] == "string"
