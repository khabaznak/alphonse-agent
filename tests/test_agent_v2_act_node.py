from __future__ import annotations

from importlib import import_module

from alphonse.agent_v2.core.intelligence.pdca.nodes import act_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState

act_node_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.act_node")


def test_act_node_new_verdict_renders_acceptance_criteria_prompt_and_stubs_llm() -> None:
    task = TaskState(goal="Ask Gaby about coffee", user="Alex", check_verdict="new")

    result = act_node(task)

    assert result is task
    assert task.metadata["acceptance_criteria_llm_stubbed"] is True
    assert task.metadata["acceptance_criteria_updated"] is False
    assert "acceptance_criteria_prompt" in task.metadata
    assert "1.- [ ] The required outcome is true" in task.metadata["acceptance_criteria_prompt"]
    assert "Ask Gaby about coffee" in task.metadata["acceptance_criteria_prompt"]
    assert "Act prepared acceptance criteria" in task.updates_md


def test_act_node_steer_verdict_prompt_includes_existing_acceptance_criteria() -> None:
    task = TaskState(
        goal="Ask Gaby about coffee",
        user="Alex",
        check_verdict="steer",
        check_reason="Gaby answered.",
        acceptance_criteria_md='1.- [ ] Alex has confirmation from Gaby if she wanted coffee',
    )

    act_node(task)

    prompt = task.metadata["acceptance_criteria_prompt"]
    assert "steer" in prompt
    assert "Gaby answered." in prompt
    assert "1.- [ ] Alex has confirmation from Gaby if she wanted coffee" in prompt


def test_act_node_does_not_mutate_acceptance_criteria() -> None:
    original = '1.- [ ] Alex has confirmation from Gaby if she wanted coffee'
    task = TaskState(goal="Ask Gaby", check_verdict="new", acceptance_criteria_md=original)

    act_node(task)

    assert task.acceptance_criteria_md == original


def test_act_node_updates_acceptance_criteria_when_llm_stub_returns_result(monkeypatch) -> None:
    generated = "1.- [ ] Alex has confirmation from Gaby if she wanted coffee"
    task = TaskState(goal="Ask Gaby", check_verdict="new")

    monkeypatch.setattr(act_node_module, "_call_acceptance_criteria_llm", lambda prompt: generated)

    act_node(task)

    assert task.acceptance_criteria_md == generated
    assert task.metadata["acceptance_criteria_updated"] is True
    assert "Act updated acceptance criteria" in task.updates_md


def test_act_node_unsupported_verdict_does_not_render_acceptance_criteria_prompt() -> None:
    task = TaskState(goal="Continue task", check_verdict="wip")

    act_node(task)

    assert "acceptance_criteria_prompt" not in task.metadata
    assert "acceptance_criteria_llm_stubbed" not in task.metadata
    assert "no implemented action" in task.updates_md


def test_act_node_prompt_includes_checkbox_instructions_and_task_state_context() -> None:
    task = TaskState(goal="Create a file", user="Alex", check_verdict="new")
    task.append_conversation_message("Alex", "Create a file")

    act_node(task)

    prompt = task.metadata["acceptance_criteria_prompt"]
    assert "numbered markdown checkbox rows" in prompt
    assert "1.- [ ]" in prompt
    assert "2.- [x]" in prompt
    assert '# Task State' in prompt
    assert '- Alex: "Create a file"' in prompt
