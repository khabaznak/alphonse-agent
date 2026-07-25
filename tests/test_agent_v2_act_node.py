from __future__ import annotations

from importlib import import_module

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.intelligence.pdca.nodes import act_node
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.intelligence.task_state import TaskState

act_node_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.act_node")


def test_act_node_new_verdict_renders_acceptance_criteria_prompt_and_stubs_llm() -> None:
    task = TaskState(goal="Ask Gaby about coffee", user="Alex", check_verdict="new")

    result = act_node(task)

    assert result is task
    assert task.metadata["acceptance_criteria_llm_stubbed"] is True
    assert task.metadata["acceptance_criteria_updated"] is False
    assert task.metadata["act_route"] == "end"
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
    assert task.metadata["act_route"] == "plan"
    assert "Act updated acceptance criteria" in task.updates_md


def test_act_node_uses_inference_without_tools_for_acceptance_criteria() -> None:
    provider = StubInferenceProvider(
        markdown_by_purpose={InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] File exists"}
    )
    router = InferenceRouter(
        provider=provider,
        default_profile=ModelProfile(provider="openai", model="gpt", profile_id="default"),
    )
    task = TaskState(goal="Create a file", project_id="alpha", user="alex", check_verdict="new")

    act_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), inference=router))

    assert task.acceptance_criteria_md == "1.- [ ] File exists"
    assert task.metadata["acceptance_criteria_llm_stubbed"] is False
    assert task.metadata["acceptance_criteria_model_profile"] == "default"
    assert provider.requests[0].purpose == InferencePurpose.ACCEPTANCE_CRITERIA
    assert provider.requests[0].tools == ()


def test_act_node_unsupported_verdict_does_not_render_acceptance_criteria_prompt() -> None:
    task = TaskState(goal="Continue task", check_verdict="wip")

    act_node(task)

    assert "acceptance_criteria_prompt" not in task.metadata
    assert "acceptance_criteria_llm_stubbed" not in task.metadata
    assert task.metadata["act_route"] == "end"
    assert "cannot continue work-in-progress" in task.updates_md


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


def test_act_node_routes_wip_with_acceptance_criteria_to_plan_before_cycle_limit() -> None:
    task = TaskState(goal="Continue task", check_verdict="wip", acceptance_criteria_md="1.- [ ] Done")

    act_node(task)

    assert task.metadata["act_route"] == "plan"
    assert task.check_verdict == "wip"
    assert "routed work-in-progress task back to Plan" in task.updates_md


def test_act_node_wip_with_completed_acceptance_criteria_routes_success_to_end() -> None:
    task = TaskState(goal="Continue task", check_verdict="wip", acceptance_criteria_md="1.- [x] Done")

    act_node(task)

    assert task.check_verdict == "mission_success"
    assert task.status == "completed"
    assert task.outcome == {"status": "success", "reason": "All acceptance criteria are complete."}
    assert task.metadata["act_route"] == "end"
    assert task.metadata["act_terminal_decision"] == "mission_success"
    assert "mission success" in task.updates_md


def test_act_node_does_not_complete_until_silent_bash_confirmation_is_sent() -> None:
    task = TaskState(
        goal="Add sunglasses",
        check_verdict="wip",
        acceptance_criteria_md="1.- [x] TODO item is added",
        metadata={"pending_silent_bash_confirmation": {"tool_call_id": "bash-call"}},
    )

    act_node(task)

    assert task.metadata["act_route"] == "plan"
    assert task.status != "completed"
    assert "requires a native.respond confirmation" in task.updates_md


def test_act_node_wip_explicit_cancel_routes_failed_to_end() -> None:
    task = TaskState(
        goal="Continue task",
        check_verdict="wip",
        acceptance_criteria_md="1.- [ ] Done",
        metadata={"cancel_requested": True, "failure_reason": "User cancelled the task."},
    )

    act_node(task)

    assert task.check_verdict == "mission_failed"
    assert task.status == "failed"
    assert task.outcome == {"status": "failed", "reason": "User cancelled the task."}
    assert task.metadata["act_route"] == "end"
    assert task.metadata["act_terminal_decision"] == "mission_failed"


def test_act_node_wip_explicit_failure_metadata_routes_failed_to_end() -> None:
    task = TaskState(
        goal="Continue task",
        check_verdict="wip",
        acceptance_criteria_md="1.- [ ] Done",
        metadata={"mission_failed": True},
    )

    act_node(task)

    assert task.check_verdict == "mission_failed"
    assert task.outcome == {"status": "failed", "reason": "Mission failure was explicitly signaled."}


def test_act_node_wip_repeated_tool_exceptions_route_failed_to_end() -> None:
    task = TaskState(goal="Continue task", check_verdict="wip", acceptance_criteria_md="1.- [ ] Done")
    for index in range(3):
        task.append_plan_call(
            {
                "id": f"plan-call-{index}",
                "tool_id": "tool-1",
                "tool_name": "write_file",
                "arguments": {},
                "internal_state": "Writing.",
            }
        )
        task.record_plan_call_exception(f"plan-call-{index}", RuntimeError("failed"))

    act_node(task)

    assert task.check_verdict == "mission_failed"
    assert task.status == "failed"
    assert "3 exceptions" in task.outcome["reason"]


def test_act_node_wip_terminal_success_overrides_temporary_cycle_limit() -> None:
    task = TaskState(
        goal="Continue task",
        check_verdict="wip",
        acceptance_criteria_md="1.- [x] Done",
        pdca_cycle_count=10,
        metadata={"do_executed_since_last_act": True, "tool_call_planning_llm_stubbed": True},
    )

    act_node(task)

    assert task.check_verdict == "mission_success"
    assert task.metadata["act_terminal_decision"] == "mission_success"
    assert task.metadata.get("act_stop_reason") != "temporary_cycle_limit"


def test_act_node_wip_terminal_failure_overrides_temporary_cycle_limit() -> None:
    task = TaskState(
        goal="Continue task",
        check_verdict="wip",
        acceptance_criteria_md="1.- [ ] Done",
        pdca_cycle_count=10,
        metadata={"failure_reason": "No more attempts should be made.", "tool_call_planning_llm_stubbed": True},
    )

    act_node(task)

    assert task.check_verdict == "mission_failed"
    assert task.metadata["act_terminal_decision"] == "mission_failed"
    assert task.metadata.get("act_stop_reason") != "temporary_cycle_limit"


def test_act_node_observes_completed_do_cycle_and_continues_before_temporary_limit() -> None:
    task = TaskState(
        goal="Continue task",
        check_verdict="wip",
        acceptance_criteria_md="1.- [ ] Done",
        metadata={"do_executed_since_last_act": True},
    )

    act_node(task)

    assert task.pdca_cycle_count == 1
    assert task.metadata["completed_capd_cycle_count"] == 1
    assert task.metadata["act_route"] == "plan"
    assert "routed work-in-progress task back to Plan" in task.updates_md


def test_act_node_observes_completed_do_cycle_and_stops_at_temporary_limit() -> None:
    task = TaskState(
        goal="Continue task",
        check_verdict="wip",
        acceptance_criteria_md="1.- [ ] Done",
        pdca_cycle_count=9,
        metadata={"do_executed_since_last_act": True},
    )

    act_node(task)

    assert task.pdca_cycle_count == 10
    assert task.metadata["completed_capd_cycle_count"] == 10
    assert task.metadata["act_route"] == "end"
    assert task.metadata["act_stop_reason"] == "temporary_cycle_limit"
    assert "temporary completed-cycle limit" in task.updates_md


def test_act_node_stops_after_stubbed_planning_pass() -> None:
    task = TaskState(
        goal="Continue task",
        check_verdict="wip",
        acceptance_criteria_md="1.- [ ] Done",
        metadata={"tool_call_planning_llm_stubbed": True},
    )

    act_node(task)

    assert task.metadata["act_route"] == "end"
    assert task.metadata["act_stop_reason"] == "temporary_cycle_limit"
