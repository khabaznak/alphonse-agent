"""Act node for the v2 PDCA graph."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRequest
from alphonse.agent_v2.core.intelligence.task_state import TaskState

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
_ACCEPTANCE_CRITERIA_ACTION_VERDICTS = {"new", "steer"}
_PLAN_ROUTE = "plan"
_END_ROUTE = "end"
_TEMPORARY_MAX_COMPLETED_CYCLES = 1
_PLAN_CALL_EXCEPTION_FAILURE_THRESHOLD = 3


def act_node(task: TaskState, context: CoreLoopContext | None = None) -> TaskState:
    """Act on the check verdict without re-checking or executing work."""
    _observe_completed_do_cycle(task)
    verdict = str(task.check_verdict or "").strip().lower()
    if verdict == "wip":
        return _act_on_wip(task)

    if _temporary_cycle_limit_reached(task):
        task.metadata["act_route"] = _END_ROUTE
        task.metadata["act_stop_reason"] = "temporary_cycle_limit"
        task.append_update("Act stopped the CAPD cycle at the temporary completed-cycle limit.")
        return task

    if verdict in _ACCEPTANCE_CRITERIA_ACTION_VERDICTS:
        prompt = _render_acceptance_criteria_prompt(task)
        generated_criteria = _call_acceptance_criteria_inference(prompt, task, context)
        task.metadata["acceptance_criteria_prompt"] = prompt
        task.metadata["acceptance_criteria_llm_stubbed"] = context is None or context.inference is None
        if generated_criteria:
            task.acceptance_criteria_md = str(generated_criteria).strip()
            task.metadata["acceptance_criteria_updated"] = True
            task.append_update("Act updated acceptance criteria from LLM result.")
        else:
            task.metadata["acceptance_criteria_updated"] = False
            task.append_update(
                "Act prepared acceptance criteria generation/revision prompt; LLM execution is stubbed."
            )
        task.metadata["act_route"] = _PLAN_ROUTE if _markdown_has_acceptance_criteria(task.acceptance_criteria_md) else _END_ROUTE
        return task

    if verdict in {"mission_success", "mission_failed"}:
        task.metadata["act_route"] = _END_ROUTE
        task.append_update(f"Act routed terminal check verdict to end: {verdict}.")
        return task

    task.metadata["act_route"] = _PLAN_ROUTE if _markdown_has_acceptance_criteria(task.acceptance_criteria_md) else _END_ROUTE
    task.append_update(f"Act has no implemented action for check verdict: {verdict or 'none'}.")
    return task


def _act_on_wip(task: TaskState) -> TaskState:
    if task.acceptance_criteria_all_complete():
        return _mark_mission_success(task, "All acceptance criteria are complete.")

    failure_reason = _mission_failure_reason(task)
    if failure_reason:
        return _mark_mission_failed(task, failure_reason)

    if _temporary_cycle_limit_reached(task):
        task.metadata["act_route"] = _END_ROUTE
        task.metadata["act_stop_reason"] = "temporary_cycle_limit"
        task.append_update("Act stopped the CAPD cycle at the temporary completed-cycle limit.")
        return task

    if _markdown_has_acceptance_criteria(task.acceptance_criteria_md):
        task.metadata["act_route"] = _PLAN_ROUTE
        task.append_update("Act routed work-in-progress task back to Plan.")
        return task

    task.metadata["act_route"] = _END_ROUTE
    task.append_update("Act cannot continue work-in-progress task without acceptance criteria.")
    return task


def _mark_mission_success(task: TaskState, reason: str) -> TaskState:
    task.set_check_result(
        verdict="mission_success",
        reason=reason,
        confidence=1.0,
        evidence_refs=list(task.check_evidence_refs or []),
        new_message_count=task.check_new_message_count,
    )
    task.status = "completed"
    task.outcome = {"status": "success", "reason": reason}
    task.metadata["act_route"] = _END_ROUTE
    task.metadata["act_terminal_decision"] = "mission_success"
    task.append_update("Act marked the task as mission success.")
    return task


def _mark_mission_failed(task: TaskState, reason: str) -> TaskState:
    task.set_check_result(
        verdict="mission_failed",
        reason=reason,
        confidence=1.0,
        evidence_refs=list(task.check_evidence_refs or []),
        new_message_count=task.check_new_message_count,
    )
    task.status = "failed"
    task.outcome = {"status": "failed", "reason": reason}
    task.metadata["act_route"] = _END_ROUTE
    task.metadata["act_terminal_decision"] = "mission_failed"
    task.append_update("Act marked the task as mission failed.")
    return task


def _mission_failure_reason(task: TaskState) -> str:
    if task.metadata.get("cancel_requested") is True:
        return str(task.metadata.get("failure_reason") or "Task was cancelled.")
    if task.metadata.get("mission_failed") is True:
        return str(task.metadata.get("failure_reason") or "Mission failure was explicitly signaled.")
    failure_reason = str(task.metadata.get("failure_reason") or "").strip()
    if failure_reason:
        return failure_reason
    if task.count_plan_call_exceptions() >= _PLAN_CALL_EXCEPTION_FAILURE_THRESHOLD:
        return f"Planned tool calls reached {_PLAN_CALL_EXCEPTION_FAILURE_THRESHOLD} exceptions."
    return ""


def _render_acceptance_criteria_prompt(task: TaskState) -> str:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("acceptance_criteria_prompt.j2")
    return template.render(
        check_verdict=task.check_verdict or "",
        check_reason=task.check_reason,
        existing_acceptance_criteria_md=task.acceptance_criteria_md,
        task_state_md=task.to_markdown_prompt(),
    ).strip()


def _call_acceptance_criteria_llm(prompt: str) -> str | None:
    """Stub for the future acceptance criteria LLM call."""
    _ = prompt
    return None


def _call_acceptance_criteria_inference(
    prompt: str,
    task: TaskState,
    context: CoreLoopContext | None = None,
) -> str | None:
    if context is not None and context.inference is not None:
        result = context.inference.generate_markdown(
            InferenceRequest(
                prompt=prompt,
                purpose=InferencePurpose.ACCEPTANCE_CRITERIA,
                project_id=task.project_id,
                user=task.user,
                task_id=task.task_id,
            )
        )
        if result.model_profile is not None:
            task.metadata["acceptance_criteria_model_profile"] = result.model_profile.profile_id
        return str(result.content or "").strip() or None
    return _call_acceptance_criteria_llm(prompt)


def _observe_completed_do_cycle(task: TaskState) -> None:
    if task.metadata.get("do_executed_since_last_act") is not True:
        return
    task.pdca_cycle_count = max(0, int(task.pdca_cycle_count or 0)) + 1
    task.metadata["do_executed_since_last_act"] = False
    task.metadata["completed_capd_cycle_count"] = task.pdca_cycle_count
    task.append_update(f"Act observed completed CAPD execution cycle {task.pdca_cycle_count}.")


def _temporary_cycle_limit_reached(task: TaskState) -> bool:
    completed_cycle_limit = max(0, int(task.pdca_cycle_count or 0)) >= _TEMPORARY_MAX_COMPLETED_CYCLES
    stubbed_planning_pass = task.metadata.get("tool_call_planning_llm_stubbed") is True
    return completed_cycle_limit or stubbed_planning_pass


def _markdown_has_acceptance_criteria(value: str) -> bool:
    rendered = str(value or "").strip()
    return bool(rendered and rendered != "- (none)")
