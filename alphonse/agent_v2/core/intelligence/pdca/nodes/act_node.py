"""Act node for the v2 PDCA graph."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from alphonse.agent_v2.core.intelligence.task_state import TaskState

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
_ACCEPTANCE_CRITERIA_ACTION_VERDICTS = {"new", "steer"}
_PLAN_ROUTE = "plan"
_END_ROUTE = "end"
_TEMPORARY_MAX_COMPLETED_CYCLES = 1


def act_node(task: TaskState) -> TaskState:
    """Act on the check verdict without re-checking or executing work."""
    _observe_completed_do_cycle(task)
    if _temporary_cycle_limit_reached(task):
        task.metadata["act_route"] = _END_ROUTE
        task.metadata["act_stop_reason"] = "temporary_cycle_limit"
        task.append_update("Act stopped the CAPD cycle at the temporary completed-cycle limit.")
        return task

    verdict = str(task.check_verdict or "").strip().lower()
    if verdict in _ACCEPTANCE_CRITERIA_ACTION_VERDICTS:
        prompt = _render_acceptance_criteria_prompt(task)
        generated_criteria = _call_acceptance_criteria_llm(prompt)
        task.metadata["acceptance_criteria_prompt"] = prompt
        task.metadata["acceptance_criteria_llm_stubbed"] = True
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
