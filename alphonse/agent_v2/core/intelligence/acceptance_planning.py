"""Acceptance-contract authoring owned by the Plan stage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.acceptance_contract import apply_amendment
from alphonse.agent_v2.core.intelligence.acceptance_contract import contract_from_markdown

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def plan_acceptance_contract(task: "TaskState", context: "CoreLoopContext | None") -> bool:
    """Create initial criteria or incorporate explicit steering while Plan is active."""
    if not task.acceptance_contract:
        if str(task.acceptance_criteria_md or "").strip() not in {"", "- (none)"}:
            task.acceptance_contract = contract_from_markdown(
                task.acceptance_criteria_md, source_message_id=str(task.message_id or ""),
            )
            return bool(task.acceptance_contract)
    if context is None or context.inference is None:
        if not task.acceptance_contract:
            criteria = _call_acceptance_criteria_llm(_render_acceptance_criteria_prompt(task))
            if criteria:
                task.set_acceptance_contract_from_markdown(criteria)
        return bool(task.acceptance_contract)
    if not task.acceptance_contract:
        prompt = _render_acceptance_criteria_prompt(
            task,
            user_context_md=_user_context_md(task, context),
            project_context_md=_project_context_md(task, context),
            philosophy_md=_agent_prompt_md(context, "Philosophy.md"),
            global_context_md=_agent_prompt_md(context, "GlobalContext.md"),
        )
        result = context.inference.generate_markdown(InferenceRequest(
            prompt=prompt, purpose=InferencePurpose.ACCEPTANCE_CRITERIA,
            project_id=task.project_id, user=task.user, task_id=task.task_id,
            cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
        ))
        criteria = str(result.content or "").strip()
        task.metadata["acceptance_criteria_plan_prompt"] = prompt
        if result.model_profile is not None:
            task.metadata["acceptance_criteria_model_profile"] = result.model_profile.profile_id
        if not criteria:
            return False
        task.set_acceptance_contract_from_markdown(criteria)
        task.metadata["acceptance_criteria_updated"] = True
        task.append_update("Plan established the acceptance criteria for the mission.")
        context.record_memory_event(task, "Acceptance Criteria", task.acceptance_criteria_md)
        return bool(task.acceptance_contract)

    source_ids = task.metadata.get("pending_steering_message_ids")
    if not isinstance(source_ids, list) or not source_ids:
        return True
    prompt = _render_acceptance_criteria_amendment_prompt(
        task, user_context_md=_user_context_md(task, context), project_context_md=_project_context_md(task, context),
    )
    result = context.inference.generate_json(InferenceRequest(
        prompt=prompt, purpose=InferencePurpose.ACCEPTANCE_CRITERIA,
        project_id=task.project_id, user=task.user, task_id=task.task_id,
        cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
    ))
    task.metadata["acceptance_criteria_amendment_plan_prompt"] = prompt
    if result.model_profile is not None:
        task.metadata["acceptance_criteria_model_profile"] = result.model_profile.profile_id
    if isinstance(result.json_value, dict):
        contract, rejected = apply_amendment(
            task.acceptance_contract, result.json_value, source_message_id=str(source_ids[-1]),
        )
        task.acceptance_contract = contract
        task.sync_acceptance_criteria_view()
        task.metadata["acceptance_criteria_amendment_rejections"] = rejected
        task.metadata["acceptance_criteria_updated"] = bool(result.json_value.get("operations")) and not rejected
        context.record_memory_event(task, "Acceptance Contract Amendment", {"amendment": result.json_value, "rejected": rejected})
    task.metadata.pop("pending_steering_message_ids", None)
    return True


def _render_acceptance_criteria_prompt(
    task: "TaskState", *, user_context_md: str = "", project_context_md: str = "",
    philosophy_md: str = "", global_context_md: str = "",
) -> str:
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape(default_for_string=False), trim_blocks=True, lstrip_blocks=True)
    return env.get_template("acceptance_criteria_prompt.j2").render(
        check_verdict=task.check_verdict or "new", check_reason=task.check_reason,
        existing_acceptance_criteria_md=task.acceptance_criteria_md, user_context_md=user_context_md,
        project_context_md=project_context_md, philosophy_md=philosophy_md, global_context_md=global_context_md,
        task_state_md=task.to_markdown_prompt(include_memory=not bool(task.metadata.get("memory_context_consumed"))),
    ).strip()


def _call_acceptance_criteria_llm(prompt: str) -> str | None:
    """Test/offline stub; live generation is routed through Plan's inference context."""
    _ = prompt
    return None


def _render_acceptance_criteria_amendment_prompt(task: "TaskState", *, user_context_md: str = "", project_context_md: str = "") -> str:
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape(default_for_string=False), trim_blocks=True, lstrip_blocks=True)
    return env.get_template("acceptance_criteria_amendment_prompt.j2").render(
        acceptance_contract=task.acceptance_contract, recent_conversation_md=task.recent_conversation_md,
        user_context_md=user_context_md, project_context_md=project_context_md,
    ).strip()


def _project_context_md(task: "TaskState", context: "CoreLoopContext") -> str:
    if context.project_store is None or not str(task.project_id or "").strip():
        return ""
    render = getattr(context.project_store, "render_project_context", None)
    return str(render(task.project_id, requester_user_id=task.user) or "").strip() if callable(render) else ""


def _user_context_md(task: "TaskState", context: "CoreLoopContext") -> str:
    if not callable(context.user_context_provider):
        return ""
    try:
        return str(context.user_context_provider(task.user) or "").strip()
    except (OSError, KeyError):
        return ""


def _agent_prompt_md(context: "CoreLoopContext", name: str) -> str:
    load = getattr(context.prompts, "load", None) if context.prompts is not None else None
    return str(load(name).content or "").strip() if callable(load) else ""
