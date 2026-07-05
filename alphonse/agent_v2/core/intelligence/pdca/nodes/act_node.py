"""Act node for the v2 PDCA graph."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from alphonse.agent_v2.core.intelligence.task_state import TaskState

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
_ACCEPTANCE_CRITERIA_ACTION_VERDICTS = {"new", "steer"}


def act_node(task: TaskState) -> TaskState:
    """Act on the check verdict without re-checking or executing work."""
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
        return task

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
