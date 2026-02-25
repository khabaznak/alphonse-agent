from __future__ import annotations

from pathlib import Path

from alphonse.agent.cognition.prompt_templates_runtime import render_prompt_template

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_TEMPLATE_MISSING_FALLBACK = (
    "Template configuration problem detected. "
    "Tell the user there is a prompt-template issue and ask them to contact the administrator."
)


def _load_template(filename: str) -> str:
    path = _TEMPLATES_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return _TEMPLATE_MISSING_FALLBACK


NEXT_STEP_SYSTEM_PROMPT = _load_template("pdca.next_step.system.j2")
NEXT_STEP_USER_TEMPLATE = _load_template("pdca.next_step.user.j2")
NEXT_STEP_REPAIR_USER_TEMPLATE = _load_template("pdca.next_step.repair.user.j2")
GOAL_CLARIFICATION_SYSTEM_PROMPT = _load_template("pdca.goal_clarification.system.j2")
GOAL_CLARIFICATION_USER_TEMPLATE = _load_template("pdca.goal_clarification.user.j2")
PROGRESS_CHECKIN_SYSTEM_PROMPT = _load_template("pdca.progress_checkin.system.j2")
PROGRESS_CHECKIN_USER_TEMPLATE = _load_template("pdca.progress_checkin.user.j2")


def render_pdca_prompt(template: str, variables: dict[str, object]) -> str:
    return render_prompt_template(template, variables).strip()
