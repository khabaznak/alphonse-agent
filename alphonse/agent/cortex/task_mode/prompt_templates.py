from __future__ import annotations

from pathlib import Path

from alphonse.agent.cognition.prompt_templates_runtime import render_prompt_template
from alphonse.agent.cognition.template_loader import load_template_or_fallback

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(filename: str) -> str:
    path = _TEMPLATES_DIR / filename
    return load_template_or_fallback(path)


NEXT_STEP_SYSTEM_PROMPT = _load_template("pdca.next_step.system.j2")
NEXT_STEP_USER_TEMPLATE = _load_template("pdca.next_step.user.j2")
NEXT_STEP_REPAIR_USER_TEMPLATE = _load_template("pdca.next_step.repair.user.j2")
PROGRESS_CHECKIN_SYSTEM_PROMPT = _load_template("pdca.progress_checkin.system.j2")
PROGRESS_CHECKIN_USER_TEMPLATE = _load_template("pdca.progress_checkin.user.j2")


def render_pdca_prompt(template: str, variables: dict[str, object]) -> str:
    return render_prompt_template(template, variables).strip()
