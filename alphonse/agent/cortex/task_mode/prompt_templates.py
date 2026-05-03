from __future__ import annotations

from alphonse.agent.cognition.prompt_templates_runtime import PROMPT_SEEDS_DIR
from alphonse.agent.cognition.prompt_templates_runtime import render_prompt_template
from alphonse.agent.cognition.template_loader import load_template_or_fallback


def _load_template(filename: str) -> str:
    path = PROMPT_SEEDS_DIR / filename
    return load_template_or_fallback(path)


NEXT_STEP_SYSTEM_PROMPT = _load_template("pdca.next_step.system.j2")
NEXT_STEP_USER_TEMPLATE = _load_template("pdca.next_step.user.j2")


def render_pdca_prompt(template: str, variables: dict[str, object]) -> str:
    return render_prompt_template(template, variables).strip()
