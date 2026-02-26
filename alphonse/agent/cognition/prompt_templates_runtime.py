from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment
from alphonse.agent.cognition.template_loader import load_template_or_fallback

PROMPT_SEEDS_DIR = Path(__file__).resolve().parent / "prompt_seeds"


def _seed_text(filename: str) -> str:
    path = PROMPT_SEEDS_DIR / filename
    return load_template_or_fallback(path)


def list_prompt_seed_templates() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not PROMPT_SEEDS_DIR.exists():
        return items
    for path in sorted(PROMPT_SEEDS_DIR.glob("*")):
        if not path.is_file():
            continue
        try:
            template = path.read_text(encoding="utf-8")
        except Exception:
            continue
        items.append(
            {
                "id": path.name,
                "key": path.stem,
                "template": template,
                "enabled": True,
                "source": "file_seed",
            }
        )
    return items


def get_prompt_seed_template(template_id: str) -> dict[str, Any] | None:
    path = PROMPT_SEEDS_DIR / str(template_id)
    if not path.is_file():
        return None
    try:
        template = path.read_text(encoding="utf-8")
    except Exception:
        return None
    return {
        "id": path.name,
        "key": path.stem,
        "template": template,
        "enabled": True,
        "source": "file_seed",
    }


SLOT_QUESTION_SYSTEM_PROMPT = _seed_text(
    "slot_question.system.j2",
)

FIRST_DECISION_SYSTEM_PROMPT = _seed_text(
    "first_decision.system.j2",
)

FIRST_DECISION_USER_TEMPLATE = _seed_text(
    "first_decision.user.j2",
)

PLANNING_SYSTEM_PROMPT = _seed_text(
    "planning.system.j2",
)

PLANNING_USER_TEMPLATE = _seed_text(
    "planning.user.j2",
)

SLOT_QUESTION_USER_TEMPLATE = _seed_text(
    "slot_question.user.j2",
)

CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT = _seed_text(
    "capability_gap.apology.system.j2",
)

CAPABILITY_GAP_APOLOGY_USER_TEMPLATE = _seed_text(
    "capability_gap.apology.user.j2",
)

SCHEDULER_PARAPHRASE_SYSTEM_PROMPT = _seed_text("scheduler.paraphrase.system.j2")
SCHEDULER_NORMALIZE_TIME_SYSTEM_PROMPT = _seed_text("scheduler.normalize_time.system.j2")
JOBS_YOU_JUST_REMEMBERED_SYSTEM_PROMPT = _seed_text("jobs.you_just_remembered.system.j2")
RENDERER_UTTERANCE_SYSTEM_PROMPT = _seed_text("renderer.utterance.system.j2")
CAPABILITY_GAP_REFLECTION_SYSTEM_PROMPT = _seed_text("capability_gap.reflection.system.j2")
CAPABILITY_GAP_REFLECTION_USER_TEMPLATE = _seed_text("capability_gap.reflection.user.j2")


def render_prompt_template(template: str, variables: Mapping[str, Any]) -> str:
    env = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
    rendered = env.from_string(template).render(**dict(variables))
    for name, value in variables.items():
        rendered = rendered.replace("{" + str(name) + "}", str(value))
    return rendered
