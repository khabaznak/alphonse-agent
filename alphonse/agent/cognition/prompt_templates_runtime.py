from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment

PROMPT_SEEDS_DIR = Path(__file__).resolve().parent / "prompt_seeds"


def _seed_text(filename: str, fallback: str) -> str:
    path = PROMPT_SEEDS_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return fallback


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
    (
        "# Role\n"
        "You generate one concise clarification question for an assistant.\n\n"
        "# Rules\n"
        "- Return plain text only.\n"
        "- Do not include extra explanations.\n"
    ),
)

SLOT_QUESTION_USER_TEMPLATE = _seed_text(
    "slot_question.user.j2",
    (
        "# Intent\n{{ INTENT_NAME }}\n\n"
        "# Missing Slot Name\n{{ SLOT_NAME }}\n\n"
        "# Missing Slot Type\n{{ SLOT_TYPE }}\n\n"
        "# Locale\n{{ LOCALE }}\n\n"
        "# Task\nAsk exactly one natural question to collect this slot.\n"
    ),
)

CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT = _seed_text(
    "capability_gap.apology.system.j2",
    (
        "# Role\n"
        "You are Alphonse.\n\n"
        "# Task\n"
        "Produce one short apology message to the user.\n\n"
        "# Rules\n"
        "- State clearly that you cannot complete the request because a required ability or tool is missing.\n"
        "- Be polite, concise, and direct.\n"
        "- Do not mention internal stack traces or code details.\n"
        "- Return plain text only.\n"
    ),
)

CAPABILITY_GAP_APOLOGY_USER_TEMPLATE = _seed_text(
    "capability_gap.apology.user.j2",
    (
        "# User Message\n{{ USER_MESSAGE }}\n\n"
        "# Intent\n{{ INTENT }}\n\n"
        "# Failure Reason\n{{ GAP_REASON }}\n\n"
        "# Missing Slots\n{{ MISSING_SLOTS }}\n\n"
        "# Locale\n{{ LOCALE }}\n\n"
        "# Task\nWrite the apology now.\n"
    ),
)


def render_prompt_template(template: str, variables: Mapping[str, Any]) -> str:
    env = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
    rendered = env.from_string(template).render(**dict(variables))
    for name, value in variables.items():
        rendered = rendered.replace("{" + str(name) + "}", str(value))
    return rendered
