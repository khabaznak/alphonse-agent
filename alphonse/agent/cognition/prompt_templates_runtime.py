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

FIRST_DECISION_SYSTEM_PROMPT = _seed_text(
    "first_decision.system.j2",
    (
        "# Role\n"
        "You are Alphonse, a routing controller for a conversational agent.\n\n"
        "# Routes\n"
        "- `direct_reply`: you can answer now without tools, using the `RECENT CONVERSATION` section when relevant.\n"
        "- `tool_plan`: a tool-based plan is required.\n"
        "- `clarify`: one short clarification question is required.\n\n"
        "# Grounding\n"
        "- Treat the `RECENT CONVERSATION` section as authoritative for what the user already said earlier in this session/day.\n"
        "- If the latest user message is a follow-up that can be resolved from `RECENT CONVERSATION` (counts, names, prior answers), choose `direct_reply` and answer.\n"
        "- Do NOT ask clarifying questions when the answer is explicitly present in `RECENT CONVERSATION`.\n\n"
        "# Tool awareness\n"
        "- The tool list is awareness only. It is NOT a request to use tools.\n\n"
        "# Rules\n"
        "- Prefer `direct_reply` for greetings, language preference/capability questions, and simple conversation.\n"
        "- Use `tool_plan` only when external data or side effects are required.\n"
        "- Do not mention internal tool/function names in clarify questions.\n\n"
        "# Output\n"
        "Return strict JSON only.\n"
        "- Output must be a single JSON object (no markdown, no prose).\n"
        "- Use double quotes for all keys and string values.\n"
        "- Do not include trailing commas.\n\n"
        "Required keys:\n"
        "- `route`: one of `direct_reply`, `tool_plan`, `clarify`\n"
        "- `intent`: short intent label string\n"
        "- `confidence`: number from 0.0 to 1.0\n"
        "- `reply_text`: text to send when route is `direct_reply`, otherwise empty string\n"
        "- `clarify_question`: text to ask when route is `clarify`, otherwise empty string\n"
    ),
)

FIRST_DECISION_USER_TEMPLATE = _seed_text(
    "first_decision.user.j2",
    (
        "# UTTERANCE POLICY\n"
        "{{ POLICY_BLOCK }}\n\n"
        "# AVAILABLE TOOL NAMES (AWARENESS ONLY)\n"
        "{{ TOOL_NAMES }}\n\n"
        "{{ RECENT_CONVERSATION }}\n\n"
        "# USER MESSAGE\n"
        "{{ USER_MESSAGE }}\n"
    ),
)

PLANNING_SYSTEM_PROMPT = _seed_text(
    "planning.system.j2",
    (
        "# Role\n"
        "You are Alphonse planning engine.\n\n"
        "# Task\n"
        "Produce a compact executable plan using available tools.\n\n"
        "# Grounding\n"
        "- Treat the `RECENT CONVERSATION` section as authoritative for this session/day.\n"
        "- Use `RECENT CONVERSATION` to resolve follow-ups and to avoid unnecessary questions.\n\n"
        "# Output\n"
        "Return strict JSON only.\n"
        "- Output must be a single JSON object (no markdown, no prose).\n"
        "- Use double quotes for all keys and string values.\n"
        "- Do not include trailing commas.\n\n"
        "Required fields:\n"
        "- `intention`: string\n"
        "- `confidence`: one of `low`, `medium`, `high`\n"
        "- `acceptance_criteria`: array of strings\n"
        "- `planning_interrupt`: object or null\n"
        "  - `question`: string\n"
        "  - `slot`: string\n"
        "  - `bind`: object\n"
        "  - `missing_data`: array\n"
        "- `execution_plan`: array of step objects\n"
        "  - each step has `tool` and `parameters`\n\n"
        "# Rules\n"
        "- Prefer direct executable steps with concrete parameters.\n"
        "- Use planning_interrupt only when required user data is missing.\n"
        "- Never ask the user about internal tool/function names.\n"
        "- Keep execution_plan empty if planning_interrupt is present.\n"
    ),
)

PLANNING_USER_TEMPLATE = _seed_text(
    "planning.user.j2",
    (
        "# UTTERANCE POLICY\n"
        "{{ POLICY_BLOCK }}\n\n"
        "{{ RECENT_CONVERSATION }}\n\n"
        "# USER MESSAGE\n"
        "{{ USER_MESSAGE }}\n\n"
        "# LOCALE\n"
        "- {{ LOCALE }}\n\n"
        "# PLANNING CONTEXT\n"
        "{{ PLANNING_CONTEXT }}\n\n"
        "# AVAILABLE TOOLS\n"
        "{{ AVAILABLE_TOOLS }}\n"
    ),
)

PLANNING_TOOLS_TEMPLATE = _seed_text(
    "planning.tools.md.j2",
    (
        "## Tool Catalog\n"
        "### askQuestion(question:string)\n"
        "- Description: Ask the user one clear question and wait for their answer.\n"
        "- When to use: Only when required user data is missing.\n"
        "- Returns: user_answer_captured\n"
        "- Inputs:\n"
        "  - `question` (string, required)\n\n"
        "### getTime()\n"
        "- Description: Get your current time now.\n"
        "- When to use: Use for current time/date and as a reference for scheduling or deadline calculations.\n"
        "- Returns: current_time\n"
        "- Inputs: none\n\n"
        "### createReminder(ForWhom:string, Time:string, Message:string)\n"
        "- Description: Create a reminder directly for a person at a specific time.\n"
        "- When to use: Use when the user asks to be reminded.\n"
        "- Returns: scheduled_reminder_id\n"
        "- Inputs:\n"
        "  - `ForWhom` (string, required)\n"
        "  - `Time` (string, required)\n"
        "  - `Message` (string, required)\n\n"
        "### getMySettings()\n"
        "- Description: Get your current runtime settings (timezone, locale, tone, address style, channel context).\n"
        "- When to use: Use before time or language-sensitive decisions when settings are needed.\n"
        "- Returns: settings\n"
        "- Inputs: none\n\n"
        "### getUserDetails()\n"
        "- Description: Get known user/channel details for the current conversation context.\n"
        "- When to use: Use when user identity/context details are needed before planning or scheduling.\n"
        "- Returns: user_details\n"
        "- Inputs: none\n"
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
