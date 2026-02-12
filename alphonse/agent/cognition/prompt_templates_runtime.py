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


GRAPH_PLAN_CRITIC_SYSTEM_PROMPT = _seed_text(
    "graph.plan_critic.system.j2",
    "You are a strict plan-step critic and repairer. "
    "Your job is to repair one invalid tool call step. "
    "You must choose only from AVAILABLE TOOLS CATALOG and output JSON only.\n",
)

GRAPH_PLAN_CRITIC_USER_TEMPLATE = _seed_text(
    "graph.plan_critic.user.j2",
    (
        "Repair this invalid execution step.\n"
        "Rules:\n"
        "- Select exactly one tool from AVAILABLE TOOLS CATALOG.\n"
        "- Keep the original user goal.\n"
        "- If required parameters are missing, choose askQuestion and ask for only the missing data.\n"
        "- Never ask the user to choose or confirm tool/function names.\n"
        "- Keep output minimal and executable now.\n"
        "- Output JSON only with shape: "
        '{"tool":"<tool_name>","parameters":{...}}'
        "\n\n"
        "User message:\n{{ USER_MESSAGE }}\n\n"
        "Invalid step:\n{{ INVALID_STEP_JSON }}\n\n"
        "Validation exception:\n{{ VALIDATION_EXCEPTION_JSON }}\n\n"
        "AVAILABLE TOOLS SIGNATURES:\n{{ AVAILABLE_TOOL_SIGNATURES }}\n\n"
        "AVAILABLE TOOLS CATALOG:\n{{ AVAILABLE_TOOL_CATALOG }}\n"
    ),
)

INTENT_DISCOVERY_PLAN_CRITIC_SYSTEM_FALLBACK = _seed_text(
    "intent_discovery.plan_critic.system.j2",
    "You are Alphonse, a strict plan-shape repairer. "
    "Fix only structural issues in an executionPlan. "
    "Output valid JSON only.\n",
)

INTENT_DISCOVERY_QUESTION_POLICY_BLOCK = _seed_text(
    "intent_discovery.question_policy.txt",
    (
        "GLOBAL QUESTION POLICY:\n"
        "- askQuestion is not part of the execution plan.\n"
        "- askQuestion is a planning interrupt for missing end-user data.\n"
        "- If missing data is detected, emit planning_interrupt and stop further planning this turn.\n"
        "- Do not place askQuestion inside execution_plan.\n"
        "- Never ask the user to choose internal tool/function names.\n"
    ),
)

INTENT_DISCOVERY_PLAN_SYNTH_SYSTEM_FALLBACK = _seed_text(
    "intent_discovery.plan_synth.system.j2",
    (
        "Your name is Alphonse, you are the genius virtual butler for the family."
        "The family members talk to you via messages, sometimes short messages, sometimes long paragraphs."
        "Some of them have instructions, others are favors they ask of you but you are tremendously helpful"
        " for the family. Your objective is to determine what to do with the provided message "
        "and create a plan for you to execute as a virtual agent. you must guarantee success."
        "RULES:\n"
        "- You will create a plan as a sequence of discrete steps.  "
        "- Your output is only the sequence of discrete and well defined steps. no explanations or anything else.  "
        "- for better readability use <ol> and <li> html tags to enumerate the steps.  "
    ),
)

INTENT_DISCOVERY_PLAN_SYNTH_USER_FALLBACK = _seed_text(
    "intent_discovery.plan_synth.user.j2",
    (
        "USER:\n"
        '{"message": "{MESSAGE_TEXT}", "message_author": "{USER}"}\n\n'
    ),
)

INTENT_DISCOVERY_TOOL_BINDER_SYSTEM_FALLBACK = _seed_text(
    "intent_discovery.tool_binder.system.j2",
    (
        "You are Alphonse a master planner. Your mission is to prune the following plan of steps which "
        "may not be necessary when compared agains a tool menu. "
        "The previous agent who created the plan did not know the existence of such tools but you have "
        "that information. "
        "You must output a revised plan with only the steps which make use of the provided tools. "
        "Do not output any explanations or justifications; only output the plan."
        "RULES: \n"
        "- Review the plan and compare against the tool menu. The main objective is to determine "
        "if the plan has unnecessary or invalid steps given the tools at your disposal. "
        "For example, a plan might have 2 steps and a tool might achieve it with only 1.  \n"
        "- Output the new revised plan, same format as it is right now but verified by you.  \n"
        "- PRO Tip: if a plan step has been executed or does not require or does not match a tool "
        "it means that it is an invalid plan step and should be pruned."
    ),
)

INTENT_DISCOVERY_TOOL_BINDER_USER_FALLBACK = _seed_text(
    "intent_discovery.tool_binder.user.j2",
    (
        "{QUESTION_POLICY}\n"
        "PLAN TO REVIEW:\n"
        "{PLAN}\n\n"
        "TOOL_MENU:\n{TOOL_MENU}\n\n"
        "CONTEXT:\n{CONTEXT_JSON}\n\n"
        "EXCEPTION_HISTORY:\n{EXCEPTION_HISTORY_JSON}\n"
    ),
)

INTENT_DISCOVERY_PLAN_REFINER_SYSTEM_FALLBACK = _seed_text(
    "intent_discovery.plan_refiner.system.j2",
    "You are Alphonse Plan Refiner. Produce the final execution-ready plan using tool IDs only. "
    "Output valid JSON only.\n",
)

INTENT_DISCOVERY_PLAN_REFINER_USER_FALLBACK = _seed_text(
    "intent_discovery.plan_refiner.user.j2",
    (
        "{QUESTION_POLICY}\n"
        "Rules:\n"
        "- Preserve step_id continuity.\n"
        "- If planning_interrupt is needed, set status=NEEDS_USER_INPUT and execution_plan=[].\n\n"
        "Return JSON:\n"
        "{\n"
        '  "plan_version":"v1",\n'
        '  "status":"READY|NEEDS_USER_INPUT|BLOCKED",\n'
        '  "execution_plan":[{"step_id":"S1","sequence":1,"kind":"TOOL|QUESTION","tool_id":0,"parameters":{},"acceptance_links":[]}],\n'
        '  "planning_interrupt":{"tool_id":0,"tool_name":"askQuestion","question":"...","slot":"...","bind":{},"missing_data":[],"reason":"..."},\n'
        '  "acceptance_criteria":["..."],\n'
        '  "repair_log":[]\n'
        "}\n\n"
        "STEP_A_PLAN:\n{PLAN_A_JSON}\n\n"
        "STEP_B_BINDINGS:\n{BINDINGS_B_JSON}\n\n"
        "TOOL_MENU:\n{TOOL_MENU_JSON}\n\n"
        "CONTEXT:\n{CONTEXT_JSON}\n\n"
        "EXCEPTION_HISTORY:\n{EXCEPTION_HISTORY_JSON}\n"
    ),
)

INTENT_DISCOVERY_PLAN_CRITIC_USER_TEMPLATE = _seed_text(
    "intent_discovery.plan_critic.user.j2",
    (
        "Repair this executionPlan.\n"
        "Rules:\n"
        '- Return JSON with shape {"executionPlan": [{"tool"|"action": "...", "parameters": {...}}]}.\n'
        "- Keep the same intent and acceptance criteria direction.\n"
        "- If missing data, use askQuestion with user-facing question only.\n"
        "- Never ask user to choose internal tool/function names.\n"
        "- Do not output explanations.\n\n"
        "Message:\n{{ CHUNK_TEXT }}\n\n"
        "Intention:\n{{ INTENTION }}\n\n"
        "Acceptance:\n{{ ACCEPTANCE_JSON }}\n\n"
        "Validation issue:\n{{ ISSUE_JSON }}\n\n"
        "Invalid plan:\n{{ INVALID_PLAN_JSON }}\n\n"
        "AVAILABLE TOOLS:\n{{ AVAILABLE_TOOLS }}\n"
    ),
)

SLOT_QUESTION_SYSTEM_PROMPT = _seed_text(
    "slot_question.system.j2",
    (
        "You generate one concise clarification question for an assistant.\n"
        "Return plain text only.\n"
        "Do not include extra explanations.\n"
    ),
)

SLOT_QUESTION_USER_TEMPLATE = _seed_text(
    "slot_question.user.j2",
    (
        "Intent: {{ INTENT_NAME }}\n"
        "Missing slot name: {{ SLOT_NAME }}\n"
        "Missing slot type: {{ SLOT_TYPE }}\n"
        "Locale: {{ LOCALE }}\n"
        "Ask exactly one natural question to collect this slot.\n"
    ),
)

CAPABILITY_GAP_APOLOGY_SYSTEM_PROMPT = _seed_text(
    "capability_gap.apology.system.j2",
    (
        "You are Alphonse. You must produce one short apology message to the user.\n"
        "State clearly that you cannot complete the request because a required ability or tool is missing.\n"
        "Be polite, concise, and direct.\n"
        "Do not mention internal stack traces or code details.\n"
        "Return plain text only.\n"
    ),
)

CAPABILITY_GAP_APOLOGY_USER_TEMPLATE = _seed_text(
    "capability_gap.apology.user.j2",
    (
        "User message:\n{{ USER_MESSAGE }}\n\n"
        "Intent:\n{{ INTENT }}\n\n"
        "Failure reason:\n{{ GAP_REASON }}\n\n"
        "Missing slots:\n{{ MISSING_SLOTS }}\n\n"
        "Locale:\n{{ LOCALE }}\n\n"
        "Write the apology now.\n"
    ),
)


def render_prompt_template(template: str, variables: Mapping[str, Any]) -> str:
    env = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
    rendered = env.from_string(template).render(**dict(variables))
    for name, value in variables.items():
        rendered = rendered.replace("{" + str(name) + "}", str(value))
    return rendered
