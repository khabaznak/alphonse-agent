from __future__ import annotations

import json
import logging
from typing import Any
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block

logger = logging.getLogger(__name__)

_PLANNER_TOOL_CARDS: list[dict[str, Any]] = [
    {
        "tool": "askQuestion",
        "description": "Ask the user one clear question and wait for their answer.",
        "when_to_use": "Only when required user data is missing.",
        "returns": "user_answer_captured",
        "input_parameters": [
            {"name": "question", "type": "string", "required": True},
        ],
    },
    {
        "tool": "getTime",
        "description": "Get your current time now.",
        "when_to_use": "Use for current time/date and as a reference for scheduling or deadline calculations.",
        "returns": "current_time",
        "input_parameters": [],
    },
    {
        "tool": "createTimeEventTrigger",
        "description": "Create a time-based trigger from a time expression.",
        "when_to_use": "Use when a reminder should fire at a specific time.",
        "returns": "event_trigger",
        "input_parameters": [
            {"name": "time", "type": "string", "required": True},
        ],
    },
    {
        "tool": "scheduleReminder",
        "description": "Schedule a reminder using a trigger.",
        "when_to_use": "Use when the user asks to be reminded.",
        "returns": "scheduled_reminder_id",
        "input_parameters": [
            {"name": "Message", "type": "string", "required": True},
            {"name": "To", "type": "string", "required": True},
            {"name": "From", "type": "string", "required": True},
            {"name": "EventTrigger", "type": "object", "required": True},
        ],
    },
    {
        "tool": "getMySettings",
        "description": "Get your current runtime settings (timezone, locale, tone, address style, channel context).",
        "when_to_use": "Use before time or language-sensitive decisions when settings are needed.",
        "returns": "settings",
        "input_parameters": [],
    },
    {
        "tool": "getUserDetails",
        "description": "Get known user/channel details for the current conversation context.",
        "when_to_use": "Use when user identity/context details are needed before planning or scheduling.",
        "returns": "user_details",
        "input_parameters": [],
    },
]


def discover_plan(
    *,
    text: str,
    llm_client: object | None,
    available_tools: str,
    locale: str | None = None,
    tone: str | None = None,
    address_style: str | None = None,
    channel_type: str | None = None,
    planning_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not llm_client:
        return {"messages": [], "plans": []}
    compact_context = _compact_planning_context(text=text, planning_context=planning_context)
    system_prompt = (
        "You are Alphonse planning engine.\n"
        "Produce a compact executable plan using available tools.\n"
        "Return JSON only with shape:\n"
        '{'
        '"intention":"string",'
        '"confidence":"low|medium|high",'
        '"acceptance_criteria":["..."],'
        '"planning_interrupt":{"question":"...", "slot":"answer", "bind":{}, "missing_data":[]}|null,'
        '"execution_plan":[{"tool":"tool.name","parameters":{"key":"value"}}]'
        '}\n'
        "Rules:\n"
        "- Prefer direct executable steps with concrete parameters.\n"
        "- Use planning_interrupt only when required user data is missing.\n"
        "- Never ask the user about internal tool/function names.\n"
        "- Keep execution_plan empty if planning_interrupt is present.\n"
    )
    user_prompt = (
        f"{render_utterance_policy_block(locale=locale, tone=tone, address_style=address_style, channel_type=channel_type)}\n"
        "MESSAGE:\n"
        f"{text}\n\n"
        "LOCALE:\n"
        f"{str(locale or 'en-US')}\n\n"
        "COMPACT_CONTEXT_JSON:\n"
        f"{json.dumps(compact_context, ensure_ascii=False)}\n\n"
        "AVAILABLE_TOOLS:\n"
        f"{available_tools}\n"
    )
    raw = _call_llm(llm_client, system_prompt=system_prompt, user_prompt=user_prompt)
    payload = _parse_json(raw)
    if not isinstance(payload, dict):
        return {
            "messages": [],
            "plans": [],
            "planning_error": {"code": "INVALID_JSON", "message": "Planner returned invalid JSON."},
        }
    interrupt = _extract_planning_interrupt(payload, locale=str(locale or "en-US"))
    if interrupt is not None:
        return {
            "messages": [
                {
                    "action": str(payload.get("intention") or "overall"),
                    "message": text,
                    "intention": str(payload.get("intention") or "overall"),
                    "confidence": str(payload.get("confidence") or "medium"),
                }
            ],
            "plans": [],
            "planning_interrupt": interrupt,
        }
    execution_plan = payload.get("execution_plan")
    if not isinstance(execution_plan, list):
        execution_plan = payload.get("executionPlan")
    if not isinstance(execution_plan, list):
        return {
            "messages": [],
            "plans": [],
            "planning_error": {
                "code": "MISSING_EXECUTION_PLAN",
                "message": "Planner did not provide execution_plan.",
            },
        }
    normalized: list[dict[str, Any]] = []
    for step in execution_plan:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or step.get("action") or "").strip()
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        if not tool:
            continue
        normalized.append({"tool": tool, "parameters": params, "executed": False})
    issue = _validate_execution_plan(normalized)
    if issue is not None:
        logger.info(
            "planning engine invalid execution plan code=%s detail=%s",
            issue.get("code"),
            issue.get("message"),
        )
        return {"messages": [], "plans": [], "planning_error": issue}
    intention = str(payload.get("intention") or "overall").strip() or "overall"
    confidence = str(payload.get("confidence") or "medium").strip().lower() or "medium"
    acceptance = payload.get("acceptance_criteria")
    if not isinstance(acceptance, list):
        acceptance = []
    return {
        "messages": [
            {
                "action": intention,
                "message": text,
                "intention": intention,
                "confidence": confidence,
            }
        ],
        "plans": [
            {
                "message_index": 0,
                "acceptanceCriteria": [str(item) for item in acceptance if str(item).strip()],
                "executionPlan": normalized,
            }
        ],
    }


def format_available_abilities() -> str:
    lines: list[str] = []
    for card in _PLANNER_TOOL_CARDS:
        tool = str(card.get("tool") or "").strip()
        if not tool:
            continue
        params = card.get("input_parameters") if isinstance(card.get("input_parameters"), list) else []
        rendered = ", ".join(_render_param_signature(item) for item in params if isinstance(item, dict))
        description = str(card.get("description") or "No description.")
        lines.append(f"- {tool}({rendered}) -> {description}")
    return "\n".join(lines)


def format_available_ability_catalog() -> str:
    return json.dumps({"tools": _PLANNER_TOOL_CARDS}, ensure_ascii=False)


def planner_tool_names() -> list[str]:
    return [
        str(card.get("tool") or "").strip()
        for card in _PLANNER_TOOL_CARDS
        if str(card.get("tool") or "").strip()
    ]


def _validate_execution_plan(plan: list[dict[str, Any]]) -> dict[str, str] | None:
    if not isinstance(plan, list) or not plan:
        return {"code": "EMPTY_PLAN", "message": "executionPlan must contain at least one step."}
    for idx, step in enumerate(plan):
        if not isinstance(step, dict):
            return {"code": "INVALID_STEP_SHAPE", "message": f"step[{idx}] must be an object."}
        tool = str(step.get("tool") or step.get("action") or "").strip()
        if not tool:
            return {"code": "MISSING_TOOL", "message": f"step[{idx}] must include tool or action."}
        params = step.get("parameters")
        if params is None:
            return {"code": "MISSING_PARAMETERS", "message": f"step[{idx}] parameters are required."}
        if not isinstance(params, dict):
            return {"code": "INVALID_PARAMETERS", "message": f"step[{idx}] parameters must be an object."}
    return None


def _parse_json(raw: str) -> Any:
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    extracted = _extract_first_json(candidate)
    if not extracted:
        return None
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return None


def _extract_first_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _call_llm(llm_client: object, *, system_prompt: str, user_prompt: str) -> str:
    try:
        return str(llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception as exc:
        logger.warning("planning engine llm call failed error=%s", exc)
        return ""


def _extract_planning_interrupt(payload: dict[str, Any], *, locale: str) -> dict[str, Any] | None:
    interrupt = payload.get("planning_interrupt")
    if not isinstance(interrupt, dict):
        return None
    question = str(interrupt.get("question") or "").strip()
    if not question:
        if locale.lower().startswith("es"):
            question = "¿Qué dato te falta para continuar?"
        else:
            question = "What detail is missing to continue?"
    slot = str(interrupt.get("slot") or "answer").strip() or "answer"
    bind = interrupt.get("bind") if isinstance(interrupt.get("bind"), dict) else {}
    missing_data = interrupt.get("missing_data")
    return {
        "tool_name": "askQuestion",
        "question": question,
        "slot": slot,
        "bind": bind,
        "missing_data": missing_data if isinstance(missing_data, list) else [],
    }


def _compact_planning_context(
    *,
    text: str,
    planning_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(planning_context, dict):
        return {"latest_user_message": text}
    compact: dict[str, Any] = {"latest_user_message": text}
    for key in ("original_message", "latest_user_answer"):
        value = planning_context.get(key)
        if isinstance(value, str) and value.strip():
            compact[key] = value.strip()
    facts = planning_context.get("facts")
    if isinstance(facts, dict) and facts:
        compact["facts"] = facts
    return compact


def _render_param_signature(param: dict[str, Any]) -> str:
    name = str(param.get("name") or "").strip()
    ptype = str(param.get("type") or "string").strip()
    required = bool(param.get("required", False))
    return f"{name}{'' if required else '?'}:{ptype}"
