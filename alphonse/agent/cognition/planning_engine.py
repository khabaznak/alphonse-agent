from __future__ import annotations

import json
import logging
import re
from typing import Any
from alphonse.agent.cognition.prompt_templates_runtime import (
    PLANNING_SYSTEM_PROMPT,
    PLANNING_TOOLS_TEMPLATE,
    PLANNING_USER_TEMPLATE,
    render_prompt_template,
)
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^###\s+([A-Za-z_]\w*)\(([^)]*)\)\s*$")
_INPUT_RE = re.compile(r"^\s*-\s+`([^`]+)`\s+\(([^,]+),\s*(required|optional)\)\s*$", re.IGNORECASE)


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
    user_prompt = render_prompt_template(
        PLANNING_USER_TEMPLATE,
        {
            "POLICY_BLOCK": render_utterance_policy_block(
                locale=locale,
                tone=tone,
                address_style=address_style,
                channel_type=channel_type,
            ),
            "USER_MESSAGE": text,
            "LOCALE": str(locale or "en-US"),
            "PLANNING_CONTEXT": _format_context_markdown(compact_context),
            "AVAILABLE_TOOLS": available_tools,
        },
    )
    raw = _call_llm(llm_client, system_prompt=PLANNING_SYSTEM_PROMPT, user_prompt=user_prompt)
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
    tools = planner_tool_catalog_data().get("tools")
    if not isinstance(tools, list):
        return ""
    lines: list[str] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        if not tool:
            continue
        params = item.get("input_parameters") if isinstance(item.get("input_parameters"), list) else []
        rendered = ", ".join(_render_param_signature(param) for param in params if isinstance(param, dict))
        description = str(item.get("description") or "No description.")
        lines.append(f"- {tool}({rendered}) -> {description}")
    return "\n".join(lines)


def format_available_ability_catalog() -> str:
    return render_prompt_template(PLANNING_TOOLS_TEMPLATE, {}).strip()


def planner_tool_catalog_data() -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    lines = format_available_ability_catalog().splitlines()
    for raw in lines:
        line = raw.rstrip()
        heading = _HEADING_RE.match(line.strip())
        if heading:
            if isinstance(current, dict):
                tools.append(current)
            current = {
                "tool": heading.group(1),
                "description": "",
                "when_to_use": "",
                "returns": "",
                "input_parameters": [],
            }
            continue
        if not isinstance(current, dict):
            continue
        stripped = line.strip()
        if stripped.startswith("- Description:"):
            current["description"] = stripped.replace("- Description:", "", 1).strip()
            continue
        if stripped.startswith("- When to use:"):
            current["when_to_use"] = stripped.replace("- When to use:", "", 1).strip()
            continue
        if stripped.startswith("- Returns:"):
            current["returns"] = stripped.replace("- Returns:", "", 1).strip()
            continue
        input_match = _INPUT_RE.match(line)
        if input_match:
            params = current.get("input_parameters")
            if not isinstance(params, list):
                params = []
                current["input_parameters"] = params
            params.append(
                {
                    "name": input_match.group(1).strip(),
                    "type": input_match.group(2).strip(),
                    "required": input_match.group(3).strip().lower() == "required",
                }
            )
    if isinstance(current, dict):
        tools.append(current)
    return {"tools": tools}


def planner_tool_names() -> list[str]:
    return [
        str(item.get("tool") or "").strip()
        for item in (planner_tool_catalog_data().get("tools") or [])
        if isinstance(item, dict) and str(item.get("tool") or "").strip()
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


def _format_context_markdown(context: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in context.items():
        lines.extend(_markdown_lines_for_value(key=key, value=value, indent=0))
    return "\n".join(lines) if lines else "- (none)"


def _markdown_lines_for_value(*, key: str, value: Any, indent: int) -> list[str]:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines = [f"{prefix}- **{key}**:"]
        if not value:
            lines.append(f"{prefix}  - (empty)")
            return lines
        for child_key, child_value in value.items():
            lines.extend(_markdown_lines_for_value(key=str(child_key), value=child_value, indent=indent + 1))
        return lines
    if isinstance(value, list):
        lines = [f"{prefix}- **{key}**:"]
        if not value:
            lines.append(f"{prefix}  - (empty)")
            return lines
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}  -")
                for child_key, child_value in item.items():
                    lines.extend(
                        _markdown_lines_for_value(key=str(child_key), value=child_value, indent=indent + 2)
                    )
            else:
                lines.append(f"{prefix}  - {str(item)}")
        return lines
    return [f"{prefix}- **{key}**: {str(value)}"]


def _render_param_signature(param: dict[str, Any]) -> str:
    name = str(param.get("name") or "").strip()
    ptype = str(param.get("type") or "string").strip()
    required = bool(param.get("required", False))
    return f"{name}{'' if required else '?'}:{ptype}"
