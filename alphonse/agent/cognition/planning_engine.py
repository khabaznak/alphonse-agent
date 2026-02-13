from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from alphonse.agent.cognition.abilities.store import AbilitySpecStore
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block

logger = logging.getLogger(__name__)

_DISCOVERY_EXCLUDED_TOOLS: set[str] = {"cancel"}


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
        return {"messages": [], "plans": []}
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
        return {"messages": [], "plans": []}
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
        return {"messages": [], "plans": []}
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
    specs = _load_ability_specs()
    lines: list[str] = [
        "- askQuestion(question:string, slot?:string, bind?:object) -> Use it to ask clarifying questions to the message author user. Only ask questions about missing data, never for missing tools.",
    ]
    for spec in specs:
        intent = str(spec.get("intent_name") or "").strip()
        if not intent or intent in _DISCOVERY_EXCLUDED_TOOLS:
            continue
        params = spec.get("input_parameters") if isinstance(spec.get("input_parameters"), list) else []
        summary = _ability_summary(spec)
        if params:
            rendered = ", ".join(_render_param_signature(item) for item in params if isinstance(item, dict))
            lines.append(f"- {intent}({rendered}) -> {summary}")
        else:
            lines.append(f"- {intent}() -> {summary}")
    return "\n".join(lines)


def format_available_ability_catalog() -> str:
    specs = _load_ability_specs()
    tools: list[dict[str, Any]] = [
        {
            "tool": "askQuestion",
            "description": "Ask the user for missing parameters to continue.",
            "when_to_use": "Use when required end-user data is missing.",
            "returns": "user_answer_captured",
            "input_parameters": [
                {"name": "question", "type": "string", "required": True},
                {"name": "slot", "type": "string", "required": False},
                {"name": "bind", "type": "object", "required": False},
            ],
            "required_parameters": ["question"],
        }
    ]
    for spec in specs:
        tool = str(spec.get("intent_name") or "").strip()
        if not tool or tool in _DISCOVERY_EXCLUDED_TOOLS:
            continue
        params_raw = spec.get("input_parameters") if isinstance(spec.get("input_parameters"), list) else []
        params: list[dict[str, Any]] = []
        for item in params_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            params.append(
                {
                    "name": name,
                    "type": str(item.get("type") or "string"),
                    "required": bool(item.get("required", False)),
                }
            )
        tools.append(
            {
                "tool": tool,
                "description": _ability_summary(spec),
                "when_to_use": "Use when this tool directly satisfies the user request.",
                "returns": "result",
                "required_parameters": [p["name"] for p in params if bool(p.get("required"))],
                "input_parameters": params,
            }
        )
    return json.dumps({"tools": tools}, ensure_ascii=False)


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
        if tool != "askQuestion" and not params:
            return {
                "code": "NON_ASKQUESTION_EMPTY_PARAMETERS",
                "message": f"step[{idx}] non-askQuestion actions must include parameters.",
            }
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


def _ability_summary(spec: dict[str, Any]) -> str:
    intent = str(spec.get("intent_name") or "").strip()
    if not intent:
        return "Use it to complete the requested task."
    return f"Use it to {intent.replace('.', ' ').replace('_', ' ').strip().lower()}."


def _render_param_signature(param: dict[str, Any]) -> str:
    name = str(param.get("name") or "").strip()
    ptype = str(param.get("type") or "string").strip()
    required = bool(param.get("required", False))
    return f"{name}{'' if required else '?'}:{ptype}"


def _load_ability_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    specs.extend(_load_specs_file())
    specs.extend(_load_specs_db())
    unique: dict[str, dict[str, Any]] = {}
    for spec in specs:
        key = str(spec.get("intent_name") or "").strip()
        if key:
            unique[key] = spec
    return [unique[key] for key in sorted(unique.keys())]


def _load_specs_file() -> list[dict[str, Any]]:
    path = (
        Path(__file__).resolve().parents[1]
        / "nervous_system"
        / "resources"
        / "ability_specs.seed.json"
    )
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _load_specs_db() -> list[dict[str, Any]]:
    try:
        return AbilitySpecStore().list_enabled_specs()
    except Exception:
        return []
