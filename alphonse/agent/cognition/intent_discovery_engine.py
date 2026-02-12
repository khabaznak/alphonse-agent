from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pathlib import Path

from alphonse.agent.cognition.prompt_store import PromptContext, SqlitePromptStore
from alphonse.agent.io import get_io_registry

logger = logging.getLogger(__name__)


_PLAN_CRITIC_KEY = "intent_discovery.plan_critic.v1"


_PLAN_CRITIC_SYSTEM_FALLBACK = (
    "You are Alphonse, a strict plan-shape repairer. "
    "Fix only structural issues in an executionPlan. "
    "Output valid JSON only."
)

_PLACEHOLDER_PATTERNS = (
    r"<[^>]+>",
    r"\byour\s+\w+\s+name\b",
    r"\bplaceholder\b",
    r"\[unknown\]",
)
_DISCOVERY_EXCLUDED_TOOLS: set[str] = {"cancel"}

_PLAN_SYNTH_KEY = "intent_discovery.plan_synth.v1"
_TOOL_BINDER_KEY = "intent_discovery.tool_binder.v1"
_PLAN_REFINER_KEY = "intent_discovery.plan_refiner.v1"

_QUESTION_POLICY_BLOCK = (
    "GLOBAL QUESTION POLICY:\n"
    "- askQuestion is not part of the execution plan.\n"
    "- askQuestion is a planning interrupt for missing end-user data.\n"
    "- If missing data is detected, emit planning_interrupt and stop further planning this turn.\n"
    "- Do not place askQuestion inside execution_plan.\n"
    "- Never ask the user to choose internal tool/function names.\n"
)

_PLAN_SYNTH_SYSTEM_FALLBACK = (
    "Your name is Alphonse, you are the genius virtual butler for the family."
    "The family members talk to you via messages, sometimes short messages, sometimes long paragraphs."
    "Some of them have instructions, others are favors they ask of you but you are tremendously helpful"
    " for the family. Your objective is to determine what to do with the provided message "
    "and create a plan for you to execute as a virtual agent. you must guarantee success."
    "RULES:\n"
    "- You will create a plan as a sequence of discrete steps.  "
    "- Your output is only the sequence of discrete and well defined steps. no explanations or anything else.  "
    "- for better readability use <ol> and <li> html tags to enumerate the steps.  "
)

_PLAN_SYNTH_USER_FALLBACK = (
    "USER:\n"
    "{\"message\": \"{MESSAGE_TEXT}\", \"message_author\": \"{USER}\"}\n\n"
)

_TOOL_BINDER_SYSTEM_FALLBACK = (
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
)

_TOOL_BINDER_USER_FALLBACK = (
    "{QUESTION_POLICY}\n"
    "PLAN TO REVIEW:\n"
    "{PLAN}\n\n"
    "TOOL_MENU:\n{TOOL_MENU}\n\n"
    "CONTEXT:\n{CONTEXT_JSON}\n\n"
    "EXCEPTION_HISTORY:\n{EXCEPTION_HISTORY_JSON}\n"
)

_PLAN_REFINER_SYSTEM_FALLBACK = (
    "You are Alphonse Plan Refiner. Produce the final execution-ready plan using tool IDs only. "
    "Output valid JSON only."
)

_PLAN_REFINER_USER_FALLBACK = (
    "{QUESTION_POLICY}\n"
    "Rules:\n"
    "- Preserve step_id continuity.\n"
    "- If planning_interrupt is needed, set status=NEEDS_USER_INPUT and execution_plan=[].\n\n"
    "Return JSON:\n"
    "{\n"
    "  \"plan_version\":\"v1\",\n"
    "  \"status\":\"READY|NEEDS_USER_INPUT|BLOCKED\",\n"
    "  \"execution_plan\":[{\"step_id\":\"S1\",\"sequence\":1,\"kind\":\"TOOL|QUESTION\",\"tool_id\":0,\"parameters\":{},\"acceptance_links\":[]}],\n"
    "  \"planning_interrupt\":{\"tool_id\":0,\"tool_name\":\"askQuestion\",\"question\":\"...\",\"slot\":\"...\",\"bind\":{},\"missing_data\":[],\"reason\":\"...\"},\n"
    "  \"acceptance_criteria\":[\"...\"],\n"
    "  \"repair_log\":[]\n"
    "}\n\n"
    "STEP_A_PLAN:\n{PLAN_A_JSON}\n\n"
    "STEP_B_BINDINGS:\n{BINDINGS_B_JSON}\n\n"
    "TOOL_MENU:\n{TOOL_MENU_JSON}\n\n"
    "CONTEXT:\n{CONTEXT_JSON}\n\n"
    "EXCEPTION_HISTORY:\n{EXCEPTION_HISTORY_JSON}\n"
)


def discover_plan(
    *,
    text: str,
    llm_client: object | None,
    available_tools: str,
    locale: str | None = None,
    planning_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    
    if not llm_client:
        return {"messages": [], "plans": []}
    store = SqlitePromptStore()
    context = PromptContext(
        locale=locale or "any",
        address_style="any",
        tone="any",
        channel="any",
        variant="default",
        policy_tier="safe",
    )

    return _story_discover_plan(
        text=text,
        llm_client=llm_client,
        store=store,
        context=context,
        locale=locale or "any",
        planning_context=planning_context,
    )


def format_available_abilities() -> str:
    specs = _load_ability_specs()
    lines: list[str] = [
        "- askQuestion(question:string, slot?:string, bind?:object) -> Use it to ask clarifying questions to the message author user. Only ask questions about missing data, never for missing tools.",
    ]
    for spec in specs:
        intent = str(spec.get("intent_name") or "").strip()
        if not intent:
            continue
        if intent in _DISCOVERY_EXCLUDED_TOOLS:
            continue
        params = spec.get("input_parameters") if isinstance(spec.get("input_parameters"), list) else []
        summary = _ability_summary(spec)
        if params:
            params_desc = ", ".join(
                _render_param_signature(p) for p in params if p.get("name")
            )
            lines.append(f"- {intent}({params_desc}) -> {summary}")
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
        if not tool:
            continue
        if tool in _DISCOVERY_EXCLUDED_TOOLS:
            continue
        raw_params = (
            spec.get("input_parameters")
            if isinstance(spec.get("input_parameters"), list)
            else []
        )
        params: list[dict[str, Any]] = []
        for item in raw_params:
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
                "required_parameters": [
                    p["name"] for p in params if bool(p.get("required"))
                ],
                "input_parameters": params,
            }
        )
    return json.dumps({"tools": tools}, ensure_ascii=False)


def _story_discover_plan(
    *,
    text: str,
    llm_client: object,
    store: SqlitePromptStore,
    context: PromptContext,
    locale: str,
    planning_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exception_history: list[dict[str, Any]] = []
    context_payload: dict[str, Any] = {
        "locale": locale,
        "channel": "any",
        "mode": "planning",
    }
    if isinstance(planning_context, dict):
        planning_facts = planning_context.get("facts")
        if not isinstance(planning_facts, dict):
            planning_facts = {}
        planning_fact_bag = planning_context.get("fact_bag")
        if isinstance(planning_fact_bag, dict):
            planning_facts = {**planning_facts, **planning_fact_bag}
        context_payload.update(
            {
                "original_message": str(
                    planning_context.get("original_message") or text
                ),
                "latest_user_answer": str(
                    planning_context.get("latest_user_answer") or ""
                ),
                "clarifications": planning_context.get("clarifications")
                if isinstance(planning_context.get("clarifications"), list)
                else [],
                "facts": planning_facts,
                "fact_bag": planning_context.get("fact_bag")
                if isinstance(planning_context.get("fact_bag"), dict)
                else planning_facts,
                "last_tool_output": planning_context.get("last_tool_output")
                if isinstance(planning_context.get("last_tool_output"), dict)
                else {},
                "completed_steps": planning_context.get("completed_steps")
                if isinstance(planning_context.get("completed_steps"), list)
                else [],
                "remaining_steps": planning_context.get("remaining_steps")
                if isinstance(planning_context.get("remaining_steps"), list)
                else [],
                "replan_on_answer": bool(planning_context.get("replan_on_answer")),
            }
        )
    tool_menu = _tool_menu_with_ids()
    tool_map = {
        int(item["tool_id"]): str(item["name"])
        for item in tool_menu.get("tools", [])
        if isinstance(item, dict) and isinstance(item.get("tool_id"), int)
    }

    plan_a = _run_step_a_plan_synth(
        text=text,
        llm_client=llm_client,
        store=store,
        context=context,
        context_payload=context_payload,
        exception_history=exception_history,
    )
    if not isinstance(plan_a, dict):
        raise RuntimeError("plan_synth returned non-dict payload")
    interrupt = _extract_planning_interrupt(plan_a)
    if interrupt is not None:
        interrupt = _sanitize_planning_interrupt(interrupt, locale=locale)
    if interrupt is not None:
        logger.info("intent discovery story interrupt source=step_a")
        return _interrupt_discovery_payload(text=text, plan_a=plan_a, interrupt=interrupt)

    plan_b = _run_step_b_tool_binder(
        llm_client=llm_client,
        store=store,
        context=context,
        plan_a=plan_a,
        tool_menu=tool_menu,
        context_payload=context_payload,
        exception_history=exception_history,
    )
    if not isinstance(plan_b, dict):
        raise RuntimeError("tool_binder returned non-dict payload")
    interrupt = _extract_planning_interrupt(plan_b)
    if interrupt is not None:
        interrupt = _sanitize_planning_interrupt(interrupt, locale=locale)
    if interrupt is not None:
        logger.info("intent discovery story interrupt source=step_b")
        return _interrupt_discovery_payload(text=text, plan_a=plan_a, interrupt=interrupt)

    plan_c = _run_step_c_plan_refiner(
        llm_client=llm_client,
        store=store,
        context=context,
        plan_a=plan_a,
        plan_b=plan_b,
        tool_menu=tool_menu,
        context_payload=context_payload,
        exception_history=exception_history,
    )
    if not isinstance(plan_c, dict):
        raise RuntimeError("plan_refiner returned non-dict payload")
    interrupt = _extract_planning_interrupt(plan_c)
    if interrupt is not None:
        interrupt = _sanitize_planning_interrupt(interrupt, locale=locale)
    if interrupt is not None:
        logger.info("intent discovery story interrupt source=step_c")
        return _interrupt_discovery_payload(text=text, plan_a=plan_a, interrupt=interrupt)

    execution_plan = _execution_plan_from_refined(plan_c, tool_map)
    if not execution_plan:
        interrupt = _derive_interrupt_from_bindings(plan_b)
        if interrupt is None:
            interrupt = _derive_interrupt_from_bindings(plan_c)
        interrupt = _sanitize_planning_interrupt(interrupt, locale=locale)
        if interrupt is None:
            raise RuntimeError("plan_refiner produced no executable steps and no interrupt")
        logger.info(
            "intent discovery story no_executable_steps -> planning_interrupt reason=%s",
            interrupt.get("reason"),
        )
        return _interrupt_discovery_payload(text=text, plan_a=plan_a, interrupt=interrupt)
    primary_intention = str(plan_a.get("primary_intention") or "overall").strip() or "overall"
    confidence = str(plan_a.get("confidence") or "medium").strip().lower() or "medium"
    acceptance = _extract_acceptance_criteria(plan_c, plan_a)
    return {
        "messages": [
            {
                "action": primary_intention,
                "message": text,
                "intention": primary_intention,
                "confidence": confidence,
            }
        ],
        "plans": [
            {
                "message_index": 0,
                "acceptanceCriteria": acceptance,
                "executionPlan": execution_plan,
            }
        ],
    }


def _run_step_a_plan_synth(
    *,
    text: str,
    llm_client: object,
    store: SqlitePromptStore,
    context: PromptContext,
    context_payload: dict[str, Any],
    exception_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    system_prompt = _get_template(
        store, _PLAN_SYNTH_KEY, context, _PLAN_SYNTH_SYSTEM_FALLBACK
    )
    user_prompt = _render_template(
        _get_template(
            store,
            _PLAN_SYNTH_KEY + ".user",
            context,
            _PLAN_SYNTH_USER_FALLBACK,
        ),
        {
            "QUESTION_POLICY": _QUESTION_POLICY_BLOCK,
            "MESSAGE_TEXT": text,
            "CONTEXT_JSON": json.dumps(context_payload, ensure_ascii=False),
            "EXCEPTION_HISTORY_JSON": json.dumps(exception_history, ensure_ascii=False),
        },
    )
    raw = _call_llm(llm_client, system_prompt, user_prompt, stage="plan_synth")
    payload = _parse_json(raw)
    if isinstance(payload, dict):
        return payload
    return {
        "plan_text": str(raw or "").strip(),
        "primary_intention": "overall",
        "confidence": "medium",
        "acceptance_criteria": [],
    }


def _run_step_b_tool_binder(
    *,
    llm_client: object,
    store: SqlitePromptStore,
    context: PromptContext,
    plan_a: dict[str, Any],
    tool_menu: dict[str, Any],
    context_payload: dict[str, Any],
    exception_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    system_prompt = _get_template(
        store, _TOOL_BINDER_KEY, context, _TOOL_BINDER_SYSTEM_FALLBACK
    )
    user_prompt = _render_template(
        _get_template(
            store,
            _TOOL_BINDER_KEY + ".user",
            context,
            _TOOL_BINDER_USER_FALLBACK,
        ),
        {
            "QUESTION_POLICY": _QUESTION_POLICY_BLOCK,
            "PLAN": _plan_for_review_text(plan_a, context_payload),
            "TOOL_MENU": _tool_menu_plain_text(tool_menu),
            "CONTEXT_JSON": json.dumps(context_payload, ensure_ascii=False),
            "EXCEPTION_HISTORY_JSON": json.dumps(exception_history, ensure_ascii=False),
        },
    )
    raw = _call_llm(llm_client, system_prompt, user_prompt, stage="tool_binder")
    payload = _parse_json(raw)
    if isinstance(payload, dict):
        return payload
    parsed = _parse_text_plan_to_payload(str(raw or ""), tool_menu)
    if parsed is not None:
        return parsed
    return {"plan_text": str(raw or "").strip()}


def _plan_for_review_text(
    plan: dict[str, Any],
    context_payload: dict[str, Any],
) -> str:
    base = json.dumps(plan, ensure_ascii=False)
    completed = context_payload.get("completed_steps")
    if not isinstance(completed, list) or not completed:
        return base
    notes: list[str] = []
    for step in completed:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or "").strip()
        if not tool:
            continue
        outcome = step.get("outcome")
        outcome_txt = json.dumps(outcome, ensure_ascii=False) if isinstance(outcome, (dict, list)) else str(outcome or "")
        note = f"[NOTE: this has been executed. This was the output: {outcome_txt}]"
        notes.append(f"- {tool} {note}")
    if not notes:
        return base
    return base + "\n\nEXECUTED_STEPS:\n" + "\n".join(notes)


def _tool_menu_plain_text(tool_menu: dict[str, Any]) -> str:
    tools = tool_menu.get("tools") if isinstance(tool_menu.get("tools"), list) else []
    lines: list[str] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        params_desc = ", ".join(f"{k}: {v}" for k, v in params.items())
        returns = str(item.get("returns") or "result").strip()
        desc = _natural_tool_description(name, str(item.get("description") or "").strip())
        signature = f"{name}({params_desc})" if params_desc else f"{name}()"
        lines.append(f"- {signature} -> {desc} Returns: {returns}.")
    return "\n".join(lines)


def _run_step_c_plan_refiner(
    *,
    llm_client: object,
    store: SqlitePromptStore,
    context: PromptContext,
    plan_a: dict[str, Any],
    plan_b: dict[str, Any],
    tool_menu: dict[str, Any],
    context_payload: dict[str, Any],
    exception_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    system_prompt = _get_template(
        store, _PLAN_REFINER_KEY, context, _PLAN_REFINER_SYSTEM_FALLBACK
    )
    user_prompt = _render_template(
        _get_template(
            store,
            _PLAN_REFINER_KEY + ".user",
            context,
            _PLAN_REFINER_USER_FALLBACK,
        ),
        {
            "QUESTION_POLICY": _QUESTION_POLICY_BLOCK,
            "PLAN_A_JSON": json.dumps(plan_a, ensure_ascii=False),
            "BINDINGS_B_JSON": json.dumps(plan_b, ensure_ascii=False),
            "TOOL_MENU_JSON": json.dumps(tool_menu, ensure_ascii=False),
            "CONTEXT_JSON": json.dumps(context_payload, ensure_ascii=False),
            "EXCEPTION_HISTORY_JSON": json.dumps(exception_history, ensure_ascii=False),
        },
    )
    raw = _call_llm(llm_client, system_prompt, user_prompt, stage="plan_refiner")
    payload = _parse_json(raw)
    if isinstance(payload, dict):
        return payload
    parsed = _parse_text_plan_to_payload(str(raw or ""), tool_menu)
    if parsed is not None:
        return parsed
    return {"plan_text": str(raw or "").strip()}


def _parse_text_plan_to_payload(
    raw_text: str,
    tool_menu: dict[str, Any],
) -> dict[str, Any] | None:
    steps = _extract_text_plan_steps(raw_text)
    if not steps:
        return None
    param_names_map = _tool_param_names_map(tool_menu)
    execution: list[dict[str, Any]] = []
    interrupt: dict[str, Any] | None = None
    for step in steps:
        parsed = _parse_tool_call_text(step, param_names_map)
        if parsed is None:
            continue
        tool = str(parsed.get("tool") or "").strip()
        params = parsed.get("parameters") if isinstance(parsed.get("parameters"), dict) else {}
        if tool == "askQuestion":
            question = str(params.get("question") or params.get("arg1") or "").strip()
            if question:
                interrupt = {
                    "tool_id": 0,
                    "tool_name": "askQuestion",
                    "question": question,
                    "slot": str(params.get("slot") or "answer").strip() or "answer",
                    "bind": params.get("bind") if isinstance(params.get("bind"), dict) else {},
                    "missing_data": ["answer"],
                    "reason": "missing_data",
                }
            continue
        execution.append({"tool": tool, "parameters": params, "executed": False})
    if not execution and interrupt is None:
        return {"plan_text": raw_text.strip()}
    payload: dict[str, Any] = {"executionPlan": execution}
    if interrupt is not None:
        payload["planning_interrupt"] = interrupt
    return payload


def _extract_text_plan_steps(raw_text: str) -> list[str]:
    text = str(raw_text or "").strip()
    if not text:
        return []
    steps: list[str] = []
    li_matches = re.findall(r"<li[^>]*>(.*?)</li\s*>", text, flags=re.IGNORECASE | re.DOTALL)
    if li_matches:
        for item in li_matches:
            step = re.sub(r"<[^>]+>", " ", item)
            step = re.sub(r"\s+", " ", step).strip(" -\t\r\n")
            if step:
                steps.append(step)
        return steps
    for line in text.splitlines():
        clean = str(line).strip()
        if not clean:
            continue
        clean = re.sub(r"^\d+[\).\-\s]+", "", clean).strip()
        if clean:
            steps.append(clean)
    return steps


def _tool_param_names_map(tool_menu: dict[str, Any]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    tools = tool_menu.get("tools") if isinstance(tool_menu.get("tools"), list) else []
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        mapping[name] = [str(k) for k in params.keys() if str(k).strip()]
    if "askQuestion" not in mapping:
        mapping["askQuestion"] = ["question", "slot", "bind"]
    return mapping


def _parse_tool_call_text(
    step: str,
    param_names_map: dict[str, list[str]],
) -> dict[str, Any] | None:
    text = str(step or "").strip()
    if not text:
        return None
    match = re.search(r"([A-Za-z0-9_.]+)\s*\((.*)\)", text)
    if match:
        tool = str(match.group(1) or "").strip()
        args_raw = str(match.group(2) or "").strip()
    else:
        tool = text.split(" ", 1)[0].strip()
        args_raw = ""
    if tool not in param_names_map:
        return None
    tokens = _split_tool_args(args_raw)
    param_names = param_names_map.get(tool) or []
    params: dict[str, Any] = {}
    for idx, token in enumerate(tokens):
        name = param_names[idx] if idx < len(param_names) else f"arg{idx + 1}"
        params[name] = _coerce_arg_value(token)
    return {"tool": tool, "parameters": params}


def _split_tool_args(args_raw: str) -> list[str]:
    raw = str(args_raw or "").strip()
    if not raw:
        return []
    out: list[str] = []
    buff: list[str] = []
    quote: str | None = None
    depth = 0
    for ch in raw:
        if quote is not None:
            buff.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            buff.append(ch)
            continue
        if ch in "([{":
            depth += 1
            buff.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            buff.append(ch)
            continue
        if ch == "," and depth == 0:
            token = "".join(buff).strip()
            if token:
                out.append(token)
            buff = []
            continue
        buff.append(ch)
    token = "".join(buff).strip()
    if token:
        out.append(token)
    return out


def _coerce_arg_value(token: str) -> Any:
    value = str(token or "").strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except Exception:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except Exception:
            return value
    parsed = _parse_json(value)
    if isinstance(parsed, (dict, list)):
        return parsed
    return value


def _tool_menu_with_ids() -> dict[str, Any]:
    parsed = _parse_json(format_available_ability_catalog())
    tools_raw = parsed.get("tools") if isinstance(parsed, dict) else []
    tools: list[dict[str, Any]] = []
    tool_id = 0
    saw_ask = False
    for item in tools_raw if isinstance(tools_raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool") or "").strip()
        if not name:
            continue
        if name == "askQuestion":
            saw_ask = True
        params = item.get("input_parameters") if isinstance(item.get("input_parameters"), list) else []
        params_map: dict[str, str] = {}
        for param in params:
            if not isinstance(param, dict):
                continue
            pname = str(param.get("name") or "").strip()
            if not pname:
                continue
            ptype = str(param.get("type") or "string").strip()
            req = bool(param.get("required", False))
            params_map[pname] = f"{ptype}{' (required)' if req else ''}"
        tools.append(
            {
                "tool_id": tool_id,
                "name": name,
                "params": params_map,
                "returns": str(item.get("returns") or "result"),
                "description": str(item.get("description") or ""),
            }
        )
        tool_id += 1
    if not saw_ask:
        tools.insert(
            0,
            {
                "tool_id": 0,
                "name": "askQuestion",
                "params": {
                    "question": "string (required)",
                    "slot": "string",
                    "bind": "object",
                },
                "returns": "user_answer_captured",
                "description": "Ask only for missing end-user data.",
            },
        )
        for idx, item in enumerate(tools):
            item["tool_id"] = idx
    return {"tools": tools}


def _extract_planning_interrupt(payload: dict[str, Any]) -> dict[str, Any] | None:
    interrupt = payload.get("planning_interrupt")
    if not isinstance(interrupt, dict):
        return None
    question = str(interrupt.get("question") or "").strip()
    if not question:
        return None
    return {
        "tool_id": _coerce_tool_id(interrupt.get("tool_id"), default=0),
        "tool_name": "askQuestion",
        "question": question,
        "slot": _normalize_interrupt_slot(interrupt.get("slot")),
        "bind": interrupt.get("bind") if isinstance(interrupt.get("bind"), dict) else {},
        "missing_data": interrupt.get("missing_data") if isinstance(interrupt.get("missing_data"), list) else [],
        "reason": str(interrupt.get("reason") or "missing_data").strip() or "missing_data",
    }


def _execution_plan_from_refined(
    refined: dict[str, Any],
    tool_map: dict[int, str],
) -> list[dict[str, Any]]:
    raw_steps = refined.get("execution_plan")
    if not isinstance(raw_steps, list):
        raw_steps = refined.get("executionPlan")
    if not isinstance(raw_steps, list):
        raw_steps = refined.get("steps")
    if not isinstance(raw_steps, list):
        raw_steps = refined.get("bindings")
    if not isinstance(raw_steps, list):
        return []
    reverse_map = {name: tool_id for tool_id, name in tool_map.items()}
    execution: list[dict[str, Any]] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        tool_id_raw = step.get("tool_id")
        tool_id = _coerce_tool_id(tool_id_raw)
        if tool_id is None:
            name_hint = str(
                step.get("tool_name") or step.get("tool") or step.get("name") or ""
            ).strip()
            if name_hint:
                tool_id = reverse_map.get(name_hint)
        if tool_id is None:
            continue
        if str(step.get("binding_type") or "").strip().upper() == "QUESTION":
            continue
        tool_name = tool_map.get(tool_id)
        if not tool_name or tool_name == "askQuestion":
            continue
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        execution.append(
            {
                "tool": tool_name,
                "parameters": params,
                "executed": False,
            }
        )
    return execution


def _derive_interrupt_from_bindings(payload: dict[str, Any]) -> dict[str, Any] | None:
    bindings = payload.get("bindings")
    if not isinstance(bindings, list):
        steps = payload.get("execution_plan")
        bindings = steps if isinstance(steps, list) else []
    for item in bindings:
        if not isinstance(item, dict):
            continue
        binding_type = str(item.get("binding_type") or "").strip().upper()
        tool_id = _coerce_tool_id(item.get("tool_id"))
        if binding_type == "QUESTION" or tool_id == 0:
            params = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
            question = str(
                params.get("question")
                or item.get("question")
                or item.get("prompt")
                or ""
            ).strip()
            if not question:
                continue
            return {
                "tool_id": 0,
                "tool_name": "askQuestion",
                "question": question,
                "slot": _normalize_interrupt_slot(params.get("slot") or item.get("slot")),
                "bind": params.get("bind") if isinstance(params.get("bind"), dict) else {},
                "missing_data": item.get("missing_data") if isinstance(item.get("missing_data"), list) else [],
                "reason": str(item.get("reason") or "missing_data").strip() or "missing_data",
            }
    return None


def _coerce_tool_id(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.isdigit():
            return int(candidate)
    return default


def _default_story_interrupt_question(locale: str) -> str:
    if str(locale or "").lower().startswith("es"):
        return "¿Me puedes dar el dato faltante para continuar con el plan?"
    return "Could you share the missing detail so I can continue the plan?"


def _normalize_interrupt_slot(raw: Any) -> str:
    slot = str(raw or "answer").strip().lower()
    if not slot or slot in {"none", "null", "n/a", "na", "-"}:
        return "answer"
    return slot


def _sanitize_planning_interrupt(
    interrupt: dict[str, Any] | None,
    *,
    locale: str,
) -> dict[str, Any] | None:
    if not isinstance(interrupt, dict):
        return None
    question = str(interrupt.get("question") or "").strip()
    missing_data = [
        str(item).strip().lower()
        for item in (interrupt.get("missing_data") if isinstance(interrupt.get("missing_data"), list) else [])
        if str(item).strip()
    ]
    if not question:
        question = _default_story_interrupt_question(locale)
    if _is_internal_tool_question(question):
        if any(item in {"time", "datetime", "when", "duration"} for item in missing_data):
            question = (
                "¿Para cuándo quieres el recordatorio?"
                if str(locale or "").lower().startswith("es")
                else "When should I set the reminder?"
            )
        elif any(item in {"task", "message", "title", "what"} for item in missing_data):
            question = (
                "¿Qué exactamente quieres que te recuerde?"
                if str(locale or "").lower().startswith("es")
                else "What exactly should I remind you about?"
            )
        else:
            return None
    sanitized = dict(interrupt)
    sanitized["tool_id"] = 0
    sanitized["tool_name"] = "askQuestion"
    sanitized["question"] = question
    sanitized["slot"] = _normalize_interrupt_slot(interrupt.get("slot"))
    sanitized["bind"] = interrupt.get("bind") if isinstance(interrupt.get("bind"), dict) else {}
    sanitized["missing_data"] = missing_data
    return sanitized


def _extract_acceptance_criteria(
    refined: dict[str, Any],
    plan_a: dict[str, Any],
) -> list[str]:
    items = refined.get("acceptance_criteria")
    if not isinstance(items, list):
        items = plan_a.get("acceptance_criteria")
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            continue
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                out.append(text)
    return out


def _interrupt_discovery_payload(
    *,
    text: str,
    plan_a: dict[str, Any],
    interrupt: dict[str, Any],
) -> dict[str, Any]:
    intention = str(plan_a.get("primary_intention") or "overall").strip() or "overall"
    confidence = str(plan_a.get("confidence") or "medium").strip().lower() or "medium"
    acceptance = _extract_acceptance_criteria({}, plan_a)
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
                "acceptanceCriteria": acceptance,
                "executionPlan": [],
            }
        ],
        "planning_interrupt": interrupt,
    }


def _call_llm(
    llm_client: object,
    system_prompt: str,
    user_prompt: str,
    *,
    stage: str = "unknown",
) -> str:
    started = time.perf_counter()
    logger.info(
        "intent discovery llm start stage=%s system_len=%s user_len=%s",
        stage,
        len(system_prompt or ""),
        len(user_prompt or ""),
    )
    try:
        response = str(
            llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "intent discovery llm done stage=%s elapsed_ms=%s response_len=%s",
            stage,
            elapsed_ms,
            len(response),
        )
        return response
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "intent discovery LLM call failed stage=%s elapsed_ms=%s error=%s",
            stage,
            elapsed_ms,
            exc,
        )
        return ""


def _ability_summary(spec: dict[str, Any]) -> str:
    intent = str(spec.get("intent_name") or "").strip()
    if not intent:
        return "Use it to complete the requested task."
    return _natural_tool_description(intent, "")


def _natural_tool_description(tool_name: str, fallback: str) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return fallback or "Use it to complete the requested task."
    predefined = {
        "askQuestion": "Use it to ask clarifying questions to the message author user. Only ask questions about missing data, never for missing tools.",
        "facts.agent.get": "Use it to find your own agent status, details, given name, and capabilities.",
        "facts.user.get": "Use it to find details about the provided user, including profile and preferences.",
        "reminder.schedule": "Use it to trigger reminders at a specific time and deliver a message in the specified channel.",
        "time.current": "Use it to know and report the current time.",
        "get_status": "Use it to retrieve your current runtime status and health summary.",
        "core.identity.query_agent_name": "Use it to get your agent name and identity label.",
        "core.identity.query_user_name": "Use it to get or confirm the user name.",
        "core.location.current": "Use it to retrieve the current saved location details for a label such as home or work.",
        "core.location.set": "Use it to save a user location, including label and address.",
        "core.users.list": "Use it to list known users in the household profile.",
        "core.onboarding.start": "Use it to start onboarding for the primary user.",
        "core.onboarding.add_user": "Use it to add a secondary user with relationship and role details.",
        "core.onboarding.authorize_channel": "Use it to authorize and link a communication channel for a user.",
        "core.onboarding.introduce_authorize": "Use it to introduce a user and authorize their channel in one flow.",
        "onboarding.location.set_home": "Use it to save or update the home location.",
        "onboarding.location.set_work": "Use it to save or update the work location.",
        "update_preferences": "Use it to update user preferences such as locale, tone, and address style.",
        "greeting": "Use it to send a simple greeting response.",
        "help": "Use it to provide help and guidance on supported capabilities.",
        "meta.capabilities": "Use it to summarize available capabilities.",
        "meta.gaps_list": "Use it to list currently known capability gaps.",
    }
    if name in predefined:
        return predefined[name]
    normalized = name.replace(".", " ").replace("_", " ").strip().lower()
    if not normalized:
        return fallback or "Use it to complete the requested task."
    return f"Use it to {normalized}."


def _ability_intents_covered(spec: dict[str, Any]) -> list[str]:
    intent = str(spec.get("intent_name") or "").strip()
    if not intent:
        return []
    aliases = [intent]
    parts = intent.split(".")
    if len(parts) > 1:
        aliases.append(parts[-1])
    return aliases


def _ability_examples(spec: dict[str, Any]) -> list[str]:
    prompts = spec.get("prompts") if isinstance(spec.get("prompts"), dict) else {}
    outputs = spec.get("outputs") if isinstance(spec.get("outputs"), dict) else {}
    step_sequence = spec.get("step_sequence") if isinstance(spec.get("step_sequence"), list) else []
    examples: list[str] = []
    if prompts:
        examples.append(
            "Prompts: " + ", ".join(sorted(str(k) for k in prompts.keys()))
        )
    if outputs:
        examples.append(
            "Outputs: " + ", ".join(sorted(str(k) for k in outputs.keys()))
        )
    if step_sequence:
        action_names = [
            str(step.get("action") or "").strip()
            for step in step_sequence
            if isinstance(step, dict) and str(step.get("action") or "").strip()
        ]
        if action_names:
            examples.append("Actions: " + ", ".join(action_names))
    if not examples:
        examples.append("No explicit examples provided.")
    return examples


def _ability_actions(spec: dict[str, Any]) -> list[str]:
    step_sequence = spec.get("step_sequence") if isinstance(spec.get("step_sequence"), list) else []
    actions: list[str] = []
    for step in step_sequence:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip()
        if action:
            actions.append(action)
    return actions


def _render_param_signature(param: dict[str, Any]) -> str:
    name = str(param.get("name") or "").strip()
    ptype = str(param.get("type") or "string").strip()
    required = bool(param.get("required", False))
    suffix = "" if required else "?"
    return f"{name}{suffix}:{ptype}"


def _is_internal_tool_question(question: str) -> bool:
    normalized = str(question or "").lower().strip()
    if not normalized:
        return True
    forbidden = (
        "tool",
        "function",
        "api",
        "endpoint",
        "method",
        "setreminder",
        "reminderset",
        "notification tool",
    )
    if any(token in normalized for token in forbidden):
        return True
    return bool(
        re.search(r"\b(confirm|choose|pick|specify)\b.{0,24}\b(tool|function|api)\b", normalized)
    )


def _io_channel_catalog() -> dict[str, Any]:
    try:
        registry = get_io_registry()
        senses = sorted(str(name) for name in registry.senses.keys())
        extremities = sorted(str(name) for name in registry.extremities.keys())
    except Exception:
        senses = []
        extremities = []
    return {
        "senses": senses,
        "extremities": extremities,
        "notes": [
            "senses = inbound channels that read/normalize external input",
            "extremities = outbound channels that deliver responses/actions",
        ],
    }


def _get_template(store: SqlitePromptStore, key: str, context: PromptContext, fallback: str) -> str:
    match = store.get_template(key, context)
    if match and match.template:
        return str(match.template)
    return fallback


def _render_template(template: str, variables: dict[str, Any]) -> str:
    rendered = template
    for name, value in variables.items():
        rendered = rendered.replace("{" + name + "}", str(value))
    return rendered


def _parse_chunks(raw: str) -> list[dict[str, Any]]:
    payload = _parse_json(raw)
    if isinstance(payload, list):
        return [_normalize_chunk(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            return [_normalize_chunk(item) for item in chunks if isinstance(item, dict)]
    return []


def _parse_acceptance(raw: str) -> list[str]:
    payload = _parse_json(raw)
    if isinstance(payload, dict):
        criteria = payload.get("acceptanceCriteria")
        if isinstance(criteria, list):
            return [str(item) for item in criteria if item]
    if isinstance(payload, list):
        return [str(item) for item in payload if item]
    return []


def _parse_execution_plan(raw: str) -> list[dict[str, Any]]:
    payload = _parse_json(raw)
    if isinstance(payload, dict):
        plan = payload.get("executionPlan")
        if isinstance(plan, list):
            return [item for item in plan if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _repair_invalid_execution_plan(
    *,
    execution_plan: list[dict[str, Any]],
    llm_client: object,
    store: SqlitePromptStore,
    context: PromptContext,
    chunk_text: str,
    intention: str,
    acceptance: list[str],
    available_tools: str,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    current = execution_plan
    issue = _validate_execution_plan(current)
    retries = 0
    while issue is not None and retries < max_retries:
        logger.info(
            "intent discovery plan repair attempt=%s issue=%s",
            retries + 1,
            issue.get("code"),
        )
        retries += 1
        critic_system = _get_template(
            store,
            _PLAN_CRITIC_KEY,
            context,
            _PLAN_CRITIC_SYSTEM_FALLBACK,
        )
        critic_user = _build_plan_critic_user_prompt(
            chunk_text=chunk_text,
            intention=intention,
            acceptance=acceptance,
            available_tools=available_tools,
            invalid_plan=current,
            issue=issue,
        )
        raw = _call_llm(
            llm_client,
            critic_system,
            critic_user,
            stage=f"plan_critic[{retries}]",
        )
        current = _parse_execution_plan(raw)
        issue = _validate_execution_plan(current)
    if issue is not None:
        logger.info(
            "intent discovery plan validation failed issue=%s detail=%s",
            issue.get("code"),
            issue.get("message"),
        )
    return current


def _build_plan_critic_user_prompt(
    *,
    chunk_text: str,
    intention: str,
    acceptance: list[str],
    available_tools: str,
    invalid_plan: list[dict[str, Any]],
    issue: dict[str, str],
) -> str:
    return (
        "Repair this executionPlan.\n"
        "Rules:\n"
        "- Return JSON with shape {\"executionPlan\": [{\"tool\"|\"action\": \"...\", \"parameters\": {...}}]}.\n"
        "- Keep the same intent and acceptance criteria direction.\n"
        "- If missing data, use askQuestion with user-facing question only.\n"
        "- Never ask user to choose internal tool/function names.\n"
        "- Do not output explanations.\n\n"
        f"Message:\n{chunk_text}\n\n"
        f"Intention:\n{intention}\n\n"
        f"Acceptance:\n{json.dumps(acceptance, ensure_ascii=False)}\n\n"
        f"Validation issue:\n{json.dumps(issue, ensure_ascii=False)}\n\n"
        f"Invalid plan:\n{json.dumps(invalid_plan, ensure_ascii=False)}\n\n"
        f"AVAILABLE TOOLS:\n{available_tools}\n"
    )


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
        for key, value in params.items():
            if value is None:
                return {"code": "EMPTY_PARAMETER", "message": f"step[{idx}] parameter '{key}' is null."}
            if isinstance(value, str):
                if not value.strip():
                    return {"code": "EMPTY_PARAMETER", "message": f"step[{idx}] parameter '{key}' is empty."}
                if any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in _PLACEHOLDER_PATTERNS):
                    return {
                        "code": "PLACEHOLDER_PARAMETER",
                        "message": f"step[{idx}] parameter '{key}' contains placeholder text.",
                    }
        if tool == "askQuestion":
            question = str(params.get("question") or "").strip()
            if not question:
                return {"code": "ASK_QUESTION_MISSING_QUESTION", "message": f"step[{idx}] askQuestion needs question."}
            if _is_internal_tool_question(question):
                return {"code": "ASK_QUESTION_INTERNAL_TOOL", "message": f"step[{idx}] asks about internal tool selection."}
        else:
            if not params:
                return {
                    "code": "NON_ASKQUESTION_EMPTY_PARAMETERS",
                    "message": f"step[{idx}] non-askQuestion actions must include parameters.",
                }
    return None


def _parse_json(raw: str) -> Any:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    salvaged = _extract_first_json(candidate)
    if salvaged is None:
        return None
    sanitized = re.sub(r",\s*([}\]])", r"\1", salvaged)
    try:
        return json.loads(sanitized)
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


def _normalize_chunk(item: dict[str, Any]) -> dict[str, Any]:
    chunk_text = str(item.get("chunk") or "").strip()
    action = str(item.get("action") or "").strip()
    intention = str(item.get("intention") or "").strip()
    if not chunk_text and action:
        chunk_text = action
    if not intention and action:
        intention = action
    normalized = dict(item)
    normalized["chunk"] = chunk_text
    normalized["intention"] = intention
    return normalized


def _load_ability_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    specs.extend(_load_specs_file())
    specs.extend(_load_specs_db())
    unique: dict[str, dict[str, Any]] = {}
    for spec in specs:
        intent = str(spec.get("intent_name") or "").strip()
        if not intent:
            continue
        unique[intent] = spec
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
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("ability spec load failed path=%s error=%s", path, exc)
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _load_specs_db() -> list[dict[str, Any]]:
    try:
        from alphonse.agent.cognition.abilities.store import AbilitySpecStore

        return AbilitySpecStore().list_enabled_specs()
    except Exception as exc:
        logger.warning("ability spec db load failed error=%s", exc)
        return []
