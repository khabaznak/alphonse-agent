from __future__ import annotations

import json
import logging
from typing import Any, Callable

from alphonse.agent.cognition.tool_schemas import planner_tool_schemas
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_REPAIR_USER_TEMPLATE
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_SYSTEM_PROMPT
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_USER_TEMPLATE
from alphonse.agent.cortex.task_mode.prompt_templates import render_pdca_prompt
from alphonse.agent.cortex.task_mode.types import NextStepProposal
from alphonse.agent.session.day_state import render_recent_conversation_block

_NEXT_STEP_MAX_ATTEMPTS = 2

# Strict JSON Schema for NextStepProposal structured output
_NEXT_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind"],
    "properties": {
        "kind": {"type": "string", "enum": ["ask_user", "call_tool", "finish"]},
        "question": {"type": "string"},
        "tool_name": {"type": "string"},
        "args": {"type": "object"},
        "final_text": {"type": "string"},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
    },
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"const": "ask_user"},
                "question": {"type": "string", "minLength": 1},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["kind", "question"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"const": "call_tool"},
                "tool_name": {"type": "string", "minLength": 1},
                "args": {"type": "object"},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["kind", "tool_name", "args"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"const": "finish"},
                "final_text": {"type": "string", "minLength": 1},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["kind", "final_text"],
        },
    ],
}


def build_next_step_node_impl(
    state: dict[str, Any],
    *,
    tool_registry: Any,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    next_step_id: Callable[[dict[str, Any]], str],
    task_plan: Callable[[dict[str, Any]], dict[str, Any]],
    has_acceptance_criteria: Callable[[dict[str, Any]], bool],
    normalize_acceptance_criteria_values: Callable[[Any], list[str]],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "plan"
    corr = correlation_id(state)
    llm_client = state.get("_llm_client")
    proposal, parse_failed, diagnostics = _propose_next_step_with_llm(
        llm_client=llm_client,
        state=state,
        task_state=task_state,
        tool_registry=tool_registry,
    )
    if diagnostics:
        for item in diagnostics:
            if not isinstance(item, dict):
                continue
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="next_step_node",
                event="graph.next_step.parse_invalid",
                attempt=int(item.get("attempt") or 0),
                parse_error_type=str(item.get("parse_error_type") or ""),
                validation_errors=item.get("validation_errors"),
                raw_output_preview=str(item.get("raw_output_preview") or ""),
                provider=str(item.get("provider") or ""),
                model=str(item.get("model") or ""),
            )
    if not has_acceptance_criteria(task_state):
        proposed_criteria = normalize_acceptance_criteria_values(
            proposal.get("acceptance_criteria") if isinstance(proposal, dict) else None
        )
        if proposed_criteria:
            task_state["acceptance_criteria"] = proposed_criteria
            append_trace_event(
                task_state,
                {
                    "type": "acceptance_criteria_derived",
                    "summary": "Derived acceptance criteria from planning context.",
                    "correlation_id": corr,
                },
            )

    if parse_failed:
        task_state["status"] = "failed"
        task_state["next_user_question"] = None
        provider, model = _provider_model_from_state(state)
        attempts = int((diagnostics[-1] if diagnostics else {}).get("attempt") or _NEXT_STEP_MAX_ATTEMPTS)
        task_state["last_validation_error"] = {
            "reason": "next_step_parse_failed",
            "attempts": attempts,
            "schema": "NextStepProposal",
            "provider": provider,
            "model": model,
        }
        append_trace_event(
            task_state,
            {
                "type": "parse_failed",
                "summary": "Next-step parse failed; task paused as internal degradation.",
                "correlation_id": corr,
            },
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="next_step_node",
            event="graph.next_step.degraded",
            reason="next_step_parse_failed",
            attempts=attempts,
            provider=provider,
            model=model,
        )
        logger.warning(
            "task_mode next_step degraded correlation_id=%s reason=next_step_parse_failed attempts=%s provider=%s model=%s",
            corr,
            attempts,
            provider,
            model,
        )
        return {"task_state": task_state}
    step_id = next_step_id(task_state)
    step_entry = {
        "step_id": step_id,
        "proposal": proposal,
        "status": "proposed",
    }
    plan = task_plan(task_state)
    plan["steps"].append(step_entry)
    plan["current_step_id"] = step_id
    task_state["next_user_question"] = None
    task_state["last_validation_error"] = None
    append_trace_event(
        task_state,
        {
            "type": "proposal_created",
            "summary": f"Created {_proposal_summary(proposal)} ({step_id}).",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode next_step proposal correlation_id=%s step_id=%s kind=%s summary=%s parse_failed=%s",
        corr,
        step_id,
        str(proposal.get("kind") or ""),
        _proposal_summary(proposal),
        parse_failed,
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="next_step_node",
        event="graph.next_step.proposed",
        step_id=step_id,
        kind=str(proposal.get("kind") or ""),
        parse_failed=parse_failed,
        summary=_proposal_summary(proposal),
    )
    return {"task_state": task_state}


def route_after_next_step_impl(
    state: dict[str, Any],
    *,
    correlation_id: Callable[[dict[str, Any]], str | None],
    logger: logging.Logger,
) -> str:
    task_state = state.get("task_state")
    if isinstance(task_state, dict) and str(task_state.get("status") or "").strip().lower() in {"waiting_user", "failed"}:
        logger.info(
            "task_mode route_after_next_step correlation_id=%s route=respond_node reason=terminal_status",
            correlation_id(state),
        )
        return "respond_node"
    logger.info(
        "task_mode route_after_next_step correlation_id=%s route=execute_step_node",
        correlation_id(state),
    )
    return "execute_step_node"


def _propose_next_step_with_llm(
    *,
    llm_client: Any,
    state: dict[str, Any],
    task_state: dict[str, Any],
    tool_registry: Any,
) -> tuple[NextStepProposal, bool, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []

    tool_menu = _build_tool_menu(tool_registry)
    working_view = _build_working_state_view(task_state)
    injected_blocks: list[str] = []
    philosophy = str(state.get("philosophy_block") or "").strip()
    soul = str(state.get("soul_block") or "").strip()
    agents = str(state.get("agents_block") or "").strip()
    if philosophy:
        injected_blocks.append("## Philosophy\n" + philosophy)
    if soul:
        injected_blocks.append("## SOUL\n" + soul)
    if agents:
        injected_blocks.append("## AGENTS\n" + agents)
    injected_guidance = "\n\n".join(injected_blocks).strip()
    user_prompt_body = render_pdca_prompt(
        NEXT_STEP_USER_TEMPLATE,
        {
            "TOOL_MENU": tool_menu,
            "INJECTED_GUIDANCE_BLOCK": injected_guidance,
            "WORKING_STATE_VIEW_JSON": json.dumps(working_view, ensure_ascii=False),
        },
    )
    recent_conversation_block = str(state.get("recent_conversation_block") or "").strip()
    if not recent_conversation_block:
        session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else None
        if session_state:
            recent_conversation_block = render_recent_conversation_block(session_state)
    user_prompt = (
        f"{recent_conversation_block}\n\n{user_prompt_body}".strip()
        if recent_conversation_block
        else user_prompt_body
    )

    structured_supported, parsed_structured = _call_llm_structured(
        llm_client=llm_client,
        system_prompt=NEXT_STEP_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    if structured_supported:
        normalized = _normalize_next_step_proposal(parsed_structured)
        if normalized is not None:
            return normalized, False, diagnostics
        diagnostics.append(
            _build_parse_diagnostic(
                state=state,
                attempt=1,
                parse_error_type="structured_payload_invalid",
                validation_errors=["structured_output_failed_schema_validation"],
                raw_output=parsed_structured,
            )
        )
    else:
        raw = _call_llm_text(
            llm_client=llm_client,
            system_prompt=NEXT_STEP_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        parsed = _parse_json_payload(raw)
        normalized = _normalize_next_step_proposal(parsed)
        if normalized is not None:
            return normalized, False, diagnostics
        diagnostics.append(
            _build_parse_diagnostic(
                state=state,
                attempt=1,
                parse_error_type="text_payload_invalid",
                validation_errors=["text_output_not_valid_next_step_json"],
                raw_output=raw,
            )
        )

    repair_prompt = _build_next_step_repair_prompt(
        original_user_prompt=user_prompt,
        latest_diagnostic=diagnostics[-1] if diagnostics else {},
    )
    repair_raw = _call_llm_text(
        llm_client=llm_client,
        system_prompt=NEXT_STEP_SYSTEM_PROMPT,
        user_prompt=repair_prompt,
    )
    repair_parsed = _parse_json_payload(repair_raw)
    repaired = _normalize_next_step_proposal(repair_parsed)
    if repaired is not None:
        return repaired, False, diagnostics
    diagnostics.append(
        _build_parse_diagnostic(
            state=state,
            attempt=2,
            parse_error_type="repair_payload_invalid",
            validation_errors=["repair_output_not_valid_next_step_json"],
            raw_output=repair_raw,
        )
    )

    return {"kind": "ask_user", "question": ""}, True, diagnostics


def _provider_model_from_state(state: dict[str, Any]) -> tuple[str | None, str | None]:
    llm_client = state.get("_llm_client")
    provider = str(getattr(llm_client, "provider", "") or "").strip() or None
    model = str(getattr(llm_client, "model", "") or "").strip() or None
    return provider, model


def _build_parse_diagnostic(
    *,
    state: dict[str, Any],
    attempt: int,
    parse_error_type: str,
    validation_errors: list[str],
    raw_output: Any,
) -> dict[str, Any]:
    provider, model = _provider_model_from_state(state)
    preview = _truncate_text(_redact_secrets(str(raw_output or "")), limit=280)
    return {
        "attempt": int(attempt),
        "parse_error_type": str(parse_error_type or "next_step_parse_invalid"),
        "validation_errors": validation_errors,
        "raw_output_preview": preview,
        "provider": provider,
        "model": model,
    }


def _build_next_step_repair_prompt(*, original_user_prompt: str, latest_diagnostic: dict[str, Any]) -> str:
    parse_error_type = str((latest_diagnostic or {}).get("parse_error_type") or "").strip()
    validation_errors = latest_diagnostic.get("validation_errors")
    lines = []
    if isinstance(validation_errors, list):
        lines = [f"- {str(item)}" for item in validation_errors if str(item).strip()]
    if not lines:
        lines = ["- next_step_output_invalid"]
    return render_pdca_prompt(
        NEXT_STEP_REPAIR_USER_TEMPLATE,
        {
            "ORIGINAL_USER_PROMPT": original_user_prompt,
            "PARSE_ERROR_TYPE": parse_error_type or "unknown",
            "VALIDATION_ERRORS_LINES": "\n".join(lines),
        },
    )


def _redact_secrets(text: str) -> str:
    value = str(text or "")
    lowered = value.lower()
    redacted = value
    secret_tokens = ("password", "pass", "token", "api_key", "secret")
    if any(token in lowered for token in secret_tokens):
        for token in secret_tokens:
            redacted = redacted.replace(token, "[redacted_key]")
    return redacted


def _truncate_text(text: str, *, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit]


def _build_working_state_view(task_state: dict[str, Any]) -> dict[str, Any]:
    facts = task_state.get("facts")
    relevant_facts = dict(facts) if isinstance(facts, dict) else {}
    if len(relevant_facts) > 8:
        keys = sorted(relevant_facts.keys())[-8:]
        relevant_facts = {key: relevant_facts[key] for key in keys}
    acceptance_criteria = _normalize_acceptance_criteria_values(task_state.get("acceptance_criteria"))
    failure_diagnostics = _latest_failure_diagnostics(task_state)
    return {
        "goal": str(task_state.get("goal") or "").strip(),
        "acceptance_criteria": acceptance_criteria,
        "relevant_facts": relevant_facts,
        "latest_failure_diagnostics": failure_diagnostics,
        "execution_eval": task_state.get("execution_eval") if isinstance(task_state.get("execution_eval"), dict) else {},
        "last_validation_error": task_state.get("last_validation_error"),
        "repair_attempts": int(task_state.get("repair_attempts") or 0),
        "pending_question": str(task_state.get("next_user_question") or "").strip() or None,
    }


def _latest_failure_diagnostics(task_state: dict[str, Any]) -> dict[str, Any]:
    plan = task_state.get("plan") if isinstance(task_state.get("plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    facts = task_state.get("facts") if isinstance(task_state.get("facts"), dict) else {}
    for raw_step in reversed(steps):
        if not isinstance(raw_step, dict):
            continue
        if str(raw_step.get("status") or "").strip().lower() != "failed":
            continue
        step_id = str(raw_step.get("step_id") or "").strip()
        proposal = raw_step.get("proposal") if isinstance(raw_step.get("proposal"), dict) else {}
        tool_name = str(proposal.get("tool_name") or "").strip()
        args = proposal.get("args") if isinstance(proposal.get("args"), dict) else {}
        fact_entry = facts.get(step_id) if step_id else None
        result = fact_entry.get("result") if isinstance(fact_entry, dict) and isinstance(fact_entry.get("result"), dict) else {}
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        return {
            "step_id": step_id,
            "tool_name": tool_name,
            "args": args,
            "error_code": str(error.get("code") or "").strip(),
            "error_message": str(error.get("message") or "").strip(),
        }
    return {}


def _build_tool_menu(tool_registry: Any) -> str:
    descriptions = _tool_descriptions()
    keys = sorted(descriptions.keys())
    lines: list[str] = []
    for name in keys:
        if not _tool_exists(tool_registry, name):
            continue
        summary = descriptions.get(name) or "Tool available."
        lines.append(f"- `{name}`: {summary}")
    return "\n".join(lines) or "- (no tools)"


def _tool_descriptions() -> dict[str, str]:
    menu: dict[str, str] = {}
    for schema in planner_tool_schemas():
        fn = schema.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        description = str(fn.get("description") or "Tool available.").strip()
        menu[name] = description
    return menu


def _tool_exists(tool_registry: Any, name: str) -> bool:
    if hasattr(tool_registry, "get"):
        return tool_registry.get(name) is not None
    return False


def _call_llm_structured(
    *,
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
) -> tuple[bool, dict[str, Any] | None]:
    complete_json = getattr(llm_client, "complete_json", None)
    if callable(complete_json):
        try:
            payload = complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_NEXT_STEP_SCHEMA,
            )
            if isinstance(payload, dict):
                return True, payload
        except Exception:
            return True, None
        return True, None
    complete_with_schema = getattr(llm_client, "complete_with_schema", None)
    if callable(complete_with_schema):
        try:
            payload = complete_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_NEXT_STEP_SCHEMA,
            )
            if isinstance(payload, dict):
                return True, payload
        except Exception:
            return True, None
        return True, None
    return False, None


def _call_llm_text(*, llm_client: Any, system_prompt: str, user_prompt: str) -> str:
    complete = getattr(llm_client, "complete", None)
    if not callable(complete):
        return ""
    try:
        return str(complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except TypeError:
        try:
            return str(complete(system_prompt, user_prompt))
        except Exception:
            return ""
    except Exception:
        return ""


def _parse_json_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    parsed = _json_loads(candidate)
    if isinstance(parsed, dict):
        return parsed
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        parsed = _json_loads(candidate[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _normalize_next_step_proposal(payload: Any) -> NextStepProposal | None:
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("kind") or "").strip()
    criteria = _normalize_acceptance_criteria_values(payload.get("acceptance_criteria"))
    if kind == "ask_user":
        question = str(payload.get("question") or "").strip()
        if not question:
            return None
        out: NextStepProposal = {"kind": "ask_user", "question": question}
        if criteria:
            out["acceptance_criteria"] = criteria
        return out
    if kind == "call_tool":
        tool_name = str(payload.get("tool_name") or "").strip()
        args = payload.get("args")
        if not tool_name or not isinstance(args, dict):
            return None
        out = {"kind": "call_tool", "tool_name": tool_name, "args": dict(args)}
        if criteria:
            out["acceptance_criteria"] = criteria
        return out
    if kind == "finish":
        final_text = str(payload.get("final_text") or "").strip()
        if not final_text:
            return None
        out = {"kind": "finish", "final_text": final_text}
        if criteria:
            out["acceptance_criteria"] = criteria
        return out
    return None


def _normalize_acceptance_criteria_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text[:180])
        if len(out) >= 8:
            break
    return out


def _proposal_summary(proposal: dict[str, Any]) -> str:
    kind = str(proposal.get("kind") or "").strip()
    if kind == "call_tool":
        tool = str(proposal.get("tool_name") or "").strip()
        return f"call_tool:{tool or 'unknown'}"
    if kind == "ask_user":
        question = str(proposal.get("question") or "").strip()
        return f"ask_user:{question[:48]}"
    if kind == "finish":
        text = str(proposal.get("final_text") or "").strip()
        return f"finish:{text[:48]}"
    return kind or "unknown"
