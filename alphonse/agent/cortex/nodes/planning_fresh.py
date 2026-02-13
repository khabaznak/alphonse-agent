from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FreshPlanningDeps:
    run_capability_gap_tool: Callable[..., dict[str, Any]]
    format_available_abilities: Callable[[], str]
    planning_context_for_discovery: Callable[[dict[str, Any], str], dict[str, Any] | None]
    discover_plan: Callable[..., dict[str, Any] | None]
    locale_for_state: Callable[[dict[str, Any]], str]
    dispatch_discovery_result: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class DispatchDiscoveryDeps:
    run_capability_gap_tool: Callable[..., dict[str, Any]]
    build_discovery_loop_state: Callable[..., dict[str, Any]]


def run_fresh_discovery_for_message(
    *,
    state: dict[str, Any],
    llm_client: Any,
    text: str,
    deps: FreshPlanningDeps,
) -> dict[str, Any] | None:
    if not llm_client:
        return deps.run_capability_gap_tool(state, llm_client=None, reason="no_llm_client")

    if not text:
        return {}

    available_tools = deps.format_available_abilities()
    planning_context = deps.planning_context_for_discovery(state, text)
    discovery = deps.discover_plan(
        text=text,
        llm_client=llm_client,
        available_tools=available_tools,
        locale=deps.locale_for_state(state),
        planning_context=planning_context,
    )
    return deps.dispatch_discovery_result(
        state=state,
        llm_client=llm_client,
        source_text=text,
        discovery=discovery,
    )


def dispatch_discovery_result(
    *,
    state: dict[str, Any],
    llm_client: Any,
    source_text: str,
    discovery: dict[str, Any] | None,
    deps: DispatchDiscoveryDeps,
) -> dict[str, Any]:
    discovery = coerce_planning_interrupt_to_discovery(discovery)
    if not isinstance(discovery, dict):
        logger.info(
            "cortex planning dispatch non-dict chat_id=%s correlation_id=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
        )

    plans = discovery.get("plans") if isinstance(discovery, dict) else None
    if not isinstance(plans, list):
        logger.info(
            "cortex planning dispatch invalid plans chat_id=%s correlation_id=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
        )
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason="invalid_plan_payload",
        )

    loop_state = deps.build_discovery_loop_state(discovery, source_message=source_text)
    if not loop_state.get("steps"):
        logger.info(
            "cortex planning dispatch empty steps chat_id=%s correlation_id=%s",
            state.get("chat_id"),
            state.get("correlation_id"),
        )
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason="empty_execution_plan",
        )

    step_count = len(loop_state.get("steps")) if isinstance(loop_state.get("steps"), list) else 0
    logger.info(
        "cortex planning dispatch accepted chat_id=%s correlation_id=%s steps=%s",
        state.get("chat_id"),
        state.get("correlation_id"),
        step_count,
    )
    state["ability_state"] = loop_state
    return {"ability_state": loop_state, "pending_interaction": None}


def coerce_planning_interrupt_to_discovery(
    discovery: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(discovery, dict):
        return discovery
    plans = discovery.get("plans")
    if isinstance(plans, list) and plans:
        return discovery
    interrupt = discovery.get("planning_interrupt")
    if not isinstance(interrupt, dict):
        return discovery
    question = str(interrupt.get("question") or "").strip()
    slot = str(interrupt.get("slot") or "answer").strip() or "answer"
    bind = interrupt.get("bind") if isinstance(interrupt.get("bind"), dict) else {}
    step_params: dict[str, Any] = {"slot": slot, "bind": bind}
    if question:
        step_params["question"] = question
    synthetic_plan = {
        "message_index": 0,
        "acceptanceCriteria": ["User answers the clarification question."],
        "executionPlan": [
            {
                "tool": "askQuestion",
                "parameters": step_params,
                "executed": False,
            }
        ],
    }
    normalized = dict(discovery)
    normalized["plans"] = [synthetic_plan]
    return normalized
