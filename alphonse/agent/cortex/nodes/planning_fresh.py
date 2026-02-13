from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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
    log_discovery_plan: Callable[[dict[str, Any], dict[str, Any] | None], None]
    run_planning_interrupt: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
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
    deps.log_discovery_plan(state, discovery if isinstance(discovery, dict) else None)
    if isinstance(discovery, dict):
        interrupt = discovery.get("planning_interrupt")
        if isinstance(interrupt, dict):
            return deps.run_planning_interrupt(state, interrupt)

    plans = discovery.get("plans") if isinstance(discovery, dict) else None
    if not isinstance(plans, list):
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason="invalid_plan_payload",
        )

    loop_state = deps.build_discovery_loop_state(discovery, source_message=source_text)
    if not loop_state.get("steps"):
        return deps.run_capability_gap_tool(
            state,
            llm_client=llm_client,
            reason="empty_execution_plan",
        )

    state["ability_state"] = loop_state
    return {"ability_state": loop_state, "pending_interaction": None}
