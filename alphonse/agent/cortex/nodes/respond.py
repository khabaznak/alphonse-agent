from __future__ import annotations

from typing import Any, Callable

from alphonse.agent.cognition.response_keys import render_response_key
from alphonse.agent.cortex.nodes.capability_gap import has_capability_gap_plan

def compose_response_from_state(state: dict[str, Any]) -> str:
    key = state.get("response_key")
    if not isinstance(key, str) or not key.strip():
        return str(state.get("response_text") or "").strip()
    return render_response_key(
        key,
        state.get("response_vars") if isinstance(state.get("response_vars"), dict) else None,
        locale=str(state.get("locale") or ""),
    )


def respond_node_impl(
    state: dict[str, Any],
    llm_client: Any,
    *,
    run_planning_cycle: Callable[[dict[str, Any], Any], dict[str, Any] | None],
    ability_registry_getter: Callable[[], Any],
    tool_registry: Any,
    run_capability_gap_tool: Callable[..., dict[str, Any]],
    emit_transition_event: Callable[[dict[str, Any], str, dict[str, Any] | None], None],
) -> dict[str, Any]:
    if state.get("response_text") or state.get("response_key"):
        return {}
    emit_transition_event(state, "thinking")
    discovery = run_planning_cycle(state, llm_client)
    if discovery:
        if isinstance(discovery, dict):
            pending = discovery.get("pending_interaction")
            if has_capability_gap_plan(discovery):
                emit_transition_event(state, "failed")
            elif pending:
                emit_transition_event(state, "waiting_user")
            else:
                emit_transition_event(state, "done")
        return discovery
    intent = state.get("intent")
    if intent:
        ability = ability_registry_getter().get(str(intent))
        if ability is not None:
            try:
                emit_transition_event(state, "executing", {"tool": str(intent)})
                return ability.execute(state, tool_registry)
            except Exception:
                return run_capability_gap_tool(
                    state,
                    llm_client=llm_client,
                    reason="ability_execution_exception",
                )
    return {}


def respond_node(
    arg: Any,
    *,
    impl: Callable[[dict[str, Any], Any], dict[str, Any]],
):
    # Backward-compatible call shape for tests: respond_node(state_dict)
    if isinstance(arg, dict):
        return impl(arg, None)

    llm_client = arg

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        return impl(state, llm_client)

    return _node


def respond_finalize_node(
    state: dict[str, Any],
    *,
    emit_transition_event: Callable[[dict[str, Any], str, dict[str, Any] | None], None],
) -> dict[str, Any]:
    pending = state.get("pending_interaction")
    if has_capability_gap_plan(state):
        emit_transition_event(state, "failed")
        return {}
    if pending:
        emit_transition_event(state, "waiting_user")
        return {}
    if state.get("response_text") or state.get("response_key"):
        emit_transition_event(state, "done")
    return {}
