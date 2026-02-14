from __future__ import annotations

from typing import Any, Callable

from alphonse.agent.cortex.nodes.capability_gap import has_capability_gap_plan
from alphonse.agent.cortex.nodes.telemetry import emit_brain_state

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
    emit_brain_state(
        state=state,
        node="respond_node",
        updates={},
        stage="start",
    )

    def _return(payload: dict[str, Any]) -> dict[str, Any]:
        return emit_brain_state(
            state=state,
            node="respond_node",
            updates=payload,
        )

    if state.get("response_text"):
        return _return({})
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
        return _return(discovery if isinstance(discovery, dict) else {})
    intent = state.get("intent")
    if intent:
        ability = ability_registry_getter().get(str(intent))
        if ability is not None:
            try:
                emit_transition_event(state, "executing", {"tool": str(intent)})
                result = ability.execute(state, tool_registry)
                return _return(result if isinstance(result, dict) else {})
            except Exception:
                return _return(run_capability_gap_tool(
                    state,
                    llm_client=llm_client,
                    reason="ability_execution_exception",
                ))
    return _return({})


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
    emit_brain_state(
        state=state,
        node="respond_node",
        updates={},
        stage="start",
    )

    def _return(payload: dict[str, Any]) -> dict[str, Any]:
        return emit_brain_state(
            state=state,
            node="respond_node",
            updates=payload,
        )

    pending = state.get("pending_interaction")
    task_state = state.get("task_state") if isinstance(state.get("task_state"), dict) else None
    task_status = str((task_state or {}).get("status") or "").strip().lower() if isinstance(task_state, dict) else ""
    if has_capability_gap_plan(state):
        emit_transition_event(state, "failed")
        return _return({})
    if pending:
        emit_transition_event(state, "waiting_user")
        return _return({})
    if isinstance(task_state, dict):
        task_response = _render_task_response(state=state, task_state=task_state)
        if isinstance(task_response, str) and task_response.strip():
            if task_status == "failed":
                emit_transition_event(state, "failed")
            elif task_status == "waiting_user":
                emit_transition_event(state, "waiting_user")
            else:
                emit_transition_event(state, "done")
            return _return({"response_text": task_response})
    if state.get("response_text"):
        emit_transition_event(state, "done")
    return _return({})


def _render_task_response(*, state: dict[str, Any], task_state: dict[str, Any]) -> str | None:
    status = str(task_state.get("status") or "").strip().lower()
    locale = str(state.get("locale") or "en-US")
    if status == "waiting_user":
        question = str(task_state.get("next_user_question") or "").strip()
        return question or ("Que detalle te falta?" if locale.lower().startswith("es") else "What detail is missing?")
    if status == "failed":
        return (
            "No pude completar esa tarea."
            if locale.lower().startswith("es")
            else "I could not complete that task."
        )
    if status == "done":
        outcome = task_state.get("outcome")
        if isinstance(outcome, dict) and str(outcome.get("kind") or "") == "reminder_created":
            evidence = outcome.get("evidence")
            if isinstance(evidence, dict):
                fire_at = str(evidence.get("fire_at") or "").strip()
                reminder_id = str(evidence.get("reminder_id") or "").strip()
                message = str(evidence.get("message") or "").strip()
                if locale.lower().startswith("es"):
                    if message:
                        return f"Listo, te recordare '{message}' en {fire_at}. ID: {reminder_id}."
                    return f"Listo, recordatorio programado para {fire_at}. ID: {reminder_id}."
                if message:
                    return f"Done, I'll remind you '{message}' at {fire_at}. ID: {reminder_id}."
                return f"Done, reminder scheduled for {fire_at}. ID: {reminder_id}."
        return "Listo." if locale.lower().startswith("es") else "Done."
    if status == "running":
        return "Sigo trabajando en eso." if locale.lower().startswith("es") else "I'm still working on that."
    return None
