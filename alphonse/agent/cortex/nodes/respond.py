from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from alphonse.agent.cortex.nodes.capability_gap import has_capability_gap_plan
from alphonse.agent.cortex.nodes.telemetry import emit_brain_state
from alphonse.agent.rendering.stage import render_stage
from alphonse.agent.rendering.types import TextDeliverable


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
                return _return(
                    run_capability_gap_tool(
                        state,
                        llm_client=llm_client,
                        reason="ability_execution_exception",
                    )
                )
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
    terminal_task_statuses = {"done", "failed"}
    is_terminal_task_state = task_status in terminal_task_statuses
    if has_capability_gap_plan(state):
        emit_transition_event(state, "failed")
        return _return({})
    if pending and not is_terminal_task_state:
        emit_transition_event(state, "waiting_user")
        return _return({})
    if isinstance(task_state, dict):
        utterance = build_utterance_from_state(state)
        if isinstance(utterance, dict):
            emit_brain_state(
                state=state,
                node="respond_node",
                updates={},
                stage="rendering_stage.start",
            )
            result = render_stage(utterance, llm_client=state.get("_llm_client"))
            if result.status == "rendered":
                emit_brain_state(
                    state=state,
                    node="respond_node",
                    updates={},
                    stage="rendering_stage.rendered",
                )
                rendered_text = next(
                    (
                        item.text
                        for item in result.deliverables
                        if isinstance(item, TextDeliverable) and str(item.text).strip()
                    ),
                    None,
                )
                if isinstance(rendered_text, str) and rendered_text.strip():
                    if task_status == "failed":
                        emit_transition_event(state, "failed")
                    elif task_status == "waiting_user":
                        emit_transition_event(state, "waiting_user")
                    else:
                        emit_transition_event(state, "done")
                    return _return({"response_text": rendered_text, "utterance": utterance})
            emit_brain_state(
                state=state,
                node="respond_node",
                updates={"render_error": result.error},
                stage="rendering_stage.failed",
                error_type=result.error or "render_failed",
            )
            emit_transition_event(state, "failed", {"reason": "render_failed"})
            return _return({"utterance": utterance, "render_error": result.error})
    if state.get("response_text"):
        emit_transition_event(state, "done")
    return _return({})


def build_utterance_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    task_state = state.get("task_state")
    if not isinstance(task_state, dict):
        return None
    status = str(task_state.get("status") or "").strip().lower()
    if not status:
        return None
    locale = str(state.get("locale") or "en-US")
    tone = str(state.get("tone") or "friendly")
    address_style = str(state.get("address_style") or ("tu" if locale.lower().startswith("es") else "you"))
    verbosity = str(state.get("verbosity") or "normal")
    utterance: dict[str, Any] = {
        "type": "task_update",
        "audience": {
            "channel_type": str(state.get("channel_type") or ""),
            "channel_target": str(state.get("channel_target") or ""),
            "person_id": state.get("actor_person_id"),
        },
        "prefs": {
            "locale": locale,
            "tone": tone,
            "address_style": address_style,
            "verbosity": verbosity,
        },
        "content": {},
        "meta": {
            "correlation_id": str(state.get("correlation_id") or ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    if status == "waiting_user":
        utterance["type"] = "question"
        utterance["content"] = {
            "question": str(task_state.get("next_user_question") or "").strip(),
        }
        return utterance
    if status == "failed":
        reason = task_state.get("last_validation_error")
        utterance["type"] = "task_failed"
        utterance["content"] = {
            "reason": reason if isinstance(reason, dict) else "task_failed",
        }
        return utterance
    if status == "done":
        outcome = task_state.get("outcome")
        if isinstance(outcome, dict) and str(outcome.get("kind") or "").strip() == "reminder_created":
            evidence = outcome.get("evidence")
            evidence_payload = evidence if isinstance(evidence, dict) else {}
            content: dict[str, Any] = {
                "subject": str(
                    evidence_payload.get("message") or evidence_payload.get("subject") or ""
                ).strip(),
                "fire_at": str(evidence_payload.get("fire_at") or "").strip(),
                "when_human": str(
                    evidence_payload.get("when_human")
                    or evidence_payload.get("original_time_expression")
                    or evidence_payload.get("fire_at")
                    or ""
                ).strip(),
                "for_whom": "me",
            }
            if verbosity == "debug":
                content["reminder_id"] = str(evidence_payload.get("reminder_id") or "").strip()
            utterance["type"] = "reminder_created"
            utterance["content"] = content
            return utterance
        summary = ""
        if isinstance(outcome, dict):
            summary = str(
                outcome.get("final_text")
                or outcome.get("summary")
                or ""
            ).strip()
        if not summary:
            summary = str(state.get("response_text") or "").strip()
        utterance["type"] = "task_done"
        utterance["content"] = {"summary": summary}
        return utterance
    if status == "running":
        utterance["type"] = "task_running"
        utterance["content"] = {}
        return utterance
    return None
