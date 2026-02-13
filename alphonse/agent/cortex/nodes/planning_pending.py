from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PendingCycleDeps:
    now_iso_utc: Callable[[], str]
    is_effectively_empty_cycle_result: Callable[[dict[str, Any]], bool]
    is_planning_loop_state: Callable[[dict[str, Any]], bool]
    next_step_index: Callable[[list[dict[str, Any]], set[str]], int | None]
    bind_answer_to_steps: Callable[[list[dict[str, Any]], dict[str, Any], str, str], None]
    ability_registry_getter: Callable[[], Any]
    tool_registry: Any
    logger_info: Callable[[str, Any, Any], None]


def handle_pending_interaction_for_cycle(
    *,
    state: dict[str, Any],
    llm_client: Any,
    text: str,
    deps: PendingCycleDeps,
) -> dict[str, Any] | None:
    pending = state.get("pending_interaction")
    if not isinstance(pending, dict):
        return None

    ask_context = pending.get("context") if isinstance(pending.get("context"), dict) else {}
    if ask_context.get("tool") == "askQuestion":
        if bool(ask_context.get("replan_on_answer")):
            answer = text
            original_message = str(
                ask_context.get("original_message")
                or ask_context.get("source_message")
                or ""
            ).strip()
            clarifications = (
                ask_context.get("clarifications")
                if isinstance(ask_context.get("clarifications"), list)
                else []
            )
            clarification_entry = {
                "answer": answer,
                "slot": str(pending.get("key") or "answer"),
                "at": deps.now_iso_utc(),
            }
            if answer:
                clarifications = [*clarifications, clarification_entry]
            state["planning_context"] = {
                "original_message": original_message or text,
                "latest_user_answer": answer,
                "clarifications": clarifications,
                "facts": {
                    str(item.get("slot") or "answer"): item.get("answer")
                    for item in clarifications
                    if isinstance(item, dict)
                },
                "replan_on_answer": True,
            }
            state["last_user_message"] = original_message or text
            state["pending_interaction"] = None
            state["ability_state"] = {}
            deps.logger_info(
                "cortex planning interrupt answered chat_id=%s correlation_id=%s replan=true",
                state.get("chat_id"),
                state.get("correlation_id"),
            )
        else:
            ability_state = state.get("ability_state")
            resumed: dict[str, Any] | None = None
            if isinstance(ability_state, dict) and deps.is_planning_loop_state(ability_state):
                steps = ability_state.get("steps")
                if isinstance(steps, list):
                    context = (
                        pending.get("context")
                        if isinstance(pending.get("context"), dict)
                        else {}
                    )
                    pending_key = str(pending.get("key") or "answer")
                    answer = str(state.get("last_user_message") or "").strip()
                    if answer:
                        step_index_raw = context.get("step_index")
                        ask_idx = (
                            int(step_index_raw)
                            if isinstance(step_index_raw, int)
                            else deps.next_step_index(steps, {"waiting"})
                        )
                        if ask_idx is not None and 0 <= ask_idx < len(steps):
                            ask_step = steps[ask_idx]
                            ask_step["status"] = "executed"
                            ask_step["executed"] = True
                            ask_step["outcome"] = "answered"
                        bind = (
                            context.get("bind")
                            if isinstance(context.get("bind"), dict)
                            else {}
                        )
                        deps.bind_answer_to_steps(steps, bind, pending_key, answer)
                        state["pending_interaction"] = None
                        state["ability_state"] = ability_state
                        _ = llm_client
                        resumed = {
                            "ability_state": ability_state,
                            "pending_interaction": None,
                        }
            if resumed is not None:
                if not deps.is_effectively_empty_cycle_result(resumed):
                    return resumed
                state["pending_interaction"] = None
                state["ability_state"] = {}
                deps.logger_info(
                    "cortex pending answer consumed with noop chat_id=%s correlation_id=%s fallback=fresh_planning",
                    state.get("chat_id"),
                    state.get("correlation_id"),
                )

    intent_name = str(pending.get("context", {}).get("ability_intent") or "")
    if intent_name:
        ability = deps.ability_registry_getter().get(intent_name)
        if ability is not None:
            state["intent"] = intent_name
            return ability.execute(state, deps.tool_registry)
    return None


@dataclass(frozen=True)
class EmptyCycleResultDeps:
    is_planning_loop_state: Callable[[dict[str, Any]], bool]
    next_step_index: Callable[[list[dict[str, Any]], set[str]], int | None]


def is_effectively_empty_cycle_result(
    result: dict[str, Any],
    *,
    deps: EmptyCycleResultDeps,
) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("response_text") or result.get("response_key"):
        return False
    plans = result.get("plans")
    if isinstance(plans, list) and plans:
        return False
    pending = result.get("pending_interaction")
    if pending:
        return False
    ability_state = result.get("ability_state")
    if isinstance(ability_state, dict) and ability_state:
        if not deps.is_planning_loop_state(ability_state):
            return False
        steps = ability_state.get("steps")
        if not isinstance(steps, list):
            return True
        if deps.next_step_index(steps, {"ready", "incomplete", "waiting"}) is not None:
            return False
        return True
    return True
