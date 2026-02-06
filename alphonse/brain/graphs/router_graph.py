from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from alphonse.brain.habits_db import Habit, list_enabled_habits_for_trigger


class RouterState(TypedDict, total=False):
    trigger: str
    context: dict[str, Any]
    event_payload: dict[str, Any]
    habit: Habit | None


def _match_habit(state: RouterState) -> dict[str, Any]:
    trigger = state.get("trigger") or ""
    context = state.get("context") or {}
    habits = list_enabled_habits_for_trigger(trigger)
    best: Habit | None = None
    best_score = -1
    for habit in habits:
        conditions = habit.conditions_json or {}
        if not _conditions_match(conditions, context):
            continue
        score = len(conditions)
        if score > best_score or (score == best_score and habit.version >= (best.version if best else 0)):
            best = habit
            best_score = score
    return {"habit": best}


def _conditions_match(conditions: dict[str, Any], context: dict[str, Any]) -> bool:
    for key, value in conditions.items():
        if key not in context:
            return False
        if context[key] != value:
            return False
    return True


def build_router_graph() -> StateGraph:
    graph = StateGraph(RouterState)
    graph.add_node("match_habit", _match_habit)
    graph.set_entry_point("match_habit")
    graph.add_edge("match_habit", END)
    return graph


def run_router(trigger: str, context: dict[str, Any], payload: dict[str, Any]) -> Habit | None:
    graph = build_router_graph().compile()
    result = graph.invoke({"trigger": trigger, "context": context, "event_payload": payload})
    return result.get("habit")
