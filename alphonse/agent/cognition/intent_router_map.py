from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alphonse.agent.cognition.message_map_llm import MessageMap


Domain = Literal[
    "commands",
    "social",
    "task_single",
    "task_multi",
    "question",
    "other",
]

Decision = Literal["route", "interrupt_and_new_plan"]


@dataclass(frozen=True)
class FsmState:
    plan_active: bool
    expected_slot: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    domain: Domain
    decision: Decision
    interrupt_current_plan: bool = False
    reason: str = ""


def route(message_map: MessageMap, fsm_state: FsmState) -> RouteDecision:
    if message_map.commands:
        return RouteDecision(
            domain="commands",
            decision="route",
            reason="commands_present",
        )

    has_actions = len(message_map.actions) > 0
    has_questions = len(message_map.questions) > 0
    social_only = (
        message_map.social.is_greeting and not has_actions and not has_questions
    )

    if fsm_state.plan_active:
        if message_map.social.is_greeting and message_map.raw_intent_hint == "social_only":
            return RouteDecision(
                domain="social",
                decision="route",
                reason="plan_active_social_only",
            )
        looks_new_task = has_actions or has_questions
        if looks_new_task and not is_expected_slot_value(message_map, fsm_state.expected_slot):
            domain: Domain = "question" if has_questions and not has_actions else "task_single"
            if len(message_map.actions) >= 2:
                domain = "task_multi"
            return RouteDecision(
                domain=domain,
                decision="interrupt_and_new_plan",
                interrupt_current_plan=True,
                reason="barge_in_new_task",
            )

    if social_only:
        return RouteDecision(domain="social", decision="route", reason="social_only")
    if len(message_map.actions) == 1:
        return RouteDecision(domain="task_single", decision="route", reason="single_action")
    if len(message_map.actions) >= 2:
        return RouteDecision(domain="task_multi", decision="route", reason="multi_action")
    if has_questions:
        return RouteDecision(domain="question", decision="route", reason="question_only")
    return RouteDecision(domain="other", decision="route", reason="other")


def is_expected_slot_value(message_map: MessageMap, expected_slot: str | None) -> bool:
    if not expected_slot:
        return False
    if expected_slot == "time_expression":
        return bool(message_map.constraints.times)
    if expected_slot == "number":
        return bool(message_map.constraints.numbers)
    if expected_slot == "geo":
        return bool(message_map.constraints.locations)
    if expected_slot == "string":
        has_entities_or_text = bool(message_map.entities) or (
            not message_map.actions
            and not message_map.questions
            and bool(message_map.social.text)
        )
        return has_entities_or_text
    return False
