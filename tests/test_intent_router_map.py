from __future__ import annotations

from alphonse.agent.cognition.intent_router_map import (
    FsmState,
    is_expected_slot_value,
    route,
)
from alphonse.agent.cognition.message_map_llm import (
    ActionFragment,
    Constraints,
    MessageMap,
    SocialFragment,
)


def _base_map() -> MessageMap:
    return MessageMap(
        language="en",
        social=SocialFragment(),
        actions=[],
        entities=[],
        constraints=Constraints(),
        questions=[],
        commands=[],
        raw_intent_hint="other",
        confidence="medium",
    )


def test_greeting_only_routes_social() -> None:
    msg = _base_map()
    msg = MessageMap(
        language=msg.language,
        social=SocialFragment(is_greeting=True, text="Hi"),
        actions=[],
        entities=[],
        constraints=msg.constraints,
        questions=[],
        commands=[],
        raw_intent_hint="social_only",
        confidence="high",
    )
    decision = route(msg, FsmState(plan_active=False))
    assert decision.domain == "social"
    assert decision.decision == "route"


def test_plan_active_greeting_does_not_interrupt() -> None:
    msg = MessageMap(
        language="en",
        social=SocialFragment(is_greeting=True, text="Good evening"),
        actions=[],
        entities=[],
        constraints=Constraints(),
        questions=[],
        commands=[],
        raw_intent_hint="social_only",
        confidence="high",
    )
    decision = route(msg, FsmState(plan_active=True, expected_slot="time_expression"))
    assert decision.domain == "social"
    assert decision.decision == "route"
    assert decision.interrupt_current_plan is False


def test_plan_active_new_imperative_interrupts() -> None:
    msg = MessageMap(
        language="en",
        social=SocialFragment(),
        actions=[ActionFragment(verb="remind", object="drink water")],
        entities=[],
        constraints=Constraints(),
        questions=[],
        commands=[],
        raw_intent_hint="single_action",
        confidence="high",
    )
    decision = route(msg, FsmState(plan_active=True, expected_slot="time_expression"))
    assert decision.decision == "interrupt_and_new_plan"
    assert decision.interrupt_current_plan is True


def test_heather_routes_stable_other() -> None:
    msg = MessageMap(
        language="en",
        social=SocialFragment(),
        actions=[],
        entities=["Heather"],
        constraints=Constraints(),
        questions=[],
        commands=[],
        raw_intent_hint="other",
        confidence="low",
    )
    decision = route(msg, FsmState(plan_active=False))
    assert decision.domain == "other"


def test_expected_slot_value_helper() -> None:
    msg = MessageMap(
        language="en",
        social=SocialFragment(),
        actions=[],
        entities=[],
        constraints=Constraints(times=["in 10 min"]),
        questions=[],
        commands=[],
        raw_intent_hint="other",
        confidence="low",
    )
    assert is_expected_slot_value(msg, "time_expression") is True
