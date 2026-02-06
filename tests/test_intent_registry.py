from __future__ import annotations

from alphonse.agent.cognition.capability_gaps.guard import GapGuardInput, PlanStatus, should_create_gap
from alphonse.agent.cognition.intent_registry import (
    IntentCategory,
    IntentMetadata,
    IntentRegistry,
    RiskLevel,
    get_registry,
    register_builtin_intents,
)
from alphonse.agent.cognition.intent_router import route_message


def test_core_conversational_never_creates_gap() -> None:
    data = GapGuardInput(
        category=IntentCategory.CORE_CONVERSATIONAL,
        plan_status=None,
        needs_clarification=False,
        reason="missing_capability",
    )
    assert should_create_gap(data) is False


def test_control_plane_never_creates_gap() -> None:
    data = GapGuardInput(
        category=IntentCategory.CONTROL_PLANE,
        plan_status=None,
        needs_clarification=False,
        reason="missing_capability",
    )
    assert should_create_gap(data) is False


def test_task_planning_state_never_creates_gap() -> None:
    data = GapGuardInput(
        category=IntentCategory.TASK_PLANE,
        plan_status=PlanStatus.PLANNING,
        needs_clarification=False,
        reason="missing_capability",
    )
    assert should_create_gap(data) is False


def test_gap_requires_missing_capability_reason() -> None:
    allowed = GapGuardInput(
        category=IntentCategory.TASK_PLANE,
        plan_status=None,
        needs_clarification=False,
        reason="missing_capability",
    )
    denied = GapGuardInput(
        category=IntentCategory.TASK_PLANE,
        plan_status=None,
        needs_clarification=False,
        reason="no_tool",
    )
    assert should_create_gap(allowed) is True
    assert should_create_gap(denied) is False


def test_dynamic_intent_registration_is_routable() -> None:
    registry = IntentRegistry()
    register_builtin_intents(registry)
    registry.register(
        "memory.search",
        IntentMetadata(
            category=IntentCategory.TASK_PLANE,
            requires_planner=True,
            default_risk=RiskLevel.LOW,
            supports_autonomy=True,
            patterns=(r"\brecall memory\b",),
        ),
    )
    routed = route_message("please recall memory about travel", registry=registry)
    assert routed.intent == "memory.search"
    assert routed.category == IntentCategory.TASK_PLANE


def test_identity_question_routes_in_spanish() -> None:
    routed = route_message("Quién eres?", registry=get_registry())
    assert routed.intent == "identity_question"
    assert routed.category == IntentCategory.CORE_CONVERSATIONAL


def test_identity_question_routes_te_llamas() -> None:
    routed = route_message("Como te llamas?", registry=get_registry())
    assert routed.intent == "identity_question"
    assert routed.category == IntentCategory.CORE_CONVERSATIONAL
    assert routed.rationale == "fast_path_regex"


def test_user_identity_question_routes_in_spanish() -> None:
    routed = route_message("Cuál es mi nombre?", registry=get_registry())
    assert routed.intent == "user_identity_question"
    assert routed.category == IntentCategory.CORE_CONVERSATIONAL


def test_identity_question_routes_in_english() -> None:
    routed = route_message("What is your name?", registry=get_registry())
    assert routed.intent == "identity_question"
    assert routed.category == IntentCategory.CORE_CONVERSATIONAL


def test_user_identity_question_routes_in_english() -> None:
    routed = route_message("What's my name?", registry=get_registry())
    assert routed.intent == "user_identity_question"
    assert routed.category == IntentCategory.CORE_CONVERSATIONAL


def test_capabilities_routes_in_spanish() -> None:
    routed = route_message("Qué sabes hacer?", registry=get_registry())
    assert routed.intent == "meta.capabilities"
    assert routed.category == IntentCategory.DEBUG_META
