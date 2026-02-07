from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from alphonse.agent.cognition.plans import CortexPlan, PlanType
from alphonse.agent.policy.engine import PolicyDecision, PolicyEngine, PolicyRule


def _context(channel_type: str, channel_target: str | int | None) -> SimpleNamespace:
    return SimpleNamespace(channel_type=channel_type, channel_target=channel_target)


def test_schedule_rule_denies_unapproved_telegram_chat(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "111,222")
    engine = PolicyEngine()

    decision = engine.approve_plan(
        CortexPlan(plan_type=PlanType.SCHEDULE_TIMED_SIGNAL, payload={}),
        _context("telegram", "333"),
    )

    assert decision.allowed is False
    assert decision.reason == "not_allowed"


def test_pairing_rule_allows_cli_channel() -> None:
    engine = PolicyEngine()

    decision = engine.approve_plan(
        CortexPlan(plan_type=PlanType.PAIR_APPROVE, payload={}),
        _context("cli", "local"),
    )

    assert decision.allowed is True


@dataclass
class DenyAllRule:
    def evaluate(self, plan: CortexPlan, exec_context: object) -> PolicyDecision | None:
        _ = plan
        _ = exec_context
        return PolicyDecision(allowed=False, reason="deny_all")


@dataclass
class CustomProvider:
    def build_rules(self) -> list[PolicyRule]:
        return [DenyAllRule()]


def test_custom_rule_provider_overrides_default_rules() -> None:
    engine = PolicyEngine(rule_providers=[CustomProvider()])

    decision = engine.approve_plan(
        CortexPlan(plan_type=PlanType.COMMUNICATE, payload={}),
        _context("telegram", "111"),
    )

    assert decision.allowed is False
    assert decision.reason == "deny_all"
