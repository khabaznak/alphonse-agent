from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from alphonse.agent.cognition.plans import CortexPlan


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None


class PolicyRule(Protocol):
    def evaluate(self, plan: CortexPlan, exec_context: object) -> PolicyDecision | None:
        ...


class PolicyRuleProvider(Protocol):
    def build_rules(self) -> list[PolicyRule]:
        ...


class PolicyEngine:
    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
        rule_providers: list[PolicyRuleProvider] | None = None,
    ) -> None:
        providers = rule_providers[:] if rule_providers else _default_rule_providers()
        resolved_rules: list[PolicyRule] = []
        for provider in providers:
            resolved_rules.extend(provider.build_rules())
        if rules:
            resolved_rules.extend(rules)
        self._rules = resolved_rules

    def register_rule(self, rule: PolicyRule) -> None:
        self._rules.append(rule)

    def approve(self, plans: list[CortexPlan], context: object) -> list[CortexPlan]:
        _ = context
        approved: list[CortexPlan] = []
        for plan in plans:
            decision = self.approve_plan(plan, context)
            if decision.allowed:
                approved.append(plan)
        return approved

    def approve_plan(self, plan: CortexPlan, exec_context: object) -> PolicyDecision:
        for rule in self._rules:
            decision = rule.evaluate(plan, exec_context)
            if decision is not None:
                return decision
        return PolicyDecision(allowed=True)


def _default_rule_providers() -> list[PolicyRuleProvider]:
    from alphonse.agent.policy.rules.telegram import TelegramPolicyRuleProvider

    return [TelegramPolicyRuleProvider()]
