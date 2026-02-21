from __future__ import annotations

import logging
import os

from alphonse.agent.cognition.plans import CortexPlan
from alphonse.agent.policy.engine import PolicyDecision, PolicyRule

logger = logging.getLogger(__name__)


class TelegramScheduleRule:
    def evaluate(self, plan: CortexPlan, exec_context: object) -> PolicyDecision | None:
        if str(plan.tool or "").strip().lower() != "schedule_timed_signal":
            return None
        channel_type = getattr(exec_context, "channel_type", None)
        channel_target = getattr(exec_context, "channel_target", None)
        if channel_type != "telegram":
            return PolicyDecision(allowed=True)
        allowed = _parse_allowed_chat_ids(
            os.getenv("TELEGRAM_ALLOWED_CHAT_IDS"),
            os.getenv("TELEGRAM_ALLOWED_CHAT_ID"),
        )
        if not allowed:
            return PolicyDecision(allowed=True)
        if channel_target is None:
            logger.warning("policy schedule denied reason=missing_target")
            return PolicyDecision(allowed=False, reason="missing_target")
        try:
            chat_id = int(str(channel_target))
        except ValueError:
            return PolicyDecision(allowed=False, reason="invalid_target")
        if chat_id not in allowed:
            return PolicyDecision(allowed=False, reason="not_allowed")
        return PolicyDecision(allowed=True)


class TelegramLanRule:
    def evaluate(self, plan: CortexPlan, exec_context: object) -> PolicyDecision | None:
        if str(plan.tool or "").strip().lower() not in {"lan_arm", "lan_disarm"}:
            return None
        channel_type = getattr(exec_context, "channel_type", None)
        channel_target = getattr(exec_context, "channel_target", None)
        if channel_type != "telegram":
            return PolicyDecision(allowed=False, reason="not_telegram")
        allowed = _parse_allowed_chat_ids(
            os.getenv("TELEGRAM_ALLOWED_CHAT_IDS"),
            os.getenv("TELEGRAM_ALLOWED_CHAT_ID"),
        )
        if not allowed:
            return PolicyDecision(allowed=False, reason="not_allowed")
        if channel_target is None:
            return PolicyDecision(allowed=False, reason="missing_target")
        try:
            chat_id = int(str(channel_target))
        except ValueError:
            return PolicyDecision(allowed=False, reason="invalid_target")
        if chat_id not in allowed:
            return PolicyDecision(allowed=False, reason="not_allowed")
        return PolicyDecision(allowed=True)


class PairingRule:
    def __init__(self, *, lan_rule: TelegramLanRule | None = None) -> None:
        self._lan_rule = lan_rule or TelegramLanRule()

    def evaluate(self, plan: CortexPlan, exec_context: object) -> PolicyDecision | None:
        if str(plan.tool or "").strip().lower() not in {"pair_approve", "pair_deny"}:
            return None
        channel_type = getattr(exec_context, "channel_type", None)
        if channel_type == "cli":
            return PolicyDecision(allowed=True)
        if channel_type != "telegram":
            return PolicyDecision(allowed=False, reason="not_telegram")
        return self._lan_rule.evaluate(
            CortexPlan(tool="lan_arm", parameters={}),
            exec_context,
        ) or PolicyDecision(allowed=True)


class TelegramPolicyRuleProvider:
    def build_rules(self) -> list[PolicyRule]:
        lan_rule = TelegramLanRule()
        return [
            TelegramScheduleRule(),
            lan_rule,
            PairingRule(lan_rule=lan_rule),
        ]


def _parse_allowed_chat_ids(primary: str | None, fallback: str | None) -> set[int] | None:
    raw = primary or fallback
    if not raw:
        return None
    ids: set[int] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            ids.add(int(entry))
        except ValueError:
            continue
    return ids or None
