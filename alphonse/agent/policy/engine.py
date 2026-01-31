from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from alphonse.agent.cognition.plans import CortexPlan, PlanType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None


class PolicyEngine:
    def approve_plan(self, plan: CortexPlan, exec_context: object) -> PolicyDecision:
        if plan.plan_type == PlanType.SCHEDULE_TIMED_SIGNAL:
            return self._approve_schedule(plan, exec_context)
        return PolicyDecision(allowed=True)

    def _approve_schedule(self, plan: CortexPlan, exec_context: object) -> PolicyDecision:
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
