from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.cognition.capability_gaps.reporting import build_daily_report
from alphonse.agent.cognition.plan_executor import PlanExecutionContext, PlanExecutor
from alphonse.agent.cognition.plans import CortexPlan
from alphonse.agent.cognition.preferences.store import (
    list_principals_with_preference,
    resolve_preference_with_precedence,
)
from alphonse.agent.nervous_system.capability_gaps import list_recent_gaps
from alphonse.agent.policy.engine import PolicyEngine
from alphonse.config import settings

logger = logging.getLogger(__name__)


def dispatch_daily_report(context: dict, payload: dict[str, Any]) -> None:
    principals = list_principals_with_preference("reports.daily.enabled", True)
    if not principals:
        logger.info("DailyReport no enabled principals")
        return
    gaps = list_recent_gaps(hours=24)
    policy = PolicyEngine()
    executor = PlanExecutor()
    for principal in principals:
        channel_type = principal.get("channel_type")
        channel_id = principal.get("channel_id")
        principal_id = principal.get("principal_id")
        if not channel_type or not channel_id:
            continue
        locale = settings.get_default_locale()
        address_style = settings.get_address_style()
        tone = settings.get_tone()
        if principal_id:
            locale = resolve_preference_with_precedence(
                key="locale",
                default=locale,
                channel_principal_id=principal_id,
            )
            address_style = resolve_preference_with_precedence(
                key="address_style",
                default=address_style,
                channel_principal_id=principal_id,
            )
            tone = resolve_preference_with_precedence(
                key="tone",
                default=tone,
                channel_principal_id=principal_id,
            )
        report = build_daily_report(locale, gaps)
        communicate_plan = CortexPlan(
            tool="communicate",
            target=str(channel_id),
            channels=[str(channel_type)],
            parameters={
                "message": report,
                "locale": locale,
                "style": tone,
            },
            payload={
                "message": report,
                "locale": locale,
                "style": tone,
            },
        )
        approved = policy.approve([communicate_plan], context)
        exec_context = PlanExecutionContext(
            channel_type=str(channel_type),
            channel_target=str(channel_id),
            actor_person_id=None,
            correlation_id=str(payload.get("correlation_id") or "daily_report"),
        )
        if approved:
            logger.info(
                "DailyReport dispatch principal_id=%s locale=%s address=%s count=%s",
                principal_id,
                locale,
                address_style,
                len(gaps),
            )
            executor.execute(approved, context, exec_context)


def build_report_metadata() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
