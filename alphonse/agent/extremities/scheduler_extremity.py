from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from alphonse.agent.cognition.skills.command_plans import (
    Actor,
    ActorChannel,
    CreateReminderPayload,
    CreateReminderPlan,
    IntentEvidence,
    PersonRef,
    ReminderDelivery,
    ReminderMessage,
    ReminderSchedule,
    TargetRef,
)
from alphonse.agent.nervous_system.timed_commands import insert_timed_signal_from_plan
from alphonse.agent.core.settings_store import get_timezone


def schedule_reminder(
    *,
    reminder_text: str,
    trigger_time: str,
    chat_id: str,
    channel_type: str,
    actor_person_id: str | None,
    intent_evidence: dict[str, Any],
    correlation_id: str,
) -> str:
    plan = CreateReminderPlan(
        plan_kind="create_reminder",
        plan_version=1,
        plan_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        created_at=datetime.utcnow().isoformat(),
        source=channel_type,
        actor=Actor(
            person_id=actor_person_id,
            channel=ActorChannel(type=channel_type, target=chat_id),
        ),
        intent_confidence=0.8,
        requires_confirmation=False,
        questions=[],
        intent_evidence=IntentEvidence.model_validate(intent_evidence),
        payload=CreateReminderPayload(
            target=TargetRef(person_ref=PersonRef(kind="person_id", id=chat_id)),
            schedule=ReminderSchedule(
                timezone=_timezone_from(trigger_time),
                trigger_at=trigger_time,
                rrule=None,
                time_of_day=None,
            ),
            message=ReminderMessage(language="es", text=reminder_text),
            delivery=ReminderDelivery(channel_type=channel_type, priority="normal"),
            idempotency_key=f"{channel_type}:{chat_id}:{trigger_time}:{reminder_text}",
        ),
        metadata=None,
        original_text=reminder_text,
    )
    insert_timed_signal_from_plan(plan)
    return plan.plan_id


def _timezone_from(trigger_time: str) -> str:
    if trigger_time.endswith("Z"):
        return "UTC"
    return get_timezone()
