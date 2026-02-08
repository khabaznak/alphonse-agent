from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from typing import Any

from alphonse.agent.nervous_system.timed_store import insert_timed_signal


@dataclass(frozen=True)
class SchedulerTool:
    def schedule_reminder(
        self,
        *,
        reminder_text: str,
        trigger_time: str,
        chat_id: str,
        channel_type: str,
        actor_person_id: str | None,
        intent_evidence: dict[str, Any],
        correlation_id: str,
        timezone_name: str,
        locale_hint: str | None,
    ) -> str:
        payload = {
            "message": reminder_text,
            "reminder_text_raw": reminder_text,
            "person_id": actor_person_id or chat_id,
            "chat_id": chat_id,
            "origin_channel": channel_type,
            "locale_hint": locale_hint,
            "created_at": datetime.now(dt_timezone.utc).isoformat(),
            "trigger_at": trigger_time,
            "intent_evidence": intent_evidence,
        }
        return insert_timed_signal(
            trigger_at=trigger_time,
            timezone=timezone_name,
            signal_type="reminder",
            payload=payload,
            target=str(actor_person_id or chat_id),
            origin=channel_type,
            correlation_id=correlation_id,
        )
