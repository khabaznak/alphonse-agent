from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dateutil.rrule import rrulestr
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.integrations.fcm import send_push_notification
from core.repositories.family_events import (
    create_family_event,
    list_due_family_events,
    list_next_family_events,
    update_family_event,
)
from core.repositories.push_devices import list_device_tokens
from rex.config import load_rex_config


def main() -> None:
    load_dotenv()
    config = load_rex_config()
    overdue_minutes = int(config.get("notifications", {}).get("overdue_minutes", 30))
    while True:
        now = datetime.now(timezone.utc)
        processed_any = process_due_events(now, overdue_minutes)
        if processed_any:
            continue

        next_event = fetch_next_event(now)
        if not next_event:
            time.sleep(30)
            continue

        next_time = _parse_event_datetime(next_event.get("event_datetime"))
        if not next_time:
            time.sleep(30)
            continue

        sleep_seconds = max(5, (next_time - now).total_seconds())
        time.sleep(sleep_seconds)


def process_due_events(now: datetime, overdue_minutes: int) -> bool:
    due_events = list_due_family_events(now.isoformat(), limit=200)
    if not due_events:
        return False

    cutoff = now - timedelta(minutes=overdue_minutes)
    for event in due_events:
        event_id = event.get("id")
        event_datetime = _parse_event_datetime(event.get("event_datetime"))
        if not event_id or not event_datetime:
            continue

        if event_datetime < cutoff:
            update_family_event(
                event_id,
                {
                    "execution_status": "skipped",
                    "error_msg": f"Skipped after {overdue_minutes} minutes overdue.",
                },
            )
            continue

        update_family_event(
            event_id,
            {
                "execution_status": "executing",
                "sent_at": now.isoformat(),
            },
        )

        title = event.get("title", "Notification")
        description = event.get("description") or ""
        target_group = event.get("target_group") or "all"
        tokens = list_device_tokens(target_group)

        if not tokens:
            update_family_event(
                event_id,
                {
                    "execution_status": "failed",
                    "error_msg": "No device tokens available.",
                },
            )
            continue

        try:
            send_push_notification(tokens, title, description, data={"event_id": event_id})
            update_family_event(event_id, {"execution_status": "executed"})
        except Exception as exc:  # noqa: BLE001
            update_family_event(
                event_id,
                {
                    "execution_status": "failed",
                    "error_msg": str(exc),
                },
            )
            continue

        if event.get("recurrence"):
            _schedule_next_occurrence(event, event_datetime)

    return True


def fetch_next_event(now: datetime) -> dict | None:
    next_events = list_next_family_events(now.isoformat(), limit=1)
    return next_events[0] if next_events else None


def _parse_event_datetime(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _schedule_next_occurrence(event: dict, event_datetime: datetime) -> None:
    recurrence = event.get("recurrence")
    if not recurrence:
        return

    rule = rrulestr(recurrence, dtstart=event_datetime)
    next_occurrence = rule.after(datetime.now(timezone.utc), inc=False)
    if not next_occurrence:
        return

    payload = {
        "owner_id": event.get("owner_id"),
        "title": event.get("title"),
        "description": event.get("description"),
        "event_datetime": next_occurrence.isoformat(),
        "recurrence": recurrence,
        "target_group": event.get("target_group"),
        "push_payload": event.get("push_payload"),
        "execution_status": "pending",
        "sent_at": None,
        "error_msg": None,
    }
    create_family_event(payload)


if __name__ == "__main__":
    main()
