from datetime import datetime, timedelta, timezone

from workers import notification_worker as worker


def test_process_due_events_marks_skipped_when_overdue(monkeypatch):
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    event_datetime = (now - timedelta(minutes=45)).isoformat()
    due_events = [{"id": "evt-1", "event_datetime": event_datetime}]

    monkeypatch.setattr(worker, "list_due_family_events", lambda *_, **__: due_events)

    updates = []

    def record_update(event_id, payload):
        updates.append((event_id, payload))

    monkeypatch.setattr(worker, "update_family_event", record_update)

    processed = worker.process_due_events(now, overdue_minutes=30)

    assert processed is True
    assert updates == [
        (
            "evt-1",
            {
                "execution_status": "skipped",
                "error_msg": "Skipped after 30 minutes overdue.",
            },
        )
    ]


def test_process_due_events_executes_notification(monkeypatch):
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    event_datetime = (now - timedelta(minutes=5)).isoformat()
    due_events = [
        {
            "id": "evt-2",
            "event_datetime": event_datetime,
            "title": "Dinner",
            "description": "Meal is ready",
            "target_group": "all",
        }
    ]

    monkeypatch.setattr(worker, "list_due_family_events", lambda *_, **__: due_events)
    monkeypatch.setattr(worker, "list_device_tokens", lambda *_: ["token-1"])
    monkeypatch.setattr(worker, "send_push_notification", lambda *_, **__: 1)

    updates = []

    def record_update(event_id, payload):
        updates.append((event_id, payload))

    monkeypatch.setattr(worker, "update_family_event", record_update)

    processed = worker.process_due_events(now, overdue_minutes=30)

    assert processed is True
    assert updates[0][1]["execution_status"] == "executing"
    assert updates[0][1]["sent_at"] == now.isoformat()
    assert updates[1][1] == {"execution_status": "executed"}


def test_process_due_events_marks_failed_when_no_tokens(monkeypatch):
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    event_datetime = (now - timedelta(minutes=2)).isoformat()
    due_events = [{"id": "evt-3", "event_datetime": event_datetime}]

    monkeypatch.setattr(worker, "list_due_family_events", lambda *_, **__: due_events)
    monkeypatch.setattr(worker, "list_device_tokens", lambda *_: [])

    updates = []

    def record_update(event_id, payload):
        updates.append((event_id, payload))

    monkeypatch.setattr(worker, "update_family_event", record_update)

    processed = worker.process_due_events(now, overdue_minutes=30)

    assert processed is True
    assert updates[0][1]["execution_status"] == "executing"
    assert updates[1][1]["execution_status"] == "failed"


def test_schedule_next_occurrence_creates_event(monkeypatch):
    event_datetime = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    next_occurrence = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)

    class FakeRule:
        def after(self, *_args, **_kwargs):
            return next_occurrence

    monkeypatch.setattr(worker, "rrulestr", lambda *_args, **_kwargs: FakeRule())

    class FakeDateTime:
        @classmethod
        def now(cls, tz=None):
            return event_datetime

    monkeypatch.setattr(worker, "datetime", FakeDateTime)

    created = []

    def record_create(payload):
        created.append(payload)

    monkeypatch.setattr(worker, "create_family_event", record_create)

    worker._schedule_next_occurrence(
        {
            "owner_id": "owner-1",
            "title": "Reminder",
            "description": "Test",
            "target_group": "all",
            "recurrence": "FREQ=DAILY",
        },
        event_datetime,
    )

    assert created
    payload = created[0]
    assert payload["event_datetime"] == next_occurrence.isoformat()
    assert payload["execution_status"] == "pending"
