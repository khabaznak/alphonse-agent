from __future__ import annotations

from typing import Any

from alphonse.agent.services.pdca_runtime import get_pdca_runtime


def upsert_pdca_task(record: dict[str, Any]) -> str:
    return get_pdca_runtime().upsert_task(record)


def get_pdca_task(task_id: str) -> dict[str, Any] | None:
    return get_pdca_runtime().get_task(task_id)


def get_latest_pdca_task_for_conversation(
    *,
    conversation_key: str,
    statuses: list[str] | None = None,
) -> dict[str, Any] | None:
    return get_pdca_runtime().get_latest_task_for_conversation(
        conversation_key=conversation_key,
        statuses=statuses,
    )


def get_latest_pdca_task_for_owner(
    *,
    owner_id: str,
    statuses: list[str] | None = None,
) -> dict[str, Any] | None:
    return get_pdca_runtime().get_latest_task_for_owner(
        owner_id=owner_id,
        statuses=statuses,
    )


def list_runnable_pdca_tasks(*, now: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    return get_pdca_runtime().list_runnable(now=now, limit=limit)


def acquire_pdca_task_lease(*, task_id: str, worker_id: str, lease_seconds: int = 30, now: str | None = None) -> bool:
    return get_pdca_runtime().acquire_lease(
        task_id=task_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        now=now,
    )


def release_pdca_task_lease(*, task_id: str, worker_id: str) -> bool:
    return get_pdca_runtime().release_lease(task_id=task_id, worker_id=worker_id)


def update_pdca_task_status(*, task_id: str, status: str, last_error: str | None = None) -> bool:
    return get_pdca_runtime().update_task_status(task_id=task_id, status=status, last_error=last_error)


def update_pdca_task_metadata(*, task_id: str, metadata: dict[str, Any]) -> bool:
    return get_pdca_runtime().update_task_metadata(task_id=task_id, metadata=metadata)


def save_pdca_checkpoint(
    *,
    task_id: str,
    state: dict[str, Any],
    expected_version: int | None = None,
) -> int | None:
    return get_pdca_runtime().save_checkpoint(task_id=task_id, state=state, expected_version=expected_version)


def load_pdca_checkpoint(task_id: str) -> dict[str, Any] | None:
    return get_pdca_runtime().load_checkpoint(task_id)


def append_pdca_event(*, task_id: str, event_type: str, payload: dict[str, Any] | None = None, correlation_id: str | None = None) -> str:
    return get_pdca_runtime().append_event(
        task_id=task_id,
        event_type=event_type,
        payload=payload,
        correlation_id=correlation_id,
    )


def list_pdca_events(*, task_id: str, limit: int = 100) -> list[dict[str, Any]]:
    return get_pdca_runtime().list_events(task_id=task_id, limit=limit)


def has_pdca_event(
    *,
    task_id: str,
    event_type: str,
    correlation_id: str | None = None,
) -> bool:
    return get_pdca_runtime().has_event(
        task_id=task_id,
        event_type=event_type,
        correlation_id=correlation_id,
    )


def describe_pdca_runtime_flush_counts(*, include_signal_queue: bool = True) -> dict[str, int]:
    return get_pdca_runtime().describe_counts(include_signal_queue=include_signal_queue)


def flush_signal_queue(*, conn: Any | None = None) -> int:
    _ = conn
    return 0


def flush_pdca_runtime_state(*, include_signal_queue: bool = True) -> dict[str, int]:
    _ = include_signal_queue
    return get_pdca_runtime().clear()


def get_pdca_queue_metrics(
    *,
    now: str | None = None,
    lookback_minutes: int = 15,
    top_owners: int = 5,
) -> dict[str, Any]:
    return get_pdca_runtime().get_metrics(
        now=now,
        lookback_minutes=lookback_minutes,
        top_owners=top_owners,
    )
