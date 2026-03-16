from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

from alphonse.agent.actions.session_context import IncomingContext
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_task
from alphonse.agent.nervous_system.pdca_queue_store import upsert_pdca_task
from alphonse.agent.services.pdca_ingress import enqueue_pdca_slice


def _incoming(*, message_id: str) -> IncomingContext:
    return IncomingContext(
        channel_type="telegram",
        address="8553589429",
        person_id=None,
        correlation_id=f"cid-{message_id}",
        message_id=message_id,
    )


def test_enqueue_pdca_slice_buffers_inputs_for_running_task(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "task_id": "task-buffer-1",
            "owner_id": "u-1",
            "conversation_key": "telegram:8553589429",
            "status": "running",
            "metadata": {
                "inputs": [
                    {
                        "message_id": "m-1",
                        "correlation_id": "cid-m-1",
                        "text": "first",
                        "channel": "telegram",
                        "received_at": "2026-03-12T05:00:00+00:00",
                        "consumed_at": None,
                        "sequence": 1,
                    }
                ],
                "next_unconsumed_index": 0,
                "input_dirty": False,
            },
        }
    )

    returned = enqueue_pdca_slice(
        context={},
        incoming=_incoming(message_id="m-2"),
        state={"timezone": "UTC"},
        session_key="telegram:8553589429",
        session_user_id="u-1",
        day_session={"session_id": "u-1|2026-03-12", "user_id": "u-1", "date": "2026-03-12"},
        payload={"text": "second"},
        correlation_id="cid-m-2",
    )
    assert returned == task_id

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 2
    assert str(inputs[0].get("text") or "") == "first"
    assert str(inputs[1].get("text") or "") == "second"
    assert str(inputs[1].get("actor_id") or "") == "u-1"
    assert bool(metadata.get("input_dirty")) is True
    assert int(metadata.get("next_unconsumed_index") or 0) == 0


def test_enqueue_pdca_slice_dedupes_same_message_id_and_correlation(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "task_id": "task-buffer-2",
            "owner_id": "u-1",
            "conversation_key": "telegram:8553589429",
            "status": "queued",
        }
    )
    kwargs = dict(
        context={},
        incoming=_incoming(message_id="m-dup"),
        state={"timezone": "UTC"},
        session_key="telegram:8553589429",
        session_user_id="u-1",
        day_session={"session_id": "u-1|2026-03-12", "user_id": "u-1", "date": "2026-03-12"},
        payload={"text": "same"},
        correlation_id="cid-dup",
    )

    _ = enqueue_pdca_slice(**kwargs)
    _ = enqueue_pdca_slice(**kwargs)

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 1


def test_enqueue_pdca_slice_owner_only_rejects_non_owner_actor(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "task_id": "task-owner-only",
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "running",
            "metadata": {
                "steering_scope": "owner_only",
                "inputs": [],
                "next_unconsumed_index": 0,
                "input_dirty": False,
            },
        }
    )

    returned = enqueue_pdca_slice(
        context={},
        incoming=_incoming(message_id="m-owner-only"),
        state={"timezone": "UTC"},
        session_key="telegram:8553589429",
        session_user_id="not-owner",
        day_session={"session_id": "owner-1|2026-03-12", "user_id": "owner-1", "date": "2026-03-12"},
        payload={"text": "ignored"},
        correlation_id="cid-owner-only",
    )
    assert returned == task_id

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 0
    log = metadata.get("steering_decision_log")
    assert isinstance(log, list) and log
    assert str(log[-1].get("decision") or "") == "rejected"
    assert str(log[-1].get("reason") or "") == "owner_only_scope"


def test_enqueue_pdca_slice_targeted_scope_honors_timeout_fallback(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

    task_id = upsert_pdca_task(
        {
            "task_id": "task-targeted",
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "running",
            "metadata": {
                "steering_scope": "targeted",
                "target_actor_id": "wife-1",
                "target_wait_timeout_at": future,
                "inputs": [],
                "next_unconsumed_index": 0,
                "input_dirty": False,
            },
        }
    )

    _ = enqueue_pdca_slice(
        context={},
        incoming=_incoming(message_id="m-targeted-reject"),
        state={"timezone": "UTC"},
        session_key="telegram:8553589429",
        session_user_id="not-wife",
        day_session={"session_id": "owner-1|2026-03-12", "user_id": "owner-1", "date": "2026-03-12"},
        payload={"text": "too early"},
        correlation_id="cid-targeted-reject",
    )

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 0
    log = metadata.get("steering_decision_log")
    assert isinstance(log, list) and log
    assert str(log[-1].get("reason") or "") == "target_waiting_for_actor"

    _ = upsert_pdca_task(
        {
            "task_id": task_id,
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "running",
            "metadata": {
                **metadata,
                "target_wait_timeout_at": past,
            },
        }
    )

    _ = enqueue_pdca_slice(
        context={},
        incoming=_incoming(message_id="m-targeted-fallback"),
        state={"timezone": "UTC"},
        session_key="telegram:8553589429",
        session_user_id="not-wife",
        day_session={"session_id": "owner-1|2026-03-12", "user_id": "owner-1", "date": "2026-03-12"},
        payload={"text": "after timeout"},
        correlation_id="cid-targeted-fallback",
    )

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 1
    assert str(inputs[0].get("text") or "") == "after timeout"
    log = metadata.get("steering_decision_log")
    assert isinstance(log, list) and log
    assert str(log[-1].get("reason") or "") == "target_timeout_fallback"


def test_enqueue_pdca_slice_accepts_attachment_only_input(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "task_id": "task-attachment-only",
            "owner_id": "u-1",
            "conversation_key": "telegram:8553589429",
            "status": "running",
            "metadata": {
                "inputs": [],
                "next_unconsumed_index": 0,
                "input_dirty": False,
            },
        }
    )

    returned = enqueue_pdca_slice(
        context={},
        incoming=_incoming(message_id="m-a1"),
        state={"timezone": "UTC"},
        session_key="telegram:8553589429",
        session_user_id="u-1",
        day_session={"session_id": "u-1|2026-03-12", "user_id": "u-1", "date": "2026-03-12"},
        payload={
            "text": "",
            "content": {"attachments": [{"kind": "voice", "provider": "telegram", "file_id": "voice-1"}]},
        },
        correlation_id="cid-a1",
    )
    assert returned == task_id

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    assert str(metadata.get("pending_user_text") or "").startswith("[attachments:")
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 1
    assert isinstance(inputs[0].get("attachments"), list)
    assert str(inputs[0].get("steering_text") or "").startswith("[attachments:")


def test_enqueue_pdca_slice_persists_correlation_in_metadata_state(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = enqueue_pdca_slice(
        context={},
        incoming=_incoming(message_id="m-corr-state"),
        state={"timezone": "UTC"},
        session_key="telegram:8553589429",
        session_user_id="u-1",
        day_session={"session_id": "u-1|2026-03-12", "user_id": "u-1", "date": "2026-03-12"},
        payload={"text": "create task with correlation"},
        correlation_id="cid-corr-state",
    )

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    state = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    assert str(state.get("correlation_id") or "") == "cid-corr-state"
