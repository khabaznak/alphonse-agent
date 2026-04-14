from __future__ import annotations

from pathlib import Path
from alphonse.agent.actions.conscious_message_handler import IncomingMessageEnvelope
from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.actions.pdca_task_boundary import build_task_record_for_message
from alphonse.agent.actions.pdca_task_boundary import select_pending_pdca_task_for_message
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_task
from alphonse.agent.nervous_system.pdca_queue_store import upsert_pdca_task
from alphonse.agent.services.pdca_ingress import enqueue_pdca_slice


def _envelope(
    *,
    message_id: str,
    text: str = "",
    attachments: list[dict[str, object]] | None = None,
    controls: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> IncomingMessageEnvelope:
    return IncomingMessageEnvelope.from_payload(build_incoming_message_envelope(
        message_id=message_id,
        channel_type="telegram",
        channel_target="8553589429",
        provider="telegram",
        text=text,
        correlation_id=f"cid-{message_id}",
        actor_external_user_id="ext-u-1",
        attachments=attachments,
        controls=controls,
        metadata=metadata,
    ))


def _day_session(user_id: str = "u-1") -> dict[str, object]:
    return {"session_id": f"{user_id}|2026-03-12", "user_id": user_id, "date": "2026-03-12"}


def _context(envelope: IncomingMessageEnvelope, correlation_id: str) -> dict[str, object]:
    return {
        "signal": Signal(
            type="sense.user_communication.message.user.received",
            payload=envelope.to_dict(),
            correlation_id=correlation_id,
        ),
    }


def _task_record(
    *,
    envelope: IncomingMessageEnvelope,
    session_user_id: str = "u-1",
    day_session: dict[str, object] | None = None,
    correlation_id: str,
    existing_task: dict[str, object] | None = None,
):
    return build_task_record_for_message(
        envelope=envelope,
        session_user_id=session_user_id,
        day_session=dict(day_session or _day_session(session_user_id)),
        correlation_id=correlation_id,
        existing_task=existing_task,
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

    envelope = _envelope(message_id="m-2", text="second")
    day_session = _day_session("u-1")
    existing_task = get_pdca_task(task_id)
    assert isinstance(existing_task, dict)
    returned = enqueue_pdca_slice(
        context=_context(envelope, "cid-m-2"),
        task_record=_task_record(
            envelope=envelope,
            day_session=day_session,
            correlation_id="cid-m-2",
            existing_task=existing_task,
        ),
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
    state = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    assert state.get("check_provenance") == "slice_resume"


def test_enqueue_pdca_slice_dedupes_same_message_id_and_correlation(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
            {
                "task_id": "task-buffer-2",
                "owner_id": "u-1",
                "conversation_key": "telegram:8553589429",
                "status": "running",
            }
        )
    envelope = _envelope(message_id="m-dup", text="same")
    day_session = _day_session("u-1")
    existing_task = get_pdca_task(task_id)
    assert isinstance(existing_task, dict)
    kwargs = dict(
        context=_context(envelope, "cid-dup"),
        task_record=_task_record(
            envelope=envelope,
            day_session=day_session,
            correlation_id="cid-dup",
            existing_task=existing_task,
        ),
    )

    _ = enqueue_pdca_slice(**kwargs)
    _ = enqueue_pdca_slice(**kwargs)

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 1


def test_enqueue_pdca_slice_rejects_mismatched_existing_task_and_creates_new_task(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "task_id": "task-policy-ignored",
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "running",
            "metadata": {
                "steering_scope": "owner_only",
                "allowed_actor_ids": ["owner-1"],
                "target_actor_id": "owner-1",
                "inputs": [],
                "next_unconsumed_index": 0,
                "input_dirty": False,
            },
        }
    )

    envelope = _envelope(message_id="m-policy", text="must still enqueue")
    day_session = {"session_id": "owner-1|2026-03-12", "user_id": "owner-1", "date": "2026-03-12"}
    existing_task = get_pdca_task(task_id)
    assert isinstance(existing_task, dict)
    returned = enqueue_pdca_slice(
        context=_context(envelope, "cid-policy"),
        task_record=_task_record(
            envelope=envelope,
            session_user_id="not-owner",
            day_session=day_session,
            correlation_id="cid-policy",
            existing_task=existing_task,
        ),
    )
    assert returned != task_id

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 0

    created = get_pdca_task(returned)
    assert isinstance(created, dict)
    assert str(created.get("owner_id") or "") == "not-owner"
    created_metadata = created.get("metadata") if isinstance(created.get("metadata"), dict) else {}
    created_inputs = created_metadata.get("inputs")
    assert isinstance(created_inputs, list)
    assert len(created_inputs) == 1
    assert str(created_inputs[0].get("text") or "") == "must still enqueue"


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

    envelope = _envelope(
        message_id="m-a1",
        attachments=[{"kind": "voice", "provider": "telegram", "file_id": "voice-1"}],
    )
    day_session = _day_session("u-1")
    existing_task = get_pdca_task(task_id)
    assert isinstance(existing_task, dict)
    returned = enqueue_pdca_slice(
        context=_context(envelope, "cid-a1"),
        task_record=_task_record(
            envelope=envelope,
            day_session=day_session,
            correlation_id="cid-a1",
            existing_task=existing_task,
        ),
    )
    assert returned == task_id

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    assert str(metadata.get("pending_user_text") or "").startswith("[attachments:")
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 1
    assert str(inputs[0].get("text") or "").startswith("[attachments:")
    assert isinstance(inputs[0].get("attachments"), list)


def test_enqueue_pdca_slice_preserves_contact_payload_and_extras(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "task_id": "task-contact-preserve",
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

    envelope = _envelope(
        message_id="m-contact-1",
        attachments=[
            {
                "kind": "contact",
                "provider": "telegram",
                "contact": {"user_id": 8593816828, "first_name": "Gabriela", "last_name": "Villasana"},
                "contact_user_id": "8593816828",
                "labels": ["lead", "priority"],
                "meta": {"source": "telegram_card"},
            }
        ],
    )
    day_session = _day_session("u-1")
    existing_task = get_pdca_task(task_id)
    assert isinstance(existing_task, dict)
    returned = enqueue_pdca_slice(
        context=_context(envelope, "cid-contact-1"),
        task_record=_task_record(
            envelope=envelope,
            day_session=day_session,
            correlation_id="cid-contact-1",
            existing_task=existing_task,
        ),
    )
    assert returned == task_id

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 1
    attachments = inputs[0].get("attachments")
    assert isinstance(attachments, list)
    assert len(attachments) == 1
    first = attachments[0]
    assert isinstance(first, dict)
    assert str(first.get("kind") or "") == "contact"
    assert str(first.get("contact_user_id") or "") == "8593816828"
    contact = first.get("contact")
    assert isinstance(contact, dict)
    assert int(contact.get("user_id") or 0) == 8593816828
    assert first.get("labels") == ["lead", "priority"]
    assert first.get("meta") == {"source": "telegram_card"}


def test_enqueue_pdca_slice_persists_correlation_in_metadata_state(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    envelope = _envelope(message_id="m-corr-state", text="create task with correlation")
    day_session = _day_session("u-1")
    task_id = enqueue_pdca_slice(
        context=_context(envelope, "cid-corr-state"),
        task_record=_task_record(
            envelope=envelope,
            day_session=day_session,
            correlation_id="cid-corr-state",
        ),
    )

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    state = metadata.get("state") if isinstance(metadata.get("state"), dict) else {}
    assert str(state.get("correlation_id") or "") == "cid-corr-state"
    assert state.get("check_provenance") == "entry"
    assert str(metadata.get("initial_message_id") or "") == "m-corr-state"
    assert str(metadata.get("initial_correlation_id") or "") == "cid-corr-state"


def test_enqueue_pdca_slice_keeps_non_contact_media_behavior(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "task_id": "task-photo-preserve",
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

    envelope = _envelope(
        message_id="m-photo-1",
        attachments=[
            {
                "kind": "photo",
                "provider": "telegram",
                "file_id": "photo-1",
                "mime_type": "image/jpeg",
                "size_bytes": 1234,
                "width": 200,
                "height": 100,
            }
        ],
    )
    day_session = _day_session("u-1")
    existing_task = get_pdca_task(task_id)
    assert isinstance(existing_task, dict)
    _ = enqueue_pdca_slice(
        context=_context(envelope, "cid-photo-1"),
        task_record=_task_record(
            envelope=envelope,
            day_session=day_session,
            correlation_id="cid-photo-1",
            existing_task=existing_task,
        ),
    )

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    attachments = inputs[0].get("attachments")
    assert isinstance(attachments, list)
    first = attachments[0]
    assert isinstance(first, dict)
    assert str(first.get("kind") or "") == "photo"
    assert str(first.get("file_id") or "") == "photo-1"
    assert str(first.get("mime_type") or "") == "image/jpeg"
    assert int(first.get("width") or 0) == 200
    assert int(first.get("height") or 0) == 100


def test_enqueue_pdca_slice_creates_new_task_instead_of_routing_to_running_owner_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    running_task = upsert_pdca_task(
        {
            "task_id": "task-running-owner",
            "owner_id": "u-1",
            "conversation_key": "telegram:old-thread",
            "status": "running",
            "metadata": {
                "inputs": [],
                "next_unconsumed_index": 0,
                "input_dirty": False,
            },
        }
    )
    conversation_task = upsert_pdca_task(
        {
            "task_id": "task-conversation-latest",
            "owner_id": "u-1",
            "conversation_key": "telegram:8553589429",
            "status": "queued",
            "metadata": {
                "inputs": [],
                "next_unconsumed_index": 0,
                "input_dirty": False,
            },
        }
    )

    envelope = _envelope(message_id="m-owner-route", text="new greeting")
    day_session = _day_session("u-1")
    returned = enqueue_pdca_slice(
        context=_context(envelope, "cid-owner-route"),
        task_record=_task_record(
            envelope=envelope,
            day_session=day_session,
            correlation_id="cid-owner-route",
        ),
    )
    assert returned != running_task
    assert returned != conversation_task

    running = get_pdca_task(running_task)
    assert isinstance(running, dict)
    running_metadata = running.get("metadata") if isinstance(running.get("metadata"), dict) else {}
    running_inputs = running_metadata.get("inputs")
    assert isinstance(running_inputs, list)
    assert len(running_inputs) == 0

    created = get_pdca_task(returned)
    assert isinstance(created, dict)
    assert str(created.get("conversation_key") or "") == "telegram:8553589429"
    metadata = created.get("metadata") if isinstance(created.get("metadata"), dict) else {}
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert len(inputs) == 1
    assert str(inputs[0].get("text") or "") == "new greeting"


def test_message_boundary_selects_only_waiting_same_conversation_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    _ = upsert_pdca_task(
        {
            "task_id": "task-running-owner-boundary",
            "owner_id": "u-1",
            "conversation_key": "telegram:old-thread",
            "status": "running",
        }
    )
    _ = upsert_pdca_task(
        {
            "task_id": "task-queued-same-conversation",
            "owner_id": "u-1",
            "conversation_key": "telegram:8553589429",
            "status": "queued",
        }
    )
    selected = select_pending_pdca_task_for_message(
        envelope=_envelope(message_id="m-boundary-1", text="hi"),
        session_user_id="u-1",
    )
    assert selected is None

    waiting_task = upsert_pdca_task(
        {
            "task_id": "task-waiting-same-conversation",
            "owner_id": "u-1",
            "conversation_key": "telegram:8553589429",
            "status": "waiting_user",
        }
    )
    selected = select_pending_pdca_task_for_message(
        envelope=_envelope(message_id="m-boundary-2", text="answer"),
        session_user_id="u-1",
    )
    assert isinstance(selected, dict)
    assert str(selected.get("task_id") or "") == waiting_task


def test_enqueue_pdca_slice_force_new_task_skips_running_task(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    running_task = upsert_pdca_task(
        {
            "task_id": "task-running-owner-force-new",
            "owner_id": "u-1",
            "conversation_key": "telegram:old-thread",
            "status": "running",
            "metadata": {
                "inputs": [],
                "next_unconsumed_index": 0,
                "input_dirty": False,
            },
        }
    )

    envelope = _envelope(
        message_id="m-force-new",
        text="run job prompt in isolated task",
        controls={"force_new_task": True},
        metadata={"source": "job_runner", "job": {"job_id": "job_123"}},
    )
    day_session = _day_session("u-1")
    returned = enqueue_pdca_slice(
        context=_context(envelope, "cid-force-new"),
        task_record=_task_record(
            envelope=envelope,
            day_session=day_session,
            correlation_id="cid-force-new",
        ),
    )
    assert returned != running_task

    running = get_pdca_task(running_task)
    assert isinstance(running, dict)
    running_metadata = running.get("metadata") if isinstance(running.get("metadata"), dict) else {}
    running_inputs = running_metadata.get("inputs")
    assert isinstance(running_inputs, list)
    assert len(running_inputs) == 0

    created = get_pdca_task(returned)
    assert isinstance(created, dict)
    metadata = created.get("metadata") if isinstance(created.get("metadata"), dict) else {}
    assert str(metadata.get("trigger_source") or "") == "job_runner"
    assert str(metadata.get("trigger_job_id") or "") == "job_123"
    assert bool(metadata.get("force_new_task")) is True
