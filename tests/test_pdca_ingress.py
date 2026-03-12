from __future__ import annotations

from pathlib import Path

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
