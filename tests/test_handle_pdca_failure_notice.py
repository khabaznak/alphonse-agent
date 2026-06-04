from __future__ import annotations

from pathlib import Path

from alphonse.agent.actions.handle_pdca_failure_notice import HandlePdcaFailureNoticeAction
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import list_pdca_events
from alphonse.agent.nervous_system.pdca_queue_store import save_pdca_checkpoint
from alphonse.agent.nervous_system.pdca_queue_store import upsert_pdca_task
from alphonse.agent.nervous_system.senses.bus import Signal


def test_pdca_failure_notice_sends_deterministic_reply_once(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "failed",
            "metadata": {
                "last_user_channel": "telegram",
                "last_user_target": "8553589429",
            },
        }
    )
    action = HandlePdcaFailureNoticeAction()
    signal = Signal(
        type="pdca.failed",
        payload={
            "task_id": task_id,
            "correlation_id": "cid-fail-1",
            "failure_code": "engine_unavailable",
            "user_notice_required": True,
        },
        source="handle_pdca_slice_request",
    )
    first = action.execute({"signal": signal})
    second = action.execute({"signal": signal})

    assert first.delivers_message is True
    assert "inference engine is currently unavailable" in str(first.payload.get("message") or "")
    assert second.intention_key == "NOOP"
    events = list_pdca_events(task_id=task_id, limit=20)
    sent = [event for event in events if event["event_type"] == "failure.notice.sent"]
    assert len(sent) == 1


def test_pdca_failure_notice_prefers_task_specific_reason(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    reason = 'I could not find the "AC" entity.'
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "failed",
            "last_error": reason,
            "metadata": {
                "last_user_channel": "telegram",
                "last_user_target": "8553589429",
            },
        }
    )
    action = HandlePdcaFailureNoticeAction()
    signal = Signal(
        type="pdca.failed",
        payload={
            "task_id": task_id,
            "correlation_id": "cid-fail-specific",
            "failure_code": "engine_unavailable",
            "user_notice_required": True,
        },
        source="handle_pdca_slice_request",
    )
    result = action.execute({"signal": signal})

    assert result.delivers_message is True
    assert result.payload.get("message") == reason
    direct_reply = result.payload.get("direct_reply")
    assert isinstance(direct_reply, dict)
    assert direct_reply.get("text") == reason
    events = list_pdca_events(task_id=task_id, limit=20)
    sent = [event for event in events if event["event_type"] == "failure.notice.sent"]
    assert sent
    assert sent[-1]["payload"].get("message") == reason


def test_pdca_failure_notice_uses_checkpoint_failure_text(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    reason = "The scheduler rejected the reminder time."
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "failed",
            "metadata": {
                "last_user_channel": "telegram",
                "last_user_target": "8553589429",
            },
        }
    )
    task_record = TaskRecord(task_id=task_id, status="failed")
    task_record.outcome = {
        "kind": "task_failed",
        "summary": reason,
        "final_text": reason,
    }
    save_pdca_checkpoint(
        task_id=task_id,
        state={"task_record": task_record},
        expected_version=0,
    )
    action = HandlePdcaFailureNoticeAction()
    signal = Signal(
        type="pdca.failed",
        payload={
            "task_id": task_id,
            "correlation_id": "cid-fail-checkpoint",
            "failure_code": "engine_unavailable",
            "user_notice_required": True,
        },
        source="handle_pdca_slice_request",
    )
    result = action.execute({"signal": signal})

    assert result.delivers_message is True
    assert result.payload.get("message") == reason


def test_pdca_failure_notice_skips_when_not_required(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-2",
            "conversation_key": "telegram:8553589429",
            "status": "failed",
            "metadata": {
                "last_user_channel": "telegram",
                "last_user_target": "8553589429",
            },
        }
    )
    action = HandlePdcaFailureNoticeAction()
    signal = Signal(
        type="pdca.failed",
        payload={
            "task_id": task_id,
            "correlation_id": "cid-fail-2",
            "failure_code": "execution_failed",
            "user_notice_required": False,
        },
        source="handle_pdca_slice_request",
    )
    result = action.execute({"signal": signal})
    assert result.intention_key == "NOOP"
