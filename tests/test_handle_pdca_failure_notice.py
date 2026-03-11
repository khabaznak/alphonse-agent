from __future__ import annotations

from pathlib import Path

from alphonse.agent.actions.handle_pdca_failure_notice import HandlePdcaFailureNoticeAction
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import list_pdca_events
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

    assert first.intention_key == "MESSAGE_READY"
    assert "inference engine is currently unavailable" in str(first.payload.get("message") or "")
    assert second.intention_key == "NOOP"
    events = list_pdca_events(task_id=task_id, limit=20)
    sent = [event for event in events if event["event_type"] == "failure.notice.sent"]
    assert len(sent) == 1


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
