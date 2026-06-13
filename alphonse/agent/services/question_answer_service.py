from __future__ import annotations

from typing import Any

from alphonse.agent import identity
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.pending_questions import answer_pending_question
from alphonse.agent.nervous_system.pending_questions import find_pending_by_reply
from alphonse.agent.nervous_system.pending_questions import list_pending_for_respondent
from alphonse.agent.nervous_system.pending_questions import expire_pending_questions
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_task
from alphonse.agent.nervous_system.pdca_queue_store import load_pdca_checkpoint
from alphonse.agent.nervous_system.pdca_queue_store import save_pdca_checkpoint
from alphonse.agent.nervous_system.pdca_queue_store import update_pdca_task_metadata
from alphonse.agent.nervous_system.pdca_queue_store import update_pdca_task_status
from alphonse.agent.services.pdca_ingress import _append_input_record
from alphonse.agent.services.pdca_ingress import now_iso
from alphonse.agent.services.pdca_queue_runner import emit_pdca_dispatch_kick
from alphonse.agent.tools.send_message_tool import SendMessageTool


def process_expired_questions() -> int:
    expired = expire_pending_questions()
    for question in expired:
        task_id = str(question.get("task_id") or "").strip()
        task = get_pdca_task(task_id)
        if isinstance(task, dict):
            metadata = dict(task.get("metadata") or {}) if isinstance(task.get("metadata"), dict) else {}
            metadata["blocked_reason"] = "question_expired"
            metadata["expired_question_id"] = str(question.get("question_id") or "").strip() or None
            update_pdca_task_metadata(task_id=task_id, metadata=metadata)
            update_pdca_task_status(task_id=task_id, status="waiting_user", last_error="question_expired")
            append_pdca_event(
                task_id=task_id,
                event_type="question.expired",
                payload={"question_id": question.get("question_id")},
                correlation_id=f"question-expired:{question.get('question_id')}",
            )
        originator_id = str(question.get("originator_user_id") or "").strip()
        if originator_id:
            SendMessageTool().execute(
                state={
                    "actor_person_id": originator_id,
                    "correlation_id": f"question-expired:{question.get('question_id')}",
                },
                UserId=originator_id,
                Message="The question expired before I received an answer. Your task remains parked.",
            )
    return len(expired)
from alphonse.agent.services.question_session_projection import project_question_exchange


def route_inbound_question_answer(
    *, payload: dict[str, Any], respondent_user_id: str | None, correlation_id: str, bus: Any
) -> dict[str, Any]:
    respondent_id = str(respondent_user_id or "").strip()
    text = str(payload.get("text") or "").strip()
    if not respondent_id or not text:
        return {"handled": False}
    reply_to = str(payload.get("reply_to_provider_message_id") or "").strip()
    question = (
        find_pending_by_reply(
            respondent_user_id=respondent_id,
            reply_to_provider_message_id=reply_to,
        )
        if reply_to
        else None
    )
    if question is None:
        pending = list_pending_for_respondent(respondent_id)
        if reply_to and any(str(item.get("outbound_provider_message_id") or "").strip() for item in pending):
            return {"handled": False}
        if len(pending) > 1:
            append_pdca_event(
                task_id=str(pending[-1].get("task_id") or ""),
                event_type="question.answer_ambiguous",
                payload={"respondent_user_id": respondent_id, "pending_count": len(pending)},
                correlation_id=correlation_id,
            )
            return {
                "handled": True,
                "ambiguous": True,
                "message": "You have more than one pending question. Please reply directly to the question you are answering.",
            }
        if len(pending) != 1:
            return {"handled": False}
        question = pending[0]
    answered = answer_pending_question(
        question_id=str(question.get("question_id") or ""),
        respondent_user_id=respondent_id,
        answer_text=text,
        inbound_provider_message_id=str(payload.get("provider_message_id") or "").strip() or None,
    )
    if answered is None:
        return {"handled": False}
    task_id = str(answered.get("task_id") or "").strip()
    task = get_pdca_task(task_id)
    if not isinstance(task, dict):
        return {"handled": True, "question_id": answered.get("question_id"), "task_id": task_id}
    metadata = dict(task.get("metadata") or {}) if isinstance(task.get("metadata"), dict) else {}
    metadata, _ = _append_input_record(
        metadata=metadata,
        message_id=str(payload.get("provider_message_id") or "").strip() or None,
        channel_type=str(payload.get("service_key") or "").strip(),
        correlation_id=correlation_id,
        user_text=text,
        attachments=[],
        actor_id=respondent_id,
        now=now_iso(),
        input_kind="question_answer",
        question_id=str(answered.get("question_id") or ""),
        respondent_user_id=respondent_id,
    )
    metadata["pending_user_text"] = text
    metadata["last_user_message"] = text
    update_pdca_task_metadata(task_id=task_id, metadata=metadata)
    _append_answer_to_checkpoint(task_id=task_id, respondent_user_id=respondent_id, answer_text=text)
    update_pdca_task_status(task_id=task_id, status="queued")
    respondent_channel = str(payload.get("service_key") or "api").strip() or "api"
    project_question_exchange(
        user_id=respondent_id,
        channel=respondent_channel,
        user_message=text,
    )
    originator_id = str(answered.get("originator_user_id") or "").strip()
    if originator_id and originator_id != respondent_id:
        respondent = identity.get_user(respondent_id)
        respondent_name = str((respondent or {}).get("display_name") or respondent_id).strip()
        project_question_exchange(
            user_id=originator_id,
            channel=str(answered.get("originator_conversation_key") or "api").partition(":")[0],
            assistant_message=f"{respondent_name} answered: {text}. I am resuming your task.",
        )
    append_pdca_event(
        task_id=task_id,
        event_type="question.answered",
        payload={"question_id": answered.get("question_id"), "respondent_user_id": respondent_id},
        correlation_id=correlation_id,
    )
    emit_pdca_dispatch_kick(
        bus=bus if hasattr(bus, "emit") else None,
        task_id=task_id,
        reason="question_answered",
        correlation_id=correlation_id,
        owner_id=str(task.get("owner_id") or "").strip() or None,
        conversation_key=str(task.get("conversation_key") or "").strip() or None,
    )
    _notify_originator(answered=answered, respondent_user_id=respondent_id, answer_text=text, payload=payload)
    append_pdca_event(
        task_id=task_id,
        event_type="question.resumed",
        payload={"question_id": answered.get("question_id")},
        correlation_id=correlation_id,
    )
    return {"handled": True, "question_id": answered.get("question_id"), "task_id": task_id}


def _append_answer_to_checkpoint(*, task_id: str, respondent_user_id: str, answer_text: str) -> None:
    checkpoint = load_pdca_checkpoint(task_id)
    if not isinstance(checkpoint, dict):
        return
    state = dict(checkpoint.get("state") or {}) if isinstance(checkpoint.get("state"), dict) else {}
    raw_record = state.get("task_record")
    if not isinstance(raw_record, dict):
        return
    record = TaskRecord.from_dict(raw_record)
    user = identity.get_user(respondent_user_id)
    name = str((user or {}).get("display_name") or respondent_user_id).strip()
    record.append_recent_conversation_line(f"{name} answered: {answer_text}")
    record.status = "running"
    state["task_record"] = record.to_dict()
    save_pdca_checkpoint(
        task_id=task_id,
        state=state,
        expected_version=int(checkpoint.get("version") or 0),
    )


def _notify_originator(
    *, answered: dict[str, Any], respondent_user_id: str, answer_text: str, payload: dict[str, Any]
) -> None:
    originator_id = str(answered.get("originator_user_id") or "").strip()
    if not originator_id or originator_id == respondent_user_id:
        return
    respondent = identity.get_user(respondent_user_id)
    name = str((respondent or {}).get("display_name") or respondent_user_id).strip()
    SendMessageTool().execute(
        state={
            "channel_type": str(payload.get("service_key") or "api").strip(),
            "channel_target": str(payload.get("channel_target") or "").strip(),
            "actor_person_id": originator_id,
            "correlation_id": f"question-answer:{answered.get('question_id')}",
        },
        UserId=originator_id,
        Message=f"{name} answered: {answer_text}. I am resuming your task.",
    )
