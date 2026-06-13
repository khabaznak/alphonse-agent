from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from alphonse.agent import identity
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.pending_questions import cancel_pending_question
from alphonse.agent.nervous_system.pending_questions import bind_outbound_message
from alphonse.agent.nervous_system.pending_questions import create_pending_question
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.tools.send_message_tool import SendMessageTool
from alphonse.agent.services.question_session_projection import project_question_exchange


@dataclass(frozen=True)
class AskQuestionTool:
    canonical_name: ClassVar[str] = "askQuestion"
    capability: ClassVar[str] = "clarification"
    _send_message_tool: SendMessageTool | None = None

    def __post_init__(self) -> None:
        if self._send_message_tool is None:
            object.__setattr__(self, "_send_message_tool", SendMessageTool())

    def execute(
        self,
        *,
        question: str,
        respondent_user_id: str | None = None,
        expires_in_seconds: int | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        text = str(question or "").strip()
        if not text:
            return _failed("missing_question", "question is required")
        payload = state if isinstance(state, dict) else {}
        task_record = payload.get("task_record")
        if not isinstance(task_record, TaskRecord):
            return _failed("missing_task_record", "askQuestion requires an active PDCA task")
        task_id = str(task_record.task_id or "").strip()
        originator_id = str(task_record.user_id or payload.get("actor_person_id") or "").strip()
        respondent_id = str(respondent_user_id or originator_id).strip()
        respondent = identity.get_user(respondent_id)
        originator = identity.get_user(originator_id)
        if not task_id or not originator_id:
            return _failed("missing_task_context", "askQuestion requires task and originator identity")
        if not isinstance(respondent, dict):
            return _failed("unresolved_respondent", "respondent must be a registered user")
        service_id = identity.get_preferred_service_id(respondent_id)
        channel = identity.resolve_service_key(service_id)
        target = identity.resolve_delivery_target(user_id=respondent_id, service_id=service_id)
        if not channel or not target:
            return _failed("unresolved_delivery_target", "respondent has no preferred delivery target")
        origin_conversation = str(payload.get("conversation_key") or payload.get("chat_id") or "").strip()
        if not origin_conversation:
            origin_channel = str(payload.get("channel_type") or "api").strip()
            origin_target = str(payload.get("channel_target") or originator_id).strip()
            origin_conversation = f"{origin_channel}:{origin_target}"
        respondent_conversation = f"{channel}:{target}"
        record = create_pending_question(
            task_id=task_id,
            originator_user_id=originator_id,
            originator_conversation_key=origin_conversation,
            respondent_user_id=respondent_id,
            respondent_conversation_key=respondent_conversation,
            question_text=text,
            expires_in_seconds=expires_in_seconds,
        )
        append_pdca_event(
            task_id=task_id,
            event_type="question.created",
            payload={"question_id": record["question_id"], "respondent_user_id": respondent_id},
            correlation_id=task_record.correlation_id or None,
        )
        originator_name = str((originator or {}).get("display_name") or originator_id).strip()
        delivered_text = text if respondent_id == originator_id else f"{originator_name} asked me to ask you: {text}"
        result = self._send_message_tool.execute(
            state=payload,
            UserId=respondent_id,
            Message=delivered_text,
            Channel=channel,
            correlation_id=f"question:{record['question_id']}",
        )
        if isinstance(result.get("exception"), dict):
            cancel_pending_question(str(record["question_id"]))
            append_pdca_event(
                task_id=task_id,
                event_type="question.cancelled",
                payload={"question_id": record["question_id"], "reason": "delivery_failed"},
                correlation_id=task_record.correlation_id or None,
            )
            return result
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        bind_outbound_message(
            question_id=str(record["question_id"]),
            provider_message_id=str(output.get("provider_message_id") or output.get("message_id") or "").strip() or None,
        )
        append_pdca_event(
            task_id=task_id,
            event_type="question.delivered",
            payload={"question_id": record["question_id"], "respondent_conversation_key": respondent_conversation},
            correlation_id=task_record.correlation_id or None,
        )
        respondent_name = str(respondent.get("display_name") or respondent_id).strip()
        task_record.append_recent_conversation_line(f"Alphonse asked {respondent_name}: {text}")
        task_record.append_recent_conversation_line(f"Task parked while waiting for {respondent_name}'s answer.")
        task_record.status = "waiting_user"
        project_question_exchange(
            user_id=respondent_id,
            channel=channel,
            assistant_message=delivered_text,
            task_record=task_record.to_dict() if respondent_id == originator_id else None,
        )
        if respondent_id != originator_id:
            project_question_exchange(
                user_id=originator_id,
                channel=str(payload.get("channel_type") or "api"),
                assistant_message=f"I asked {respondent_name}: {text}. I am waiting for an answer.",
                task_record=task_record.to_dict(),
            )
            self._send_message_tool.execute(
                state=payload,
                UserId=originator_id,
                Message=f"I asked {respondent_name}: {text}. I am waiting for an answer.",
                correlation_id=f"question-waiting:{record['question_id']}",
            )
        append_pdca_event(
            task_id=task_id,
            event_type="question.waiting",
            payload={"question_id": record["question_id"], "respondent_user_id": respondent_id},
            correlation_id=task_record.correlation_id or None,
        )
        return {
            "output": {
                "question_id": record["question_id"],
                "respondent_user_id": respondent_id,
                "respondent_conversation_key": respondent_conversation,
                "delivery_status": "delivered",
                "waiting_for_answer": True,
                "expires_at": record["expires_at"],
            },
            "exception": None,
            "metadata": {"tool": "askQuestion"},
        }


def _failed(code: str, message: str) -> dict[str, Any]:
    return {"output": None, "exception": {"code": code, "message": message}, "metadata": {"tool": "askQuestion"}}
