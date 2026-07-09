"""AG-UI adapter for Alphonse v2 UI events."""

from __future__ import annotations

import json
from typing import Any

from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.questions import QuestionInterrupt
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


class AgUiAdapter:
    """Translate Alphonse v2 UI events to AG-UI event dictionaries."""

    def __init__(self, *, question_store: SQLiteQuestionStore) -> None:
        self.question_store = question_store

    def map_event(self, event: CoreUiEvent) -> list[dict[str, Any]]:
        payload = dict(event.payload or {})
        if event.event_type == "run_started":
            task = _task(payload)
            return [
                {
                    "type": "RUN_STARTED",
                    "threadId": _thread_id(task),
                    "runId": _run_id(task),
                    "input": {"task": task},
                }
            ]
        if event.event_type == "run_finished":
            return self._map_run_finished(payload)
        if event.event_type == "state_snapshot":
            return [{"type": "STATE_SNAPSHOT", "snapshot": dict(payload)}]
        if event.event_type == "tool_call_started":
            args = json.dumps(payload.get("arguments") or {}, sort_keys=True)
            tool_call_id = str(payload.get("tool_call_id") or "")
            return [
                {
                    "type": "TOOL_CALL_START",
                    "toolCallId": tool_call_id,
                    "toolCallName": str(payload.get("tool_name") or payload.get("tool_id") or ""),
                },
                {"type": "TOOL_CALL_ARGS", "toolCallId": tool_call_id, "delta": args},
                {"type": "TOOL_CALL_END", "toolCallId": tool_call_id},
            ]
        if event.event_type == "tool_call_result":
            return [
                {
                    "type": "TOOL_CALL_RESULT",
                    "messageId": str(payload.get("message_id") or payload.get("tool_call_id") or ""),
                    "toolCallId": str(payload.get("tool_call_id") or ""),
                    "role": "tool",
                    "content": json.dumps(payload, sort_keys=True),
                }
            ]
        if event.event_type.startswith("question_interrupt_"):
            return [{"type": "CUSTOM", "name": event.event_type, "value": payload}]
        return [{"type": "CUSTOM", "name": event.event_type, "value": payload}]

    def map_activity(self, event: CoreActivityEvent) -> dict[str, Any]:
        return {
            "type": "ACTIVITY_SNAPSHOT",
            "messageId": f"activity:{event.phase.value}",
            "activityType": event.phase.value.upper(),
            "content": {
                "label": event.label,
                "message": event.message,
                "speaker": event.speaker,
            },
            "replace": True,
        }

    def resume(self, run_agent_input: dict[str, Any]) -> list[CoreUiEvent]:
        """Map AG-UI resume input into v2 question answer events."""
        events: list[CoreUiEvent] = []
        for item in run_agent_input.get("resume") or []:
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("interruptId") or "").strip()
            if not question_id:
                continue
            if str(item.get("status") or "").strip().lower() == "cancelled":
                cancelled = self.question_store.cancel_question(question_id)
                events.append(
                    CoreUiEvent(
                        event_type="question_interrupt_cancelled",
                        payload={"question_id": question_id, "cancelled": cancelled},
                    )
                )
                continue
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            question = self.question_store.get_question(question_id)
            respondent = str(payload.get("respondent_user_id") or "").strip()
            if not respondent and question is not None:
                respondent = question.respondent_user_id
            result = self.question_store.route_answer(
                respondent_user_id=respondent,
                payload=dict(payload),
                question_id=question_id,
            )
            events.append(
                CoreUiEvent(
                    event_type="question_interrupt_resolved" if result.resumed_task is not None else "question_answer_rejected",
                    payload=result.to_dict(),
                )
            )
        return events

    def _map_run_finished(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        task = _task(payload)
        question_payload = payload.get("question")
        if isinstance(question_payload, dict):
            question = QuestionInterrupt.from_dict(question_payload)
            return [
                {"type": "STATE_SNAPSHOT", "snapshot": {"task_state": task}},
                {"type": "MESSAGES_SNAPSHOT", "messages": _messages_from_task(task)},
                {
                    "type": "RUN_FINISHED",
                    "threadId": question.thread_id,
                    "runId": question.run_id,
                    "outcome": {
                        "type": "interrupt",
                        "interrupts": [_interrupt_from_question(question)],
                    },
                },
            ]
        return [
            {
                "type": "RUN_FINISHED",
                "threadId": _thread_id(task),
                "runId": _run_id(task),
                "outcome": {"type": "success"},
            }
        ]


def _interrupt_from_question(question: QuestionInterrupt) -> dict[str, Any]:
    reason = "confirmation" if question.kind == "yes_no" else "input_required"
    return {
        "id": question.question_id,
        "reason": reason,
        "message": question.message,
        "responseSchema": question.response_schema,
        "metadata": {
            "alphonse": {
                "taskId": question.task_id,
                "kind": question.kind,
                "respondentUserId": question.respondent_user_id,
                "choices": [choice.to_dict() for choice in question.choices],
            }
        },
    }


def _task(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload.get("task") or payload.get("task_state")
    return dict(task) if isinstance(task, dict) else {}


def _thread_id(task: dict[str, Any]) -> str:
    return str(task.get("project_id") or task.get("correlation_id") or task.get("task_id") or "thread").strip()


def _run_id(task: dict[str, Any]) -> str:
    return str(task.get("message_id") or task.get("task_id") or task.get("correlation_id") or "run").strip()


def _messages_from_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = str(task.get("recent_conversation_md") or "").strip()
    if not conversation or conversation == "- (none)":
        return []
    return [{"id": "recent-conversation", "role": "assistant", "content": conversation}]
