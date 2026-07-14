"""v2-native AskQuestion tool."""

from __future__ import annotations

from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry import ToolDefinition

ASK_QUESTION_TOOL_ID = "native.ask_question"
ASK_QUESTION_TOOL_NAME = "ask_question"

ASK_QUESTION_ARGUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {
            "type": "string",
            "description": "The user-visible question to ask.",
        },
        "question_kind": {
            "type": "string",
            "enum": ["open_text", "yes_no", "single_choice"],
            "description": "The answer form to present.",
        },
        "choices": {
            "type": "array",
            "description": "Choices for single_choice questions.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["id", "label"],
            },
        },
        "respondent_user_id": {
            "type": "string",
            "description": "Optional registered user to ask; defaults to the task owner.",
        },
        "expires_in_seconds": {
            "type": "integer",
            "description": "Optional expiry window, in seconds.",
        },
    },
    "required": ["question", "question_kind"],
}


def build_ask_question_tool_definition() -> ToolDefinition:
    """Build the native AskQuestion tool definition."""
    descriptor = ToolDescriptor(
        tool_id=ASK_QUESTION_TOOL_ID,
        name=ASK_QUESTION_TOOL_NAME,
        kind=ToolKind.NATIVE,
        description=(
            "Ask a required user or delegated-user question, park the active task, "
            "and resume the task when the answer arrives."
        ),
        argument_schema=dict(ASK_QUESTION_ARGUMENT_SCHEMA),
        capabilities=("conversation", "interrupt", "user_input", "task_parking"),
        tags=("native", "conversation", "interrupt"),
    )
    return ToolDefinition(
        descriptor=descriptor,
        callable=execute_ask_question,
        argument_schema=dict(ASK_QUESTION_ARGUMENT_SCHEMA),
        enabled=True,
        accepts_context=True,
    )


def execute_ask_question(
    arguments: dict[str, Any],
    *,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Create a question interrupt and park the active task."""
    if context is None:
        raise ValueError("ask_question_context_required")
    question = str(arguments.get("question") or "").strip()
    if not question:
        raise ValueError("ask_question_question_required")
    kind = str(arguments.get("question_kind") or "open_text").strip() or "open_text"
    store = context.question_store if context.question_store is not None else SQLiteQuestionStore.default()
    if not hasattr(store, "create_question"):
        raise TypeError("ask_question_store_invalid")

    delivery_metadata: dict[str, Any] = {
        "origin_channel": dict(context.task.metadata.get("channel"))
        if isinstance(context.task.metadata.get("channel"), dict)
        else {},
    }
    interrupt = store.create_question(
        task=context.task,
        question=question,
        kind=kind,
        choices=arguments.get("choices"),
        respondent_user_id=str(arguments.get("respondent_user_id") or "").strip() or None,
        expires_in_seconds=_optional_int(arguments.get("expires_in_seconds")),
        delivery_metadata=delivery_metadata,
    )
    if context.memory is not None and interrupt.respondent_user_id != str(context.task.user or ""):
        child_id = str(interrupt.metadata.get("child_task_id") or "").strip()
        if child_id:
            recipient_task = TaskState(task_id=child_id, user=interrupt.respondent_user_id, project_id=context.task.project_id, goal=question, status="waiting_user")
            context.memory.ensure_project_scope(user_id=recipient_task.user or "", project_id=recipient_task.project_id)
            if recipient_task.project_id and context.project_store is not None:
                add_member = getattr(context.project_store, "add_member", None)
                if callable(add_member):
                    add_member(recipient_task.project_id, recipient_task.user or "")
            context.memory.start_task(recipient_task)
            context.memory.event(recipient_task, "Delegated Question", {"from": context.task.user, "question": question, "question_id": interrupt.question_id})

    delivery_result: Any = None
    if context.delivery_sink is not None:
        try:
            delivery_result = context.delivery_sink(
                {
                    "event_type": "question.deliver",
                    "question": interrupt.to_dict(),
                    "task": context.task.to_dict(),
                }
            )
        except Exception:
            store.cancel_question(interrupt.question_id)
            raise
        if isinstance(delivery_result, dict) and delivery_result:
            if delivery_result.get("exception"):
                store.cancel_question(interrupt.question_id)
                raise RuntimeError("ask_question_delivery_failed")
            store.bind_delivery_metadata(question_id=interrupt.question_id, metadata=delivery_result)

    if context.ui_event_sink is not None:
        context.ui_event_sink(
            _ui_event(
                "question_interrupt_opened",
                {
                    "question": interrupt.to_dict(),
                    "delivery_result": delivery_result if isinstance(delivery_result, dict) else None,
                },
            )
        )

    return {
        "question_interrupt": interrupt.to_dict(),
        "waiting_for_answer": True,
        "delivery_result": delivery_result if isinstance(delivery_result, dict) else None,
    }


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ask_question_expires_in_seconds_invalid") from exc


def _ui_event(event_type: str, payload: dict[str, Any]) -> Any:
    from alphonse.agent_v2.core.core import CoreUiEvent

    return CoreUiEvent(event_type=event_type, payload=payload)
