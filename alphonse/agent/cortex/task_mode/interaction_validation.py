from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from alphonse.agent.cortex.task_mode.task_record import TaskRecord

_WAIT_MARKERS = (
    "wait for",
    "wait until",
    "await",
    "waiting for",
    "esperar",
    "esperando",
    "hasta que responda",
    "su respuesta",
    "respuesta del usuario",
    "answer before",
    "user answer",
    "user response",
)
_QUESTION_START = re.compile(
    r"^(what|when|where|which|who|why|how|can|could|would|do|does|did|is|are|was|were|"
    r"qué|que|cuándo|cuando|dónde|donde|cuál|cual|quién|quien|por qué|por que|cómo|como|"
    r"puedes|podrías|podrias|quieres|deseas)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlannerInteractionIssue:
    code: str
    message: str


def validate_planner_interaction(
    *,
    task_record: TaskRecord,
    planner_output: dict[str, Any],
) -> PlannerInteractionIssue | None:
    tool_call = planner_output.get("tool_call") if isinstance(planner_output, dict) else None
    if not isinstance(tool_call, dict):
        return None
    tool_name = str(tool_call.get("tool_name") or "").strip()
    if tool_name != "communication.send_message":
        return None
    args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
    message = str(args.get("Message") or args.get("message") or "").strip()
    intent = str(planner_output.get("planner_intent") or "").strip()
    if not _is_question(message) or not _requires_answer_before_continuing(intent):
        return None
    if _equivalent_message_in_history(message=message, history=task_record.get_tool_call_history_md()):
        return PlannerInteractionIssue(
            code="repeated_blocking_question_delivery",
            message="Equivalent blocking question was already delivered in this task.",
        )
    return PlannerInteractionIssue(
        code="blocking_question_requires_ask_tool",
        message=(
            "A question that blocks task continuation must use communication.ask_question, "
            "not communication.send_message."
        ),
    )


def planner_repair_instruction(issue: PlannerInteractionIssue) -> str:
    return (
        "The previous tool selection is invalid. "
        f"{issue.message} Return one corrected canonical tool call. "
        "Use communication.ask_question with `question` and optional `respondent_user_id`; "
        "do not resend the question with communication.send_message."
    )


def _is_question(message: str) -> bool:
    rendered = " ".join(str(message or "").strip().split())
    return bool(rendered and ("?" in rendered or _QUESTION_START.search(rendered)))


def _requires_answer_before_continuing(intent: str) -> bool:
    normalized = " ".join(str(intent or "").strip().lower().split())
    return bool(normalized and any(marker in normalized for marker in _WAIT_MARKERS))


def _equivalent_message_in_history(*, message: str, history: str) -> bool:
    normalized_message = _normalize(message)
    normalized_history = _normalize(history)
    return bool(normalized_message and normalized_message in normalized_history)


def _normalize(value: str) -> str:
    return " ".join(str(value or "").lower().replace('"', " ").split())
