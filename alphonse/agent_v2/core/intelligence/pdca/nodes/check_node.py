"""Check node for the v2 PDCA graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.queue import MessageSelector

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext


def check_node(task: TaskState, context: CoreLoopContext | None = None) -> TaskState:
    """Classify the task and fold related steering messages into it."""
    is_new_task = _markdown_is_empty(task.acceptance_criteria_md)
    steering_count = _consume_steering_messages(task, context)

    if is_new_task:
        verdict = "new"
        reason = "No acceptance criteria were present; treating this as a new task."
    elif steering_count > 0:
        verdict = "steer"
        reason = "Related queued messages were consumed as steering for this task."
    else:
        verdict = "wip"
        reason = "Acceptance criteria exist, but no steering messages were available yet."

    task.set_check_result(
        verdict=verdict,
        reason=reason,
        confidence=1.0,
        new_message_count=steering_count,
    )
    return task


def _consume_steering_messages(task: TaskState, context: CoreLoopContext | None) -> int:
    if context is None:
        return 0

    consumed = 0
    consumed += _consume_matching(
        task,
        context,
        MessageSelector(user=task.user, project_id=task.project_id),
    )
    if task.correlation_id:
        consumed += _consume_matching(
            task,
            context,
            MessageSelector(correlation_id=task.correlation_id),
        )
    return consumed


def _consume_matching(task: TaskState, context: CoreLoopContext, selector: MessageSelector) -> int:
    consumed = 0
    while True:
        queued = context.consume_message(selector)
        if queued is None:
            return consumed
        task.append_conversation_message(queued.message.user, queued.message.prompt)
        consumed += 1


def _markdown_is_empty(value: str) -> bool:
    rendered = str(value or "").strip()
    return not rendered or rendered == "- (none)"
