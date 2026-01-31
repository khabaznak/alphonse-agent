from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.cognition.skills.command_plans import (
    CommandPlan,
    CreateReminderPlan,
    GreetingPlan,
    SendMessagePlan,
    UnknownPlan,
)
from alphonse.agent.nervous_system.timed_commands import insert_timed_signal_from_plan


@dataclass(frozen=True)
class ExecutorContext:
    actor_person_id: str | None


def execute_greeting_v1(plan: GreetingPlan, _context: ExecutorContext) -> str:
    return "¡Hola! ¿En qué te ayudo?"


def execute_unknown_v1(plan: UnknownPlan, _context: ExecutorContext) -> str:
    if plan.questions:
        return plan.questions[0]
    return "¿Puedes aclarar qué necesitas?"


def execute_create_reminder_v1(plan: CreateReminderPlan, _context: ExecutorContext) -> str:
    evidence = plan.intent_evidence
    if not (evidence.score >= 0.6 or evidence.reminder_verbs or evidence.time_hints or evidence.quoted_spans):
        raise ValueError("insufficient_intent_evidence")
    insert_timed_signal_from_plan(plan)
    trigger_at = plan.payload.schedule.trigger_at or "(por confirmar)"
    return f"Programé el recordatorio para {trigger_at}."


def execute_send_message_v1(plan: SendMessagePlan, context: ExecutorContext) -> str:
    evidence = plan.intent_evidence
    if not (evidence.score >= 0.6 or evidence.reminder_verbs or evidence.time_hints or evidence.quoted_spans):
        raise ValueError("insufficient_intent_evidence")
    target_id = plan.payload.target.person_ref.id or plan.payload.target.person_ref.name
    if target_id and context.actor_person_id and target_id != context.actor_person_id:
        return "¿Confirmas que debo enviar ese mensaje?"
    return plan.payload.message.text


EXECUTOR_MAP = {
    "actions.execute_greeting_v1": execute_greeting_v1,
    "actions.execute_unknown_v1": execute_unknown_v1,
    "actions.execute_create_reminder_v1": execute_create_reminder_v1,
    "actions.execute_send_message_v1": execute_send_message_v1,
}
