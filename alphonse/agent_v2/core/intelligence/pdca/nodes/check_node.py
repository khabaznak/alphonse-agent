"""Check node for the v2 PDCA graph."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.acceptance_contract import apply_status_patch
from alphonse.agent_v2.core.messages.queue import MessageSelector

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext


def check_node(task: TaskState, context: CoreLoopContext | None = None, *, consume_steering: bool = True) -> TaskState:
    """Classify the task, optionally folding related steering messages into it."""
    if context is not None:
        context.emit_activity(
            phase=ImprovementPhase.CHECK,
            label="deliberating",
            message="Reviewing the task and queued steering messages.",
        )
    is_new_task = _markdown_is_empty(task.acceptance_criteria_md)
    steering_count = _consume_steering_messages(task, context) if consume_steering else 0

    if is_new_task:
        verdict = "new"
        reason = "No acceptance criteria were present; treating this as a new task."
    elif steering_count > 0:
        verdict = "steer"
        reason = "New steering was incorporated and requires an explicit acceptance-contract amendment review."
    else:
        verdict = "wip"
        reason = _review_wip_acceptance_criteria(task, context)
        if task.acceptance_criteria_all_complete():
            verdict = "mission_success"
            reason = "Every required acceptance criterion is supported by verified tool evidence."

    task.set_check_result(
        verdict=verdict,
        reason=reason,
        confidence=1.0,
        new_message_count=steering_count,
    )
    if context is not None:
        context.emit_activity(
            phase=ImprovementPhase.CHECK,
            label="criteria refreshed",
            message="Acceptance criteria are up to date after review.",
            progress={"acceptance_criteria": "" if task.acceptance_criteria_md == "- (none)" else task.acceptance_criteria_md},
        )
    return task


def _review_wip_acceptance_criteria(
    task: TaskState,
    context: CoreLoopContext | None = None,
    *,
    require_latest_call: bool = True,
) -> str:
    latest_call = task.get_latest_executed_plan_call()
    if latest_call is None and require_latest_call:
        return "Acceptance criteria exist, but no steering messages or executed tool results were available yet."
    if not _valid_evidence_refs(task):
        task.metadata["criteria_review_updated"] = False
        return "No successful tool evidence is available to satisfy acceptance criteria."

    if context is None or context.system_one is None or not callable(getattr(context.system_one, "evaluate", None)):
        from alphonse.agent_v2.system_one import SystemOneUnavailableError
        raise SystemOneUnavailableError("check_criteria_review_unavailable")
    try:
        result = context.system_one.evaluate(
            contract=task.ensure_acceptance_contract(),
            phase={"latest_executed_call": latest_call, "plan": task.plan_json},
            evidence={"entries": _evidence_journal(task)},
        )
    except Exception as exc:
        from alphonse.agent_v2.system_one import SystemOneUnavailableError
        raise SystemOneUnavailableError(f"check_criteria_review:{type(exc).__name__}") from exc
    updates = [
        dict(update) for update in result.updates
        if str(update.get("criterion_id") or "") not in set(result.ambiguous_criterion_ids)
    ]
    contract, rejected = apply_status_patch(
        task.ensure_acceptance_contract(), {"updates": updates}, valid_evidence_refs=_valid_evidence_refs(task),
    )
    task.acceptance_contract = contract
    task.sync_acceptance_criteria_view()
    task.metadata["criteria_review_rejections"] = rejected
    task.metadata["criteria_review_updated"] = bool(updates)
    task.metadata["system_one_check_review"] = result.to_metadata()
    context.emit_telemetry({
        "event": "system_one_check_review", "task_id": task.task_id,
        "duration_ms": result.duration_ms, "model": result.model,
        "ambiguous_criterion_count": len(result.ambiguous_criterion_ids),
        "usage": dict(result.usage),
    })
    task.append_update("Check reviewed acceptance evidence with Jev; ambiguous criteria remain unresolved.")
    return "Acceptance criterion statuses were reviewed against verified tool evidence."


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
        pending = context.messages.peek(selector)
        if pending is None:
            return consumed
        metadata = pending.message.metadata if isinstance(pending.message.metadata, dict) else {}
        if str(metadata.get("source") or "") in {"scheduled_task", "event_automation"}:
            # Automation occurrences must be processed as independent tasks so
            # their delivery metadata survives through outbox projection.
            return consumed
        disposition = str(metadata.get("routing_disposition") or "pdca_task")
        if disposition not in {"steering", "correlated_response"}:
            return consumed
        queued = context.consume_message(selector)
        if queued is None:
            return consumed
        task.append_conversation_message(queued.message.user, queued.message.prompt)
        if queued.message.user == task.user:
            task.merge_attachments(queued.message.metadata)
        consumed += 1
        source_ids = task.metadata.setdefault("pending_steering_message_ids", [])
        if isinstance(source_ids, list):
            source_ids.append(str(queued.message_id or ""))


def _markdown_is_empty(value: str) -> bool:
    rendered = str(value or "").strip()
    return not rendered or rendered == "- (none)"


def _valid_evidence_refs(task: TaskState) -> set[str]:
    journal = _evidence_journal(task)
    if journal:
        latest_status_by_ref: dict[str, str] = {}
        for item in journal:
            evidence_ref = str(item.get("evidence_ref") or "").strip()
            if evidence_ref:
                latest_status_by_ref[evidence_ref] = str(item.get("status") or "").strip()
        return {
            evidence_ref for evidence_ref, status in latest_status_by_ref.items() if status == "success"
        }
    refs: set[str] = set()
    try:
        calls = json.loads(task.plan_json) if task.plan_json and task.plan_json != "- (none)" else []
    except json.JSONDecodeError:
        calls = []
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict) or not isinstance(call.get("execution"), dict):
            continue
        call_id = str(call.get("id") or "").strip()
        if call_id:
            refs.add(f"tool-call:{call_id}")
    return refs


def _evidence_journal(task: TaskState) -> list[dict[str, Any]]:
    if task.evidence_journal:
        return [dict(item) for item in task.evidence_journal if isinstance(item, dict)]
    # Backward compatibility for checkpoints created before the evidence journal.
    journal: list[dict[str, Any]] = []
    try:
        calls = json.loads(task.plan_json) if task.plan_json and task.plan_json != "- (none)" else []
    except json.JSONDecodeError:
        calls = []
    for call in calls if isinstance(calls, list) else []:
        execution = call.get("execution") if isinstance(call, dict) else None
        if not isinstance(execution, dict):
            continue
        call_id = str(call.get("id") or "").strip()
        journal.append(
            {
                "evidence_ref": f"tool-call:{call_id}",
                "call_id": call_id,
                "tool_id": str(call.get("tool_id") or "program"),
                "status": str(execution.get("status") or ""),
                "result": execution.get("result"),
                "exception": execution.get("exception"),
            }
        )
    return journal
