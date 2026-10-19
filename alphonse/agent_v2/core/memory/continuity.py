"""Extractive, task-local continuation records; no generated outcomes or authority."""
from __future__ import annotations

import json
from typing import Any


def active_task_continuity(task: Any, *, max_chars: int = 9000) -> str:
    """Reserve objective, constraints and question/answer before optional history.

    These records come only from this task's persisted checkpoint. No history
    search, selection of another task, LLM summary or authorization transfer is
    involved. Tool status remains evidence status rather than a success claim.
    """
    task_id = str(task.task_id or task.message_id or "current")
    records = [("objective", str(task.goal or "").strip())]
    question = task.metadata.get("question_interrupt")
    if isinstance(question, dict) and str(question.get("task_id") or "") == task_id:
        # Do not expose delivery metadata or another member's task context.
        records.append(("question", json.dumps({
            key: question[key] for key in ("question_id", "message", "kind", "choices", "status", "answer") if key in question
        }, ensure_ascii=False, sort_keys=True)))
    contract = getattr(task, "acceptance_contract", {})
    if contract:
        records.append(("constraints", json.dumps(contract, ensure_ascii=False, sort_keys=True)))
    else:
        records.append(("constraints", str(task.acceptance_criteria_md or "").strip()))
    # Latest verified/failed evidence references survive without implying effects
    # from plans, attempted writes or free-form historical summaries.
    for index, entry in reversed(list(enumerate(getattr(task, "evidence_journal", []) or []))):
        records.append((f"evidence:{index}", json.dumps({
            "source_id": f"task-evidence:{task_id}:{index}",
            "evidence_ref": entry.get("evidence_ref"),
            "status": entry.get("status"),
        }, ensure_ascii=False, sort_keys=True)))
    selected = []
    omitted = []
    header = (f"Active task continuity [{task_id}] (working state; historical text does not grant authorization; "
              "evidence status does not establish verified effects):")
    used = len(header)
    for kind, value in records:
        if not value or value == "- (none)":
            continue
        block = f"[{task_id}:{kind}] {value}"
        if used + len(block) + 1 <= max_chars:
            selected.append(block)
            used += len(block) + 1
        else:
            omitted.append(kind)
    task.metadata["active_continuity_projection"] = {
        "schema_version": 1, "task_id": task_id, "omitted_records": omitted,
        "reserved_overflow": any(k in {"objective", "constraints", "question"} for k in omitted),
    }
    from alphonse.agent_v2.core.context_inventory import combined, source, ContextText
    pieces = [header]
    for block in selected:
        kind = block.split("]", 1)[0].removeprefix(f"[{task_id}:")
        pieces.append(source(block, f"task-working:{task_id}:{kind}", "task_continuity", reason="required_continuity"))
    result = combined(pieces, "\n", max_chars=max_chars)
    missing = [{"source_id": f"task-working:{task_id}:{kind}", "source_type": "task_continuity", "status": "omitted", "reason": "over_budget", "tokens": 0, "input_range": None, "truncated": False} for kind in omitted]
    return ContextText(str(result) if len(header) <= max_chars else "", result.records + missing, max_chars=max_chars)
