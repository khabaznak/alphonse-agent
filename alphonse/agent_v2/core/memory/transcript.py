"""Bounded continuity from complete conversation exchanges, never sliced messages."""
from __future__ import annotations

from typing import Any

from .contracts import conversation_item, visible_unique

OMITTED = "- (earlier conversation omitted; complete records available through recall)"


def _render(speaker: str, content: str) -> str:
    escaped = content.replace('"', '\\"')
    return f'- {speaker}: "{escaped}"'


def conversation_transcript(task: Any, store: Any, *, max_chars: int = 9000) -> str:
    """Reserve current steering first, then up to three recent complete exchanges.

    Channel, date, project and session changes are not exchange boundaries. This
    is a continuity window, not semantic recall or permission to resume old work.
    Every source in the manifest was included in full. Omissions are explicit.
    """
    limit = max(0, int(max_chars))
    recent = getattr(store, "list_recent", None)
    events = []
    unavailable = False
    if callable(recent) and task.user:
        try:
            events = list(recent(owner_user_id=task.user, limit=30))
        except Exception:
            unavailable = True
    authorized = []
    event_by_id = {}
    for event in events:
        item = conversation_item(event, str(task.user or ""))
        if item is not None:
            authorized.append(item)
            event_by_id[item.source.source_id] = event
    items = visible_unique(authorized, str(task.user or ""))
    rows = []
    for item in items:
        event = event_by_id[item.source.source_id]
        role = str(getattr(event, "role", "")).strip().lower()
        content = str(getattr(event, "content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        label = role.capitalize()
        project = str(getattr(event, "project_id", "") or "").strip()
        if project:
            label += f" [{project}]"
        if item.source.provenance == "legacy_import":
            label += " [legacy ledger import; unverified provenance]"
        elif item.source.provenance != "recorded":
            label += " [historical record; unverified provenance]"
        rows.append({"text": f"- {label}: {content}", "role": role,
                     "content": content, "source": item.source.to_dict()})

    # Structured checkpoint records retain multiline messages as a single unit.
    # An old Markdown checkpoint is one opaque record: never split it on lines.
    records = getattr(task, "conversation_records", []) or []
    current = [{"text": _render(str(r.get("speaker") or "unknown"), str(r.get("content") or "")),
                "role": "current", "content": str(r.get("content") or ""), "source": None}
               for r in records if r.get("content")]
    markdown = str(getattr(task, "recent_conversation_md", "") or "").strip()
    structured = "\n".join(r["text"] for r in current)
    if markdown and markdown != "- (none)" and markdown != structured:
        legacy = markdown.removesuffix("\n" + structured).rstrip() if structured else markdown
        current.insert(0, {"text": legacy, "role": "current", "content": "", "source": None})
    # Only the newest canonical user turn can deduplicate an unlinked current
    # turn. Repeated historical 'yes' messages must remain distinct exchanges.
    if rows and current and rows[-1]["role"] == "user" and rows[-1]["content"] == current[-1]["content"]:
        current[-1] = rows.pop()
    task_id = str(getattr(task, "task_id", None) or getattr(task, "message_id", None) or "current")
    for index, record in enumerate(current):
        record["record_id"] = f"task-working:{task_id}:conversation:{index}"

    exchanges: list[list[dict]] = []
    for row in rows:
        if row["role"] == "user" or not exchanges:
            exchanges.append([])
        exchanges[-1].append(row)
    omitted = []
    for exchange in exchanges[:-3]:
        omitted.extend({"source_id": r["source"]["source_id"], "reason": "continuity_window"} for r in exchange)
    exchanges = exchanges[-3:]

    selected_current = []
    selected_history = []
    used = len(OMITTED) + 1  # Reserve an omission notice before admitting records.
    current_omitted = 0
    current_omitted_ids = []
    # Newest steering has precedence over earlier task messages.
    for record in reversed(current):
        cost = len(record["text"]) + 1
        if used + cost <= limit:
            selected_current.insert(0, record)
            used += cost
        else:
            current_omitted += 1
            current_omitted_ids.append(record["record_id"])
            if record["source"]:
                omitted.append({"source_id": record["source"]["source_id"], "reason": "current_budget"})
    blocked = False
    for exchange in reversed(exchanges):
        cost = sum(len(r["text"]) + 1 for r in exchange)
        if not blocked and used + cost <= limit:
            selected_history.insert(0, exchange)
            used += cost
        else:
            blocked = True  # Never skip a missing antecedent to show older work.
            omitted.extend({"source_id": r["source"]["source_id"], "reason": "exchange_budget"} for r in exchange)
    selected = [r for exchange in selected_history for r in exchange] + selected_current
    has_omissions = bool(omitted or current_omitted or len(events) >= 30)
    rendered = "\n".join(([OMITTED] if has_omissions and len(OMITTED) <= limit else []) + [r["text"] for r in selected])
    if isinstance(getattr(task, "metadata", None), dict):
        task.metadata["conversation_sources"] = [r["source"] for r in selected if r["source"]]
        task.metadata["conversation_projection"] = {
            "schema_version": 1, "max_chars": limit, "omitted": omitted,
            "current_records_omitted": current_omitted, "requires_recall": has_omissions,
            "omitted_current_record_ids": current_omitted_ids,
            "included_current_record_ids": [r["record_id"] for r in selected_current],
            "history_unavailable": unavailable, "window_may_have_earlier_events": len(events) >= 30,
        }
    return rendered or ("- (none)" if limit >= len("- (none)") else "")
