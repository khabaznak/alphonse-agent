"""Bounded conversation context from recorded messages, separate from task audits."""

from __future__ import annotations

from typing import Any

from .contracts import conversation_item, visible_unique


def conversation_transcript(task: Any, store: Any, *, max_chars: int = 9000) -> str:
    """Preserve newest recorded turns and the current task's actual messages.

    Legacy ledger imports have weaker provenance than channel messages. Label
    them explicitly rather than silently presenting them as direct user input.
    Task plans, acceptance criteria and tool results never enter this projection.
    """
    recent = getattr(store, "list_recent", None)
    events = []
    if callable(recent) and task.user:
        try:
            events = recent(owner_user_id=task.user, limit=30)
        except Exception:
            events = []
    rows: list[str] = []
    recorded_lines: set[str] = set()
    authorized = []
    event_by_id = {}
    for event in events:
        item = conversation_item(event, str(task.user or ""))
        if item is not None:
            authorized.append(item)
            event_by_id[item.source.source_id] = event
    items = visible_unique(authorized, str(task.user or ""))
    manifest = []
    for item in items:
        event = event_by_id[item.source.source_id]
        role = str(getattr(event, "role", "")).strip().lower()
        content = str(getattr(event, "content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        project = str(getattr(event, "project_id", "")).strip()
        label = role.capitalize()
        if project:
            label += f" [{project}]"
        if item.source.provenance == "legacy_import":
            label += " [legacy ledger import; unverified provenance]"
        elif item.source.provenance != "recorded":
            label += " [historical record; unverified provenance]"
        rows.append(f"- {label}: {content}")
        manifest.append(item.source.to_dict())
        # TaskState renders incoming messages using the account ID and quotes.
        if role == "user" and item.source.provenance == "recorded":
            escaped = content.replace('"', '\\"')
            recorded_lines.add(f'- {task.user}: "{escaped}"')
    current = str(getattr(task, "recent_conversation_md", "") or "").strip()
    if current and current != "- (none)":
        rows.extend(line for line in current.splitlines() if line not in recorded_lines)
    # Keep a source manifest for assembly/recall consumers. It describes selected
    # events; the final character budget may truncate their text.
    if isinstance(getattr(task, "metadata", None), dict):
        task.metadata["conversation_sources"] = manifest
    rendered = "\n".join(rows) or "- (none)"
    if len(rendered) > max_chars:
        marker = "- (earlier conversation omitted)\n"
        rendered = marker + rendered[-(max_chars - len(marker)):]
    return rendered
