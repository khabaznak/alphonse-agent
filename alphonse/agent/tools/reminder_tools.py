from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


@dataclass(frozen=True)
class ReminderListTool:
    canonical_name: str = "reminders.list"
    capability: str = "reminders"

    def execute(
        self,
        *,
        owner_id: str | None = None,
        channel: str | None = None,
        target: str | None = None,
        text: str | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
        status: str | None = "pending",
        limit: int = 25,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        filters = _filters_with_state_defaults(
            owner_id=owner_id,
            channel=channel,
            target=target,
            state=state,
        )
        try:
            reminders = _list_reminders(
                owner_id=filters["owner_id"],
                channel=filters["channel"],
                target=filters["target"],
                text=text,
                start_at=start_at,
                end_at=end_at,
                status=status,
                limit=limit,
            )
            return _ok(result={"reminders": reminders}, tool=self.canonical_name)
        except Exception as exc:
            return _failed(code=_err_code(exc), message=str(exc), tool=self.canonical_name)


@dataclass(frozen=True)
class ReminderCancelTool:
    canonical_name: str = "reminders.cancel"
    capability: str = "reminders"

    def execute(
        self,
        *,
        reminder_id: str | None = None,
        owner_id: str | None = None,
        channel: str | None = None,
        target: str | None = None,
        text: str | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        filters = _filters_with_state_defaults(
            owner_id=owner_id,
            channel=channel,
            target=target,
            state=state,
        )
        try:
            resolved_id = str(reminder_id or "").strip()
            matches: list[dict[str, Any]] = []
            if not resolved_id:
                matches = _list_reminders(
                    owner_id=filters["owner_id"],
                    channel=filters["channel"],
                    target=filters["target"],
                    text=text,
                    start_at=start_at,
                    end_at=end_at,
                    status="pending",
                    limit=10,
                )
                if len(matches) == 0:
                    return _failed(
                        code="reminder_not_found",
                        message="No pending reminder matched the provided cancellation filters.",
                        tool=self.canonical_name,
                    )
                if len(matches) > 1:
                    return _failed(
                        code="reminder_match_ambiguous",
                        message="More than one pending reminder matched; provide reminder_id or narrower filters.",
                        tool=self.canonical_name,
                    )
                resolved_id = str(matches[0].get("reminder_id") or "").strip()
            return _cancel_reminder(reminder_id=resolved_id, matched=matches[0] if matches else None)
        except Exception as exc:
            return _failed(code=_err_code(exc), message=str(exc), tool=self.canonical_name)


def _list_reminders(
    *,
    owner_id: str | None,
    channel: str | None,
    target: str | None,
    text: str | None,
    start_at: str | None,
    end_at: str | None,
    status: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows = _fetch_reminder_rows(status=status, limit=max(int(limit or 25), 1))
    start_dt = _parse_optional_dt(start_at)
    end_dt = _parse_optional_dt(end_at)
    owner_filter = _normalized(owner_id)
    channel_filter = _normalized(channel)
    target_filter = _normalized(target)
    text_filter = _normalized(text)
    out: list[dict[str, Any]] = []
    for row in rows:
        trigger_at = _parse_optional_dt(row.get("trigger_at"))
        if start_dt is not None and trigger_at is not None and trigger_at < start_dt:
            continue
        if end_dt is not None and trigger_at is not None and trigger_at > end_dt:
            continue
        if owner_filter and owner_filter not in _owner_values(row):
            continue
        if channel_filter and channel_filter not in _channel_values(row):
            continue
        if target_filter and target_filter not in _target_values(row):
            continue
        if text_filter and text_filter not in _reminder_text(row):
            continue
        out.append(_render_reminder(row))
        if len(out) >= max(int(limit or 25), 1):
            break
    return out


def _filters_with_state_defaults(
    *,
    owner_id: str | None,
    channel: str | None,
    target: str | None,
    state: dict[str, Any] | None,
) -> dict[str, str | None]:
    state_payload = state if isinstance(state, dict) else {}
    resolved_owner = str(
        owner_id
        or state_payload.get("actor_person_id")
        or state_payload.get("user_id")
        or state_payload.get("incoming_user_id")
        or ""
    ).strip() or None
    resolved_channel = str(channel or state_payload.get("channel_type") or state_payload.get("channel") or "").strip() or None
    resolved_target = str(target or state_payload.get("channel_target") or state_payload.get("target") or "").strip() or None
    return {"owner_id": resolved_owner, "channel": resolved_channel, "target": resolved_target}


def _fetch_reminder_rows(*, status: str | None, limit: int) -> list[dict[str, Any]]:
    query = """
        SELECT id, trigger_at, timezone, status, fired_at, signal_type, payload, target, origin, correlation_id, created_at, updated_at
        FROM timed_signals
    """
    params: list[Any] = []
    rendered_status = str(status or "").strip()
    if rendered_status:
        query += " WHERE status = ?"
        params.append(rendered_status)
    query += " ORDER BY trigger_at ASC LIMIT ?"
    params.append(max(limit * 5, limit))
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, params).fetchall()
    reminders: list[dict[str, Any]] = []
    for row in rows:
        payload = _parse_payload(row[6])
        if _normalized(payload.get("kind")) != "reminder":
            continue
        reminders.append(
            {
                "id": row[0],
                "trigger_at": row[1],
                "timezone": row[2],
                "status": row[3],
                "fired_at": row[4],
                "signal_type": row[5],
                "payload": payload,
                "target": row[7],
                "origin": row[8],
                "correlation_id": row[9],
                "created_at": row[10],
                "updated_at": row[11],
            }
        )
    return reminders


def _cancel_reminder(*, reminder_id: str, matched: dict[str, Any] | None) -> dict[str, Any]:
    if not reminder_id:
        return _failed(
            code="missing_reminder_id",
            message="reminder_id is required unless filters match exactly one pending reminder.",
            tool="reminders.cancel",
        )
    row = _fetch_reminder_by_id(reminder_id)
    if row is None:
        return _failed(code="reminder_not_found", message="No reminder exists with that id.", tool="reminders.cancel")
    if _normalized(row.get("kind")) != "reminder":
        return _failed(
            code="not_a_reminder",
            message="The requested timed signal is not a one-shot reminder.",
            tool="reminders.cancel",
        )
    status = _normalized(row.get("status"))
    if status == "fired":
        return _failed(
            code="reminder_already_fired",
            message="This reminder has already fired and cannot be cancelled.",
            tool="reminders.cancel",
        )
    if status != "pending":
        return _failed(
            code="reminder_not_pending",
            message=f"This reminder is {status or 'not pending'} and cannot be cancelled.",
            tool="reminders.cancel",
        )
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            """
            UPDATE timed_signals
            SET status = 'cancelled', updated_at = datetime('now')
            WHERE id = ? AND status = 'pending'
            """,
            (reminder_id,),
        )
        conn.commit()
    if cur.rowcount != 1:
        return _failed(
            code="reminder_cancel_race",
            message="The reminder was no longer pending by the time cancellation ran.",
            tool="reminders.cancel",
        )
    rendered = matched or _render_reminder(row)
    return _ok(
        result={
            "reminder_id": reminder_id,
            "cancelled": True,
            "previous_status": "pending",
            "trigger_at": rendered.get("trigger_at"),
            "reminder_text_raw": rendered.get("reminder_text_raw"),
        },
        tool="reminders.cancel",
    )


def _fetch_reminder_by_id(reminder_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT id, trigger_at, timezone, status, fired_at, signal_type, payload, target, origin, correlation_id, created_at, updated_at
            FROM timed_signals
            WHERE id = ?
            """,
            (reminder_id,),
        ).fetchone()
    if row is None:
        return None
    payload = _parse_payload(row[6])
    return {
        "id": row[0],
        "trigger_at": row[1],
        "timezone": row[2],
        "status": row[3],
        "fired_at": row[4],
        "signal_type": row[5],
        "payload": payload,
        "target": row[7],
        "origin": row[8],
        "correlation_id": row[9],
        "created_at": row[10],
        "updated_at": row[11],
        "kind": payload.get("kind"),
    }


def _render_reminder(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    return {
        "reminder_id": row.get("id"),
        "trigger_at": row.get("trigger_at"),
        "timezone": row.get("timezone"),
        "status": row.get("status"),
        "fired_at": row.get("fired_at"),
        "origin": row.get("origin"),
        "target": row.get("target"),
        "correlation_id": row.get("correlation_id"),
        "user_id": payload.get("user_id"),
        "requested_by": payload.get("requested_by"),
        "origin_channel": payload.get("origin_channel") or payload.get("service_key"),
        "delivery_target": payload.get("delivery_target"),
        "reminder_text_raw": payload.get("reminder_text_raw") or payload.get("source_instruction") or "",
        "source_instruction": payload.get("source_instruction") or payload.get("reminder_text_raw") or "",
    }


def _owner_values(row: dict[str, Any]) -> set[str]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    return {
        _normalized(row.get("target")),
        _normalized(payload.get("user_id")),
        _normalized(payload.get("requested_by")),
        _normalized(payload.get("provider_user_id_from")),
        _normalized(payload.get("delivery_target")),
    } - {""}


def _channel_values(row: dict[str, Any]) -> set[str]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    return {
        _normalized(row.get("origin")),
        _normalized(payload.get("origin_channel")),
        _normalized(payload.get("service_key")),
    } - {""}


def _target_values(row: dict[str, Any]) -> set[str]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    return {
        _normalized(row.get("target")),
        _normalized(payload.get("delivery_target")),
        _normalized(payload.get("provider_user_id_from")),
    } - {""}


def _reminder_text(row: dict[str, Any]) -> str:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    return _normalized(
        " ".join(
            [
                str(payload.get("reminder_text_raw") or ""),
                str(payload.get("source_instruction") or ""),
                str(payload.get("prompt_text") or ""),
                str(payload.get("prompt") or ""),
            ]
        )
    )


def _parse_payload(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _parse_optional_dt(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _normalized(value: object) -> str:
    return str(value or "").strip().lower()


def _err_code(exc: Exception) -> str:
    text = str(exc or "").strip()
    if not text:
        return "reminder_tool_failed"
    return text.split(":", 1)[0]


def _ok(*, result: dict[str, Any], tool: str) -> dict[str, Any]:
    return {"output": result, "exception": None, "metadata": {"tool": tool}}


def _failed(*, code: str, message: str, tool: str) -> dict[str, Any]:
    return {"output": None, "exception": {"code": code, "message": message}, "metadata": {"tool": tool}}
