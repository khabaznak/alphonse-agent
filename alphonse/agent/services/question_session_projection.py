from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.preferences.store import get_user_preference
from alphonse.agent.session.day_state import build_next_session_state
from alphonse.agent.session.day_state import commit_session_state
from alphonse.agent.session.day_state import resolve_day_session
from alphonse.config import settings


def project_question_exchange(
    *,
    user_id: str,
    channel: str,
    user_message: str = "",
    assistant_message: str = "",
    task_record: dict[str, Any] | None = None,
) -> None:
    canonical_user_id = str(user_id or "").strip()
    if not canonical_user_id:
        return
    timezone_name = str(get_user_preference(canonical_user_id, "timezone") or "").strip() or settings.get_timezone()
    try:
        previous = resolve_day_session(
            user_id=canonical_user_id,
            channel=str(channel or "api").strip() or "api",
            timezone_name=timezone_name,
        )
        updated = build_next_session_state(
            previous=previous,
            channel=str(channel or "api").strip() or "api",
            user_message=str(user_message or ""),
            assistant_message=str(assistant_message or ""),
            task_record=task_record,
        )
        commit_session_state(updated)
    except Exception:
        return
