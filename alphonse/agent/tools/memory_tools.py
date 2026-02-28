from __future__ import annotations

from datetime import datetime
from typing import Any

from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.cognition.memory import TimeRange


def _state_user_id(state: dict[str, Any] | None, explicit: str | None = None) -> str:
    candidate = str(explicit or "").strip()
    if candidate:
        return candidate
    payload = dict(state or {})
    for key in ("incoming_user_id", "actor_person_id", "channel_target", "chat_id", "conversation_key"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return "anonymous"


def _parse_dt(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt


class SearchEpisodesTool:
    def execute(
        self,
        *,
        query: str,
        user_id: str | None = None,
        mission_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        uid = _state_user_id(state, user_id)
        service = MemoryService()
        time_range = None
        if str(start_time or "").strip() or str(end_time or "").strip():
            time_range = TimeRange(start=_parse_dt(start_time), end=_parse_dt(end_time))
        rows = service.search_episodes(
            uid,
            str(query or ""),
            mission_id=mission_id,
            time_range=time_range,
            limit=max(1, int(limit or 100)),
        )
        return {
            "status": "ok",
            "result": {"user_id": uid, "hits": rows, "count": len(rows)},
            "error": None,
            "metadata": {"tool": "search_episodes"},
        }


class GetMissionTool:
    def execute(
        self,
        *,
        mission_id: str,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        uid = _state_user_id(state, user_id)
        service = MemoryService()
        mission = service.get_mission(uid, str(mission_id or ""))
        return {
            "status": "ok",
            "result": {"user_id": uid, "mission": mission},
            "error": None,
            "metadata": {"tool": "get_mission"},
        }


class ListActiveMissionsTool:
    def execute(
        self,
        *,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        uid = _state_user_id(state, user_id)
        service = MemoryService()
        rows = service.list_active_missions(uid)
        return {
            "status": "ok",
            "result": {"user_id": uid, "missions": rows, "count": len(rows)},
            "error": None,
            "metadata": {"tool": "list_active_missions"},
        }


class GetWorkspacePointerTool:
    def execute(
        self,
        *,
        key: str,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        uid = _state_user_id(state, user_id)
        service = MemoryService()
        value = service.get_workspace_pointer(uid, str(key or ""))
        return {
            "status": "ok",
            "result": {"user_id": uid, "key": str(key or ""), "value": value},
            "error": None,
            "metadata": {"tool": "get_workspace_pointer"},
        }
