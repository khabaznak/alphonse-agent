from __future__ import annotations

from datetime import datetime
from typing import Any

from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.cognition.memory import TimeRange
from alphonse.agent.cognition.memory import resolve_memory_owner_id


def _state_user_id(state: dict[str, Any] | None, explicit: str | None = None) -> str:
    return resolve_memory_owner_id(state=state, explicit=explicit)


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
    canonical_name: str = "memory.search_episodes"
    capability: str = "memory"

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
        try:
            uid = _state_user_id(state, user_id)
        except ValueError as exc:
            return _failed(self.canonical_name, str(exc))
        service = MemoryService()
        time_range = None
        if str(start_time or "").strip() or str(end_time or "").strip():
            time_range = TimeRange(start=_parse_dt(start_time), end=_parse_dt(end_time))
        query_text = str(query or "")
        normalized_limit = max(1, int(limit or 100))
        rows = service.search_episodes(
            uid,
            query_text,
            mission_id=mission_id,
            time_range=time_range,
            limit=normalized_limit,
        )
        output_payload: dict[str, Any] = {
            "user_id": uid,
            "hits": rows,
            "count": len(rows),
        }
        return {
            "output": output_payload,
            "exception": None,
            "metadata": {"tool": "memory.search_episodes"},
        }


class GetMissionTool:
    canonical_name: str = "memory.get_mission"
    capability: str = "memory"

    def execute(
        self,
        *,
        mission_id: str,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            uid = _state_user_id(state, user_id)
        except ValueError as exc:
            return _failed(self.canonical_name, str(exc))
        service = MemoryService()
        mission = service.get_mission(uid, str(mission_id or ""))
        return {
            "output": {"user_id": uid, "mission": mission},
            "exception": None,
            "metadata": {"tool": "memory.get_mission"},
        }


class ListActiveMissionsTool:
    canonical_name: str = "memory.list_active_missions"
    capability: str = "memory"

    def execute(
        self,
        *,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            uid = _state_user_id(state, user_id)
        except ValueError as exc:
            return _failed(self.canonical_name, str(exc))
        service = MemoryService()
        rows = service.list_active_missions(uid)
        return {
            "output": {"user_id": uid, "missions": rows, "count": len(rows)},
            "exception": None,
            "metadata": {"tool": "memory.list_active_missions"},
        }


class GetWorkspacePointerTool:
    canonical_name: str = "memory.get_workspace"
    capability: str = "memory"

    def execute(
        self,
        *,
        key: str,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            uid = _state_user_id(state, user_id)
        except ValueError as exc:
            return _failed(self.canonical_name, str(exc))
        service = MemoryService()
        value = service.get_workspace_pointer(uid, str(key or ""))
        return {
            "output": {"user_id": uid, "key": str(key or ""), "value": value},
            "exception": None,
            "metadata": {"tool": "memory.get_workspace"},
        }


class UpsertOperationalFactTool:
    canonical_name: str = "memory.upsert_operational_fact"
    capability: str = "memory"

    def execute(
        self,
        *,
        key: str,
        title: str,
        fact_type: str,
        summary: str | None = None,
        content_json: Any = None,
        tags: list[str] | None = None,
        source: str | None = None,
        stability: str | None = None,
        importance: str | None = None,
        status: str | None = None,
        scope: str = "private",
        last_verified_at: str | None = None,
        confidence: float | None = None,
        created_by: str | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            owner = _state_user_id(state, created_by or user_id)
        except ValueError as exc:
            return _failed(self.canonical_name, str(exc))
        service = MemoryService()
        fact = service.upsert_operational_fact(
            created_by=owner,
            key=str(key or "").strip(),
            title=str(title or "").strip(),
            fact_type=str(fact_type or "").strip(),
            summary=summary,
            content_json=content_json,
            tags=tags,
            source=source,
            stability=stability,
            importance=importance,
            status=status,
            scope=str(scope or "private").strip().lower() or "private",
            last_verified_at=last_verified_at,
            confidence=confidence,
        )
        return {
            "output": {"created_by": owner, "fact": fact},
            "exception": None,
            "metadata": {"tool": "memory.upsert_operational_fact"},
        }


class SearchOperationalFactsTool:
    canonical_name: str = "memory.search_operational_facts"
    capability: str = "memory"

    def execute(
        self,
        *,
        query: str | None = None,
        fact_type: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
        stability: str | None = None,
        importance: str | None = None,
        scope: str | None = None,
        limit: int = 50,
        offset: int = 0,
        created_by: str | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            owner = _state_user_id(state, created_by or user_id)
        except ValueError as exc:
            return _failed(self.canonical_name, str(exc))
        service = MemoryService()
        normalized_limit = max(1, min(int(limit), 200))
        normalized_offset = max(0, int(offset))
        rows = service.search_operational_facts(
            created_by=owner,
            query=query,
            fact_type=fact_type,
            status=status,
            tags=tags,
            stability=stability,
            importance=importance,
            scope=str(scope or "").strip().lower() or None,
            limit=normalized_limit,
            offset=normalized_offset,
        )
        return {
            "output": {"created_by": owner, "facts": rows, "count": len(rows), "limit": normalized_limit, "offset": normalized_offset},
            "exception": None,
            "metadata": {"tool": "memory.search_operational_facts"},
        }


class RemoveOperationalFactTool:
    canonical_name: str = "memory.remove_operational_fact"
    capability: str = "memory"

    def execute(
        self,
        *,
        fact_id: str | None = None,
        key: str | None = None,
        created_by: str | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            owner = _state_user_id(state, created_by or user_id)
        except ValueError as exc:
            return _failed(self.canonical_name, str(exc))
        service = MemoryService()
        deleted = service.remove_operational_fact(
            created_by=owner,
            fact_id=str(fact_id or "").strip() or None,
            key=str(key or "").strip() or None,
        )
        return {
            "output": {"created_by": owner, "deleted": bool(deleted), "fact_id": str(fact_id or "").strip() or None, "key": str(key or "").strip() or None},
            "exception": None,
            "metadata": {"tool": "memory.remove_operational_fact"},
        }


def _failed(tool_name: str, code: str) -> dict[str, Any]:
    return {
        "output": None,
        "exception": {
            "code": code,
            "message": code.replace("_", " "),
            "retryable": False,
            "details": {},
        },
        "metadata": {"tool": tool_name},
    }
