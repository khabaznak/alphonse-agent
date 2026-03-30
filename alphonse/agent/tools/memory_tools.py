from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from alphonse.agent.cognition.memory import MemoryService
from alphonse.agent.cognition.memory import TimeRange
from alphonse.agent.cognition.memory import resolve_memory_owner_aliases
from alphonse.agent.cognition.memory import resolve_memory_owner_id


_CLIENT_STORE_QUERY_RE = re.compile(r"\b(client|prospect|lead|crm)\b", re.IGNORECASE)
_CLIENT_STORE_VERIFY_RE = re.compile(
    r"\b(stored|save|saved|remember|remembered|recorded|write|wrote|crm)\b",
    re.IGNORECASE,
)


def _state_user_id(state: dict[str, Any] | None, explicit: str | None = None) -> str:
    return resolve_memory_owner_id(state=state, explicit=explicit)


def _state_user_aliases(state: dict[str, Any] | None, explicit: str | None = None) -> list[str]:
    return resolve_memory_owner_aliases(state=state, explicit=explicit)


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
        aliases = _state_user_aliases(state, user_id)
        uid = aliases[0] if aliases else _state_user_id(state, user_id)
        service = MemoryService()
        time_range = None
        if str(start_time or "").strip() or str(end_time or "").strip():
            time_range = TimeRange(start=_parse_dt(start_time), end=_parse_dt(end_time))
        query_text = str(query or "")
        normalized_limit = max(1, int(limit or 100))
        crm_facts: list[dict[str, Any]] = []
        crm_status: str | None = None
        if _is_client_storage_query(query_text):
            crm_facts = service.search_operational_facts(
                created_by=uid,
                query=query_text,
                tags=["crm", "client"],
                limit=min(25, normalized_limit),
            )
            if not crm_facts:
                crm_facts = service.search_operational_facts(
                    created_by=uid,
                    query=None,
                    tags=["crm", "client"],
                    limit=min(25, normalized_limit),
                )
            crm_status = "found" if crm_facts else "not_stored_yet"
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for alias in aliases:
            hits = service.search_episodes(
                alias,
                query_text,
                mission_id=mission_id,
                time_range=time_range,
                limit=normalized_limit,
            )
            for hit in hits:
                path = str(hit.get("path") or "")
                line_no = int(hit.get("line") or 0)
                key = (path, line_no)
                if not path or line_no <= 0 or key in seen:
                    continue
                seen.add(key)
                rows.append(hit)
                if len(rows) >= normalized_limit:
                    break
            if len(rows) >= normalized_limit:
                break
        output_payload: dict[str, Any] = {
            "user_id": uid,
            "owner_aliases": aliases,
            "hits": rows,
            "count": len(rows),
        }
        if _is_client_storage_query(query_text):
            output_payload["crm_facts"] = crm_facts
            output_payload["crm_count"] = len(crm_facts)
            output_payload["crm_status"] = crm_status
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
        aliases = _state_user_aliases(state, user_id)
        uid = aliases[0] if aliases else _state_user_id(state, user_id)
        service = MemoryService()
        mission = None
        for alias in aliases:
            mission = service.get_mission(alias, str(mission_id or ""))
            if isinstance(mission, dict):
                break
        return {
            "output": {"user_id": uid, "owner_aliases": aliases, "mission": mission},
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
        aliases = _state_user_aliases(state, user_id)
        uid = aliases[0] if aliases else _state_user_id(state, user_id)
        service = MemoryService()
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for alias in aliases:
            current = service.list_active_missions(alias)
            for mission in current:
                if not isinstance(mission, dict):
                    continue
                mission_key = str(mission.get("mission_id") or "").strip()
                if mission_key and mission_key in seen:
                    continue
                if mission_key:
                    seen.add(mission_key)
                rows.append(mission)
        return {
            "output": {"user_id": uid, "owner_aliases": aliases, "missions": rows, "count": len(rows)},
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
        aliases = _state_user_aliases(state, user_id)
        uid = aliases[0] if aliases else _state_user_id(state, user_id)
        service = MemoryService()
        value = None
        for alias in aliases:
            value = service.get_workspace_pointer(alias, str(key or ""))
            if value is not None:
                break
        return {
            "output": {"user_id": uid, "owner_aliases": aliases, "key": str(key or ""), "value": value},
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
        owner = _state_user_id(state, created_by or user_id)
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
        aliases = _state_user_aliases(state, created_by or user_id)
        owner = aliases[0] if aliases else _state_user_id(state, created_by or user_id)
        service = MemoryService()
        normalized_limit = max(1, min(int(limit), 200))
        normalized_offset = max(0, int(offset))
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for alias in aliases:
            current = service.search_operational_facts(
                created_by=alias,
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
            for item in current:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("fact_id") or item.get("key") or "").strip()
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                rows.append(item)
                if len(rows) >= normalized_limit:
                    break
            if len(rows) >= normalized_limit:
                break
        return {
            "output": {"created_by": owner, "owner_aliases": aliases, "facts": rows, "count": len(rows), "limit": normalized_limit, "offset": normalized_offset},
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
        owner = _state_user_id(state, created_by or user_id)
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


def _is_client_storage_query(query: str) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    return bool(_CLIENT_STORE_QUERY_RE.search(text) and _CLIENT_STORE_VERIFY_RE.search(text))
