"""Fixed health projection contract; the protected Doctor data adapter belongs to #6.

Only a harness-installed adapter may implement read_for_member. Skill resources
cannot configure executable code, database paths, permissions or another member.
No cache and no writes: each poll must repeat the adapter's authorized query.
"""
from datetime import datetime, timezone
from typing import Any, Protocol


class CareRecords(Protocol):
    """Trusted native adapter; enforce current access before returning any record.

    Query only actor-owned records; return no other subjects. Source receipt
    writeState=verified requires persisted-effect evidence from the write workflow.
    No script from an installed skill may serve as this adapter.
    """

    def read_for_member(self, *, actor_user_id: str, skill_id: str) -> dict[str, Any] | None: ...


def valid_binding(binding):
    return binding == {"type": "care"}


def snapshot(runtime, actor, skill_id):
    adapter = getattr(runtime, "care_records", None)
    if adapter is None:
        return None
    try:
        # This is a trusted data-access interface, never a skill script/tool.
        data = adapter.read_for_member(actor_user_id=actor, skill_id=skill_id)
        if not isinstance(data, dict) or set(data) != {"member", "source", "recordedAt", "expiresAt", "items"} or data["member"] != actor:
            return None
        if not isinstance(data["source"], str) or not 0 < len(data["source"]) <= 200:
            return None
        recorded = datetime.fromisoformat(data["recordedAt"])
        expires = datetime.fromisoformat(data["expiresAt"])
        if not recorded.tzinfo or not expires.tzinfo or not recorded <= datetime.now(timezone.utc) < expires or (expires - recorded).total_seconds() > 3600:
            return None
        if not isinstance(data["items"], list) or len(data["items"]) > 50:
            return None
        items, ids = [], set()
        for item in data["items"]:
            if not isinstance(item, dict) or set(item) != {"id", "definition", "dueAt", "adherence", "recordedAt", "writeState"}:
                return None
            if any(not isinstance(item[key], str) or not 0 < len(item[key]) <= limit for key, limit in (("id", 128), ("definition", 500))):
                return None
            if item["id"] in ids or item["adherence"] not in ("recorded", "missed", "unknown") or item["writeState"] not in ("none", "verified", "failed", "unconfirmed"):
                return None
            for key in ("dueAt", "recordedAt"):
                if item[key] is not None and (not isinstance(item[key], str) or not datetime.fromisoformat(item[key]).tzinfo):
                    return None
            if item["adherence"] != "unknown" and item["recordedAt"] is None:
                return None
            ids.add(item["id"])
            items.append(dict(item))
        return {**data, "items": items, "expiresAt": expires.timestamp() * 1000}
    except Exception:
        # Unconfigured, denied and failed reads all discard previous health data.
        return None
