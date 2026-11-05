"""Read-only home snapshots from authorized skill resources; never execute providers."""
from datetime import datetime, timezone
import json
import math
import re

from alphonse.agent_v2.artifact_resources import contained_path

FAMILIES = {"sensors", "emergency_contacts"}


def valid_binding(binding):
    return (isinstance(binding, dict) and set(binding) == {"type", "family", "artifact_id", "path"}
            and binding.get("type") == "home" and isinstance(binding.get("family"), str)
            and binding["family"] in FAMILIES and isinstance(binding.get("artifact_id"), str)
            and binding["artifact_id"].startswith("artifact.") and isinstance(binding.get("path"), str)
            and 0 < len(binding["path"]) <= 256)


def _text(value, limit):
    return isinstance(value, str) and 0 < len(value.strip()) <= limit


def _time(value):
    if not isinstance(value, str):
        raise ValueError("timestamp_required")
    result = datetime.fromisoformat(value)
    if not result.tzinfo:
        raise ValueError("timezone_required")
    return result


def snapshot(runtime, actor, skill_id, binding):
    try:
        resources = runtime.core.artifact_resources
        record = resources.store.get(binding["artifact_id"])
        if not record or record.skill_id != skill_id or not record.enabled or not resources.can_use(record, actor):
            return None
        path = contained_path(resources.storage_root(record), binding["path"])
        if not path.is_file() or path.stat().st_size > 64_000:
            return None
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or set(data) != {"version", "updated_at", "expires_at", "items"} or type(data["version"]) is not int or data["version"] != 1:
            return None
        updated, expires = _time(data["updated_at"]), _time(data["expires_at"])
        now = datetime.now(timezone.utc)
        if updated > now or not 0 < (expires - updated).total_seconds() <= 86400:
            return None
        family = binding["family"]
        # Expired sensor snapshots may supply last-known readings, always labeled stale.
        # Contact actions disappear on expiry; an outdated destination must not be offered.
        if family == "emergency_contacts" and expires <= now:
            return None
        if not isinstance(data["items"], list) or len(data["items"]) > 50:
            return None
        items, ids = [], set()
        for item in data["items"]:
            if not isinstance(item, dict) or not _text(item.get("id"), 128) or item["id"] in ids:
                return None
            ids.add(item["id"])
            if family == "sensors":
                if set(item) != {"id", "name", "source", "unit", "value", "observed_at", "stale_after_seconds", "state"}:
                    return None
                if not _text(item["name"], 200) or not _text(item["source"], 200) or not _text(item["unit"], 40):
                    return None
                if item["state"] not in ("connected", "stale", "disconnected", "error"):
                    return None
                if item["value"] is not None and (type(item["value"]) not in (int, float) or not math.isfinite(item["value"])):
                    return None
                if type(item["stale_after_seconds"]) is not int or not 1 <= item["stale_after_seconds"] <= 3600:
                    return None
                observed = _time(item["observed_at"]) if item["observed_at"] is not None else None
                if observed and observed > updated or item["value"] is not None and observed is None:
                    return None
                if item["state"] in ("connected", "stale") and (observed is None or item["value"] is None):
                    return None
                state = item["state"]
                if state == "connected" and (expires <= now or (now - observed).total_seconds() >= item["stale_after_seconds"]):
                    state = "stale"
                items.append({**item, "state": state})
            else:
                if set(item) != {"id", "name", "role", "phone", "email"} or not _text(item["name"], 200) or not _text(item["role"], 200):
                    return None
                if item["phone"] is not None and (not isinstance(item["phone"], str) or not re.fullmatch(r"\+?[0-9][0-9 ()-]{1,39}", item["phone"])):
                    return None
                if item["email"] is not None and (not isinstance(item["email"], str) or len(item["email"]) > 254 or not re.fullmatch(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+", item["email"])):
                    return None
                items.append(dict(item))
        return {"family": family, "artifactId": record.artifact_id, "owner": record.owner_user_id,
                "source": record.name, "updatedAt": data["updated_at"], "expiresAt": expires.timestamp() * 1000,
                "items": items}
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return None
