"""Bounded, read-only skill snapshots. Never invoke a program while opening UI."""
from datetime import datetime, timezone
import json

from alphonse.agent_v2.artifact_resources import contained_path

FAMILIES = {"todos", "shopping", "reminders", "appointments"}


def valid_binding(binding, *, binding_type="coordination", families=FAMILIES):
    return (isinstance(binding, dict) and set(binding) == {"type", "family", "artifact_id", "path"}
            and binding.get("type") == binding_type and isinstance(binding.get("family"), str)
            and binding["family"] in families and isinstance(binding.get("artifact_id"), str)
            and binding["artifact_id"].startswith("artifact.") and isinstance(binding.get("path"), str)
            and 0 < len(binding["path"]) <= 256)


def snapshot(runtime, actor, skill_id, binding, *, validate_items=None):
    """Catalog access is the boundary; selection and snapshot fields grant no access."""
    try:
        resources = runtime.core.artifact_resources
        record = resources.store.get(binding["artifact_id"])
        if not record or record.skill_id != skill_id or not record.enabled or not resources.can_use(record, actor):
            return None
        path = contained_path(resources.storage_root(record), binding["path"])
        if not path.is_file() or path.stat().st_size > 64_000:
            return None
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or set(value) != {"version", "updated_at", "expires_at", "items"} or type(value["version"]) is not int or value["version"] != 1:
            return None
        updated, expires = (datetime.fromisoformat(value[key]) for key in ("updated_at", "expires_at"))
        now = datetime.now(timezone.utc)
        if not updated.tzinfo or not expires.tzinfo or not updated <= now < expires or (expires - updated).total_seconds() > 86400:
            return None
        if not isinstance(value["items"], list) or len(value["items"]) > 50:
            return None
        if validate_items is not None:
            items = validate_items(value["items"], updated)
            if items is None:
                return None
        else:
            items = _list_items(value["items"])
            if items is None:
                return None
        return {"family": binding["family"], "artifactId": record.artifact_id,
                "owner": record.owner_user_id, "source": record.name, "updatedAt": value["updated_at"],
                "expiresAt": expires.timestamp() * 1000, "items": items}
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return None


def _list_items(values):
    items, ids = [], set()
    for item in values:
        if not isinstance(item, dict) or set(item) - {"id", "title", "status", "when"} or not {"id", "title", "status"} <= set(item):
            return None
        if any(not isinstance(item.get(key), str) or not 0 < len(item[key]) <= limit for key, limit in (("id", 128), ("title", 300), ("status", 32))):
            return None
        if item["id"] in ids or item["status"] not in {"pending", "completed", "cancelled"}:
            return None
        if "when" in item:
            when = datetime.fromisoformat(item["when"])
            if not when.tzinfo:
                return None
        ids.add(item["id"])
        items.append(dict(item))
    return items
