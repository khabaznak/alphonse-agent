"""Read-only progress and clocks from authorized skill-owned source records."""
from datetime import datetime
import math

from alphonse.agent_v2.coordination_widgets import valid_binding as _binding, snapshot as _snapshot

FAMILIES = {"progress", "timer", "tracker"}
SCOPES = {"session", "project", "habit", "sample"}


def valid_binding(binding):
    return _binding(binding, binding_type="activity", families=FAMILIES)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 10**12


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp_required")
    parsed = datetime.fromisoformat(value)
    if not parsed.tzinfo:
        raise ValueError("timezone_required")
    return parsed


def validate_items(items, updated, family):
    ids = set()
    for item in items:
        if not isinstance(item, dict):
            return None
        common = {"id", "title", "scope", "recorded_at", "status"}
        fields = {"elapsed_ms", "duration_ms", "running_since"} if family == "timer" else {"value", "target", "unit", "display"}
        if set(item) != common | fields:
            return None
        if any(not isinstance(item.get(key), str) or not 0 < len(item[key]) <= limit for key, limit in (("id", 128), ("title", 300))):
            return None
        if item["id"] in ids or item["scope"] not in SCOPES or _timestamp(item["recorded_at"]) > updated:
            return None
        ids.add(item["id"])
        if family == "timer":
            if item["status"] not in {"running", "paused", "completed", "cancelled"}:
                return None
            if not _number(item["elapsed_ms"]) or not _number(item["duration_ms"]) or item["duration_ms"] <= 0:
                return None
            if item["status"] == "running":
                if _timestamp(item["running_since"]) > updated:
                    return None
            elif item["running_since"] is not None:
                return None
        else:
            if item["status"] not in {"active", "completed", "cancelled"} or item["display"] not in {"bar", "dots"}:
                return None
            if not _number(item["value"]) or not _number(item["target"]) or item["target"] <= 0:
                return None
            if not isinstance(item["unit"], str) or not 0 < len(item["unit"]) <= 40:
                return None
            if item["display"] == "dots" and (item["target"] > 100 or item["target"] % 1 or item["value"] % 1):
                return None
    return items


def snapshot(runtime, actor, skill_id, binding):
    return _snapshot(runtime, actor, skill_id, binding,
                     validate_items=lambda items, updated: validate_items(items, updated, binding["family"]))
