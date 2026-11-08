"""Privacy-preserving browser workflow checkpoint records.

This is not an authorization mechanism and does not make a browser available.
Only runtime-owned enums and opaque identifiers may enter a workflow record.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from typing import Any
from uuid import uuid4


KEY = "v3_browser_workflow"
VERSION = 1
FIELDS = frozenset({"identifier", "vehicle_plate", "vehicle_serial", "name", "number", "other"})
OPERATIONS = frozenset({"open", "fill", "click", "submit"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_UUID_HEX = re.compile(r"^[0-9a-f]{32}$")
_PRIVATE_REF = re.compile(r"^[A-Za-z0-9_-]{32}$")
_QUESTION_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class BrowserWorkflowError(RuntimeError):
    pass


def _id(value: Any) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))


def _time() -> str:
    return datetime.now(timezone.utc).isoformat()


def _legacy_browser_state(task: Any) -> bool:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if any(key in metadata for key in ("browser_handoff", "browser_value_refs", "browser_private_required",
                                       "browser_sensitive_context", "browser_cleanup_attempted")):
        return True
    try:
        plan = json.loads(task.plan_json) if task.plan_json else None
    except (TypeError, ValueError):
        plan = None
    stack = [task.hierarchical_state, metadata.get("v3_phase_history"), plan]
    inspected = 0
    while stack:
        item = stack.pop()
        inspected += 1
        if inspected > 20000:
            return True
        if isinstance(item, dict):
            tool = item.get("tool_id")
            if isinstance(tool, str) and tool.startswith("native.browser_") and tool != "native.browser_workflow_begin":
                return True
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def _validate(task: Any, state: Any) -> dict[str, Any]:
    """Reject unknown fields and versions rather than accepting arbitrary page data."""
    if not isinstance(state, dict) or set(state) != {"version", "actor", "task", "session", "events"}:
        raise BrowserWorkflowError("browser_workflow_invalid")
    if state["version"] != VERSION or state["actor"] != task.user or state["task"] != task.task_id:
        raise BrowserWorkflowError("browser_workflow_scope_or_version_invalid")
    if not _id(state["actor"]) or not _id(state["task"]) or not isinstance(state["session"], str) or not _UUID_HEX.fullmatch(state["session"]):
        raise BrowserWorkflowError("browser_workflow_invalid")
    events = state["events"]
    if not isinstance(events, list) or len(events) > 256:
        raise BrowserWorkflowError("browser_workflow_invalid")
    seen: dict[str, dict[str, Any]] = {}
    observations: dict[tuple[str, str], tuple[int, str]] = {}
    positions: dict[str, int] = {}
    resolved: set[str] = set()
    for position, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {"id", "kind", "field", "phase", "session", "at", "ref", "related", "source"}:
            raise BrowserWorkflowError("browser_workflow_invalid")
        if (not isinstance(event["id"], str) or not _UUID_HEX.fullmatch(event["id"])
                or not isinstance(event["session"], str) or not _UUID_HEX.fullmatch(event["session"]) or not _id(event["phase"])
                or event["kind"] not in {"requested", "supplied", "observed", "attempted", "returned", "verified"}
                or event["field"] not in FIELDS | OPERATIONS | {"form", "result", "none"}
                or (event["ref"] is not None and (not isinstance(event["ref"], str) or not _PRIVATE_REF.fullmatch(event["ref"])))
                or (event["related"] is not None and (not isinstance(event["related"], str) or not _UUID_HEX.fullmatch(event["related"])))
                or (event["source"] is not None and (not isinstance(event["source"], str) or not _QUESTION_ID.fullmatch(event["source"])))):
            raise BrowserWorkflowError("browser_workflow_invalid")
        try:
            stamp = datetime.fromisoformat(event["at"])
            if stamp.tzinfo is None:
                raise ValueError
        except (TypeError, ValueError):
            raise BrowserWorkflowError("browser_workflow_invalid") from None
        if event["id"] in seen:
            raise BrowserWorkflowError("browser_workflow_invalid")
        kind, ref, related = event["kind"], event["ref"], event["related"]
        if kind in {"supplied", "observed", "attempted"} and (ref is None or related is not None):
            raise BrowserWorkflowError("browser_workflow_invalid")
        if ((kind == "supplied" and event["field"] not in FIELDS)
                or (kind == "observed" and event["field"] not in {"form", "result"})
                or (kind == "attempted" and event["field"] not in {"fill", "submit"})):
            raise BrowserWorkflowError("browser_workflow_invalid")
        if kind == "requested" and (ref is not None or related is not None or event["field"] not in FIELDS or event["source"] is None):
            raise BrowserWorkflowError("browser_workflow_invalid")
        if kind not in {"requested", "supplied"} and event["source"] is not None:
            raise BrowserWorkflowError("browser_workflow_invalid")
        if kind in {"returned", "verified"}:
            parent = seen.get(related)
            if not parent or parent["kind"] != "attempted" or parent["session"] != event["session"]:
                raise BrowserWorkflowError("browser_workflow_invalid")
            if kind == "returned" and (ref is not None or event["field"] != parent["field"]):
                raise BrowserWorkflowError("browser_workflow_invalid")
            if kind != "returned":
                observation = observations.get((event["session"], ref))
                required_kind = "result" if parent["field"] == "submit" else "form"
                if (related in resolved or ref is None or observation is None or
                        observation[0] <= positions[related] or observation[1] != required_kind or
                        (parent["field"] == "fill" and event["field"] not in FIELDS) or
                        (parent["field"] == "submit" and event["field"] != "submit")):
                    raise BrowserWorkflowError("browser_workflow_invalid")
                resolved.add(related)
        if kind == "observed":
            observations[(event["session"], ref)] = (position, event["field"])
        seen[event["id"]] = event
        positions[event["id"]] = position
    return state


def load(task: Any) -> dict[str, Any]:
    value = task.metadata.get(KEY)
    if value is None:
        # An older browser marker cannot be safely interpreted as a new workflow.
        if _legacy_browser_state(task):
            raise BrowserWorkflowError("browser_workflow_legacy_reconciliation_required")
        raise BrowserWorkflowError("browser_workflow_missing")
    return deepcopy(_validate(task, value))


def create(task: Any) -> dict[str, Any]:
    if KEY in task.metadata or _legacy_browser_state(task):
        raise BrowserWorkflowError("browser_workflow_reconciliation_required")
    state = {"version": VERSION, "actor": task.user, "task": task.task_id,
             "session": uuid4().hex, "events": []}
    _validate(task, state)
    task.metadata[KEY] = state
    return deepcopy(state)


def rotate_session(task: Any) -> dict[str, Any]:
    """Fence prior process-local handles while retaining their historical facts."""
    state = load(task)
    state["session"] = uuid4().hex
    _validate(task, state)
    task.metadata[KEY] = state
    return deepcopy(state)


def append(task: Any, *, kind: str, field: str, phase: str,
           ref: str | None = None, related: str | None = None, source: str | None = None) -> str:
    state = load(task)
    event_id = uuid4().hex
    state["events"].append({"id": event_id, "kind": kind, "field": field, "phase": phase, "session": state["session"],
                            "at": _time(), "ref": ref, "related": related, "source": source})
    _validate(task, state)
    task.metadata[KEY] = state
    return event_id


def unresolved(state: dict[str, Any]) -> bool:
    """Returned calls remain uncertain until independently reconciled."""
    attempts = {e["id"] for e in state["events"] if e["kind"] == "attempted"}
    resolved = {e["related"] for e in state["events"] if e["kind"] == "verified"}
    return bool(attempts - resolved)


def completion_blocked(task: Any, service: Any = None) -> bool:
    if task.metadata.get("v3_browser_input_reacquisition_required"):
        return True
    if KEY not in task.metadata:
        return _legacy_browser_state(task)
    try:
        state = load(task)
        if unresolved(state) or service is None:
            return True
        # This foundation verifies that a submission occurred, not what the
        # resulting page means. End-to-end result qualification belongs to the
        # browser result integration and cannot be inferred from a click/receipt.
        if any(event["kind"] == "attempted" and event["field"] == "submit" for event in state["events"]):
            return True
        view = service.projection(task, limit=256)
        if not view.get("session_available"):
            return True
        return any(event["kind"] == "verified" and not event["reference_available"] for event in view["events"])
    except Exception:
        return True


def projection(task: Any, *, available: Any = None, session_available: bool = False, limit: int = 64) -> dict[str, Any]:
    """A bounded planner view; references and page content never leave the store."""
    state = load(task)
    entries = []
    limit = max(1, min(256, int(limit)))
    for event in state["events"][-limit:]:
        live = False
        if event["ref"] and callable(available):
            try:
                live = bool(available(state, event))
            except Exception:
                live = False
        entries.append({"kind": event["kind"], "field": event["field"], "phase": event["phase"],
                        "at": event["at"], "current_session": event["session"] == state["session"],
                        "reference_available": live})
    return {"version": VERSION, "session_available": session_available, "events": entries, "older_events_omitted": len(state["events"]) > limit,
            "unresolved_effect": unresolved(state),
            "guidance": "Historical facts do not grant access or prove current page state; unavailable references require reacquisition."}
