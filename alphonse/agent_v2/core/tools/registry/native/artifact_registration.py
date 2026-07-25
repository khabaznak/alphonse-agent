"""Native registration tool for project-local executable artifacts."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Callable
from jsonschema import Draft202012Validator
from alphonse.agent_v2.artifacts import DEFAULT_TIMEOUT_SECONDS, SQLiteArtifactStore
from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

ARTIFACT_REGISTRATION_TOOL_ID = "native.artifact_registration"
ARTIFACT_REGISTRATION_TOOL_NAME = "artifact_registration"
ARTIFACT_REGISTRATION_ARGUMENT_SCHEMA: dict[str, Any] = {"type":"object","additionalProperties":False,"properties":{"name":{"type":"string"},"description":{"type":"string"},"entrypoint_path":{"type":"string"},"argument_schema":{"type":"object"},"artifact_id":{"type":"string"},"timeout_seconds":{"type":"number"}},"required":["name","description","entrypoint_path","argument_schema"]}

def build_artifact_registration_tool_definition(store: SQLiteArtifactStore, on_changed: Callable[[], None] | None = None) -> ToolDefinition:
    descriptor = ToolDescriptor(ARTIFACT_REGISTRATION_TOOL_ID, ARTIFACT_REGISTRATION_TOOL_NAME, ToolKind.NATIVE, "Register an executable already created inside the active owned project as an enabled reusable artifact tool.", dict(ARTIFACT_REGISTRATION_ARGUMENT_SCHEMA), ("artifact_registration", "local_execution"), ("native", "artifacts"))
    return ToolDefinition(descriptor, lambda arguments, context=None: execute_artifact_registration(arguments, context=context, store=store, on_changed=on_changed), dict(ARTIFACT_REGISTRATION_ARGUMENT_SCHEMA), accepts_context=True)

def execute_artifact_registration(arguments: dict[str, Any], *, context: ToolExecutionContext | None, store: SQLiteArtifactStore, on_changed: Callable[[], None] | None = None) -> dict[str, Any]:
    if context is None or context.project_store is None: raise ValueError("artifact_registration_context_required")
    task = context.task; owner = str(task.user or "").strip(); project_id = str(task.project_id or "").strip()
    project = context.project_store.get_project(project_id, requester_user_id=owner)
    if not project or project.owner_user_id != owner: raise PermissionError("artifact_registration_project_owner_required")
    schema = arguments.get("argument_schema")
    if not isinstance(schema, dict): raise ValueError("artifact_argument_schema_required")
    try: Draft202012Validator.check_schema(schema)
    except Exception as exc: raise ValueError("artifact_argument_schema_invalid") from exc
    raw_path = str(arguments.get("entrypoint_path") or "").strip()
    root = Path(project.root_path).resolve(); entrypoint = (root / raw_path).resolve()
    try: relative = entrypoint.relative_to(root)
    except ValueError as exc: raise ValueError("artifact_entrypoint_outside_project") from exc
    if not entrypoint.is_file(): raise ValueError("artifact_entrypoint_missing")
    if not entrypoint.stat().st_mode & 0o111: raise ValueError("artifact_entrypoint_not_executable")
    requested_id = str(arguments.get("artifact_id") or "").strip()
    artifact_id = requested_id if requested_id else "artifact." + _slug(str(arguments.get("name") or ""))
    record = store.register(artifact_id=artifact_id, name=str(arguments.get("name") or ""), description=str(arguments.get("description") or ""), project_id=project.project_id, owner_user_id=owner, entrypoint_path=str(relative), argument_schema=dict(schema or {}), timeout_seconds=arguments.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    if on_changed: on_changed()
    return {"artifact": record.to_dict(), "message": f'Registered artifact "{record.name}" as {record.artifact_id}.'}

def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    if not slug: raise ValueError("artifact_id_invalid")
    return slug
