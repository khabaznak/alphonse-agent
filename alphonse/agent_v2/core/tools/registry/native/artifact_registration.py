"""Register skill-owned resources; registration never changes program/data files."""
from __future__ import annotations

import re
from typing import Any, Callable
from pathlib import Path
from jsonschema import Draft202012Validator

from alphonse.agent_v2.artifacts import ArtifactRecord, DEFAULT_TIMEOUT_SECONDS, SQLiteArtifactStore, _artifact_id, _timeout, _validate_schema
from alphonse.agent_v2.artifact_resources import ArtifactResources, EXECUTION_SCHEMA
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

ARTIFACT_REGISTRATION_TOOL_ID = "native.skill_artifact_register"
ARTIFACT_REGISTRATION_TOOL_NAME = "skill_artifact_register"
ARTIFACT_REGISTRATION_ARGUMENT_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "name": {"type": "string"}, "description": {"type": "string"}, "instructions": {"type": "string"},
        "skill_id": {"type": "string"}, "entrypoint_path": {"type": "string"},
        "argument_schema": {"type": "object"}, "artifact_id": {"type": "string"}, "timeout_seconds": {"type": "number"},
        "storage_kind": {"type": "string", "enum": ["shared", "skill", "project"], "default": "shared"},
        "execution": EXECUTION_SCHEMA,
        "shared_with_user_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    },
    "required": ["name", "description", "instructions", "skill_id", "entrypoint_path", "argument_schema"],
}


def build_artifact_registration_tool_definition(store: SQLiteArtifactStore, skill_store: SkillStore | None = None, on_changed: Callable[[], None] | None = None) -> ToolDefinition:
    descriptor = ToolDescriptor(ARTIFACT_REGISTRATION_TOOL_ID, ARTIFACT_REGISTRATION_TOOL_NAME, ToolKind.NATIVE,
        "Register an existing program under an installed skill. Default shared location: ARTIFACT_REPOSITORY/skill-name/artifact-id-without-prefix/. Supply an entrypoint relative to that storage root and explicit runtime/working-directory/configuration metadata. No active project is needed for shared or skill storage. Registration only records metadata and operating instructions; it never creates, moves or repairs program/data files. Legacy project storage requires the project owner. Artifacts are not independent tools.",
        dict(ARTIFACT_REGISTRATION_ARGUMENT_SCHEMA), ("skill_management", "local_execution"), ("native", "skills", "artifacts"))
    return ToolDefinition(descriptor, lambda arguments, context=None: execute_artifact_registration(arguments, context=context, store=store, skill_store=skill_store, on_changed=on_changed), dict(ARTIFACT_REGISTRATION_ARGUMENT_SCHEMA), accepts_context=True)


def execute_artifact_registration(arguments: dict[str, Any], *, context: ToolExecutionContext | None, store: SQLiteArtifactStore, skill_store: SkillStore | None = None, on_changed: Callable[[], None] | None = None) -> dict[str, Any]:
    if context is None or skill_store is None:
        raise ValueError("artifact_registration_context_required")
    Draft202012Validator(ARTIFACT_REGISTRATION_ARGUMENT_SCHEMA).validate(arguments)
    owner = str(context.task.user or "").strip()
    if not owner: raise PermissionError("artifact_owner_required")
    skill = skill_store.get(str(arguments.get("skill_id") or ""))
    if skill is None or not Path(skill.directory).resolve().is_relative_to(skill_store.skills_dir.resolve()):
        raise ValueError("installed_skill_required")
    name, description, instructions = (str(arguments.get(key) or "").strip() for key in ("name", "description", "instructions"))
    if not name or not description or not instructions:
        raise ValueError("artifact_name_description_and_instructions_required")
    artifact_id = _artifact_id(arguments.get("artifact_id") or "artifact." + _slug(name))
    if store.get(artifact_id) is not None: raise ValueError("artifact_id_already_registered")
    kind = str(arguments.get("storage_kind") or "shared")
    project_id = str(context.task.project_id or "") if kind == "project" else ""
    if kind == "project":
        project = context.project_store.get_project(project_id, requester_user_id=owner) if context.project_store else None
        if project is None or project.owner_user_id != owner:
            raise PermissionError("artifact_registration_project_owner_required")
    schema = arguments["argument_schema"]
    _validate_schema(schema)
    timeout = _timeout(arguments.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    record = ArtifactRecord(artifact_id, name, description, project_id, owner, str(arguments["entrypoint_path"]),
        dict(schema), timeout, True, "", "", skill.skill_id, kind, str(Path(project.root_path).resolve()) if kind == "project" else "", dict(arguments.get("execution") or {}), tuple(arguments.get("shared_with_user_ids") or ()))
    resources = context.artifact_resources or ArtifactResources(store, skill_store, context.project_store)
    resources.validate(record)
    saved = store.register(artifact_id=artifact_id, name=name, description=description, project_id=project_id,
        owner_user_id=owner, entrypoint_path=record.entrypoint_path, argument_schema=record.argument_schema,
        timeout_seconds=timeout, skill_id=skill.skill_id, storage_kind=kind, legacy_root_path=record.legacy_root_path, execution=record.execution,
        shared_with_user_ids=record.shared_with_user_ids)
    try:
        skill_store.add_artifact_instructions(skill.skill_id, artifact_id, name, instructions, saved.skill_definition())
        skill_store.update_artifact_definition(skill.skill_id, artifact_id, name, saved.skill_definition())
    except Exception:
        store.delete(artifact_id)
        raise
    if on_changed: on_changed()
    return {"artifact": saved.to_dict(), "resource": resources.describe(saved, owner), "message": f'Registered artifact "{name}" as {artifact_id}.'}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug: raise ValueError("artifact_id_invalid")
    return slug
