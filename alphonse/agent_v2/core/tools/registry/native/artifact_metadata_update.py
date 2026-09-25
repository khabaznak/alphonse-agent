"""Native tool for updating routing metadata on owned artifacts."""
from __future__ import annotations

from typing import Any, Callable

from alphonse.agent_v2.artifacts import SQLiteArtifactStore
from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

ARTIFACT_METADATA_UPDATE_TOOL_ID = "native.artifact_metadata_update"
ARTIFACT_METADATA_UPDATE_ARGUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "artifact_id": {"type": "string", "minLength": 1},
        "name": {"type": "string", "minLength": 1},
        "description": {"type": "string", "minLength": 1},
    },
    "required": ["artifact_id", "name", "description"],
}


def build_artifact_metadata_update_tool_definition(
    store: SQLiteArtifactStore,
    *,
    is_admin: Callable[[str], bool] | None = None,
    on_changed: Callable[[], None] | None = None,
) -> ToolDefinition:
    descriptor = ToolDescriptor(
        ARTIFACT_METADATA_UPDATE_TOOL_ID,
        "artifact_metadata_update",
        ToolKind.NATIVE,
        "Use this tool to change the registered catalog name or routing description of an artifact owned by the requester or managed by an admin. This updates the artifact registry metadata only; it does not edit README files, artifact programs, or other project files. Bash may inspect project files to identify and verify the artifact ID, but must not substitute file edits for this catalog update.",
        dict(ARTIFACT_METADATA_UPDATE_ARGUMENT_SCHEMA),
        ("artifact_metadata_management",),
        ("native", "artifacts", "metadata"),
        read_only=False,
    )
    return ToolDefinition(
        descriptor,
        lambda arguments, context=None: execute_artifact_metadata_update(
            arguments, context=context, store=store, is_admin=is_admin, on_changed=on_changed
        ),
        dict(ARTIFACT_METADATA_UPDATE_ARGUMENT_SCHEMA),
        accepts_context=True,
    )


def execute_artifact_metadata_update(
    arguments: dict[str, Any],
    *,
    context: ToolExecutionContext | None,
    store: SQLiteArtifactStore,
    is_admin: Callable[[str], bool] | None = None,
    on_changed: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if context is None:
        raise ValueError("artifact_metadata_update_context_required")
    actor = str(context.task.user or "").strip()
    if not actor:
        raise PermissionError("artifact_manager_required")
    artifact_id = str(arguments.get("artifact_id") or "").strip()
    record = store.get(artifact_id)
    if record is None:
        raise KeyError("artifact_not_found")
    if actor != record.owner_user_id and not (is_admin and is_admin(actor)):
        raise PermissionError("artifact_manager_required")
    name = str(arguments.get("name") or "").strip()
    description = str(arguments.get("description") or "").strip()
    if not name or not description:
        raise ValueError("artifact_name_and_description_required")
    saved = store.update_metadata(artifact_id, name=name, description=description)
    if on_changed:
        on_changed()
    return {"artifact": saved.to_dict(), "message": f'Updated metadata for artifact "{saved.artifact_id}".'}
