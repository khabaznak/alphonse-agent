"""Resolve skill-owned resources without making artifacts executable tools.

Only metadata is migrated here. Programs, environments and configuration remain
where their owner put them until a manager explicitly updates the registration.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import re
from typing import Any, Callable

from alphonse.agent_v2.artifacts import ArtifactRecord, SQLiteArtifactStore


EXECUTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "working_directory": {"type": "string"},
        "runtime_path": {"type": "string", "description": "Explicit absolute executable or a path relative to the artifact storage root; never a PATH lookup."},
        "configuration_paths": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "required_files": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    },
}


def contained_path(root: Path, value: str) -> Path:
    """Reject absolute paths, traversal and symlink escapes, even if missing."""
    raw = Path(str(value or "."))
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("artifact_path_outside_storage")
    path = (root / raw).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact_path_outside_storage")
    return path


class ArtifactResources:
    def __init__(self, store: SQLiteArtifactStore, skills: Any, projects: Any = None, *,
                 repository_root: Callable[[], Path] | None = None,
                 is_admin: Callable[[str], bool] | None = None) -> None:
        self.store, self.skills, self.projects = store, skills, projects
        self._repository_root = repository_root or (lambda: Path(skills.skills_dir).parent / "artifacts")
        self.is_admin = is_admin or (lambda _actor: False)

    @property
    def repository_root(self) -> Path:
        return Path(self._repository_root()).expanduser().absolute()

    def ensure_repository(self) -> Path:
        root = self.repository_root
        if root.is_symlink():
            raise ValueError("artifact_repository_symlink_not_allowed")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def shared_directory(self, skill_id: str, artifact_id: str) -> Path:
        skill_name = str(skill_id).removeprefix("skill:")
        artifact_name = str(artifact_id).removeprefix("artifact.")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", skill_name):
            raise ValueError("artifact_skill_invalid")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", artifact_name):
            raise ValueError("artifact_id_invalid")
        root = self.repository_root
        if root.is_symlink():
            raise ValueError("artifact_repository_symlink_not_allowed")
        if (root / skill_name).is_symlink() or (root / skill_name / artifact_name).is_symlink():
            raise ValueError("artifact_storage_symlink_not_allowed")
        return contained_path(root, f"{skill_name}/{artifact_name}")

    def access_records(self):
        """Legacy privacy guards include reconciled roots after migration."""
        import json
        for record in self.store.list():
            with self.store._connect() as conn:
                exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_legacy_migrations'").fetchone()
                row = conn.execute("SELECT payload FROM v2_legacy_migrations WHERE artifact_id=? AND state='complete'", (record.artifact_id,)).fetchone() if exists else None
            yield ArtifactRecord(**json.loads(row[0])['selected']) if row else record

    def can_use(self, record: ArtifactRecord, actor: str) -> bool:
        bindings = getattr(self, 'bindings', None)
        if bindings is not None:
            with self.store._connect() as conn:
                exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_legacy_migrations'").fetchone()
                row = conn.execute("SELECT payload FROM v2_legacy_migrations WHERE artifact_id=? AND state='complete'", (record.artifact_id,)).fetchone() if exists else None
            if row:
                import json
                try:
                    bindings._authorize(bindings._get(json.loads(row[0])['binding_id']), actor)
                    return True
                except (ValueError, PermissionError):
                    return False
        return bool(actor and (actor == record.owner_user_id or self.is_admin(actor)
                              or actor in record.shared_with_user_ids))

    def require_manager(self, record: ArtifactRecord, actor: str) -> None:
        if not actor or (actor != record.owner_user_id and not self.is_admin(actor)):
            raise PermissionError("artifact_manager_required")

    def storage_root(self, record: ArtifactRecord) -> Path:
        skill = self.skills.get(record.skill_id) if record.skill_id else None
        if skill is None:
            raise ValueError("artifact_installed_skill_required")
        if record.storage_kind == "shared":
            return self.shared_directory(skill.skill_id, record.artifact_id)
        if record.storage_kind == "skill":
            return Path(skill.directory).resolve()
        if record.storage_kind == "project":
            # Storage provenance is resolved against the registered owner, not
            # the project (or lack of project) of the requesting conversation.
            if record.legacy_root_path:
                root = Path(record.legacy_root_path).expanduser()
                if root.is_symlink():
                    raise ValueError("artifact_storage_symlink_not_allowed")
                return root.resolve()
            project = self.projects.get_project(record.project_id, requester_user_id=record.owner_user_id, include_archived=True) if self.projects else None
            if project is None:
                raise ValueError("artifact_storage_project_missing")
            return Path(project.root_path).expanduser().resolve()
        raise ValueError("artifact_storage_kind_invalid")

    def describe(self, record: ArtifactRecord, actor: str, *, management: bool = False) -> dict[str, Any]:
        if management:
            self.require_manager(record, actor)
        elif not self.can_use(record, actor):
            raise PermissionError("artifact_access_denied")
        detail: dict[str, Any] = {"artifact_id": record.artifact_id, "storage_kind": record.storage_kind,
                                  "status": "ready", "issues": [], "timeout_seconds": record.timeout_seconds}
        try:
            root = self.storage_root(record)
            entrypoint = contained_path(root, record.entrypoint_path)
            execution = record.execution
            cwd = contained_path(root, execution.get("working_directory", "."))
            runtime_value = str(execution.get("runtime_path") or "")
            runtime = (Path(runtime_value).expanduser().absolute() if Path(runtime_value).is_absolute()
                       else (contained_path(root, str(Path(runtime_value).parent)) / Path(runtime_value).name)) if runtime_value else None
            detail.update(storage_root=str(root), resolved_entrypoint_path=str(entrypoint), working_directory=str(cwd),
                          runtime_path=str(runtime) if runtime else "", invocation_prefix=[str(runtime), str(entrypoint)] if runtime else [str(entrypoint)])
            issues = detail["issues"]
            if not entrypoint.is_file(): issues.append("entrypoint_missing")
            if not cwd.is_dir(): issues.append("working_directory_missing")
            if runtime:
                if not runtime.is_file() or not os.access(runtime, os.X_OK): issues.append("runtime_missing_or_not_executable")
            elif entrypoint.is_file() and not os.access(entrypoint, os.X_OK):
                issues.append("entrypoint_not_executable")
            for field in ("configuration_paths", "required_files"):
                paths = [contained_path(root, value) for value in execution.get(field, [])]
                detail[field] = [{"path": str(path), "exists": path.is_file()} for path in paths]
                if any(not path.is_file() for path in paths): issues.append(f"{field}_missing")
            if not runtime and entrypoint.suffix == ".py":
                issues.append("runtime_not_registered")
                detail["runtime_note"] = "No explicit runtime registered. Follow the skill's verified runtime instructions; do not assume a system python3 environment is suitable."
            if not record.enabled: issues.append("disabled")
            if issues: detail["status"] = "unavailable"
        except (ValueError, OSError, PermissionError, KeyError) as exc:
            detail.update(status="unavailable", issues=[str(exc)])
        return detail

    def require_ready(self, artifact_id: str, actor: str, selected_skill_ids: list[str]) -> dict[str, Any]:
        record = self.store.get(artifact_id)
        if record is None:
            raise KeyError("artifact_not_found")
        if record.skill_id not in selected_skill_ids:
            raise PermissionError("artifact_skill_not_selected")
        detail = self.describe(record, actor)
        if detail["status"] != "ready":
            raise ValueError("artifact_resource_unavailable:" + ",".join(detail["issues"]))
        return detail

    def validate(self, record: ArtifactRecord, *, allow_missing_configuration: bool = True) -> None:
        from jsonschema import Draft202012Validator
        Draft202012Validator(EXECUTION_SCHEMA).validate(record.execution)
        for user in record.shared_with_user_ids:
            if not isinstance(user, str) or not user.strip():
                raise ValueError("artifact_shared_user_invalid")
        detail = self.describe(record, record.owner_user_id, management=True)
        allowed = {"configuration_paths_missing", "required_files_missing", "disabled"} if allow_missing_configuration else {"disabled"}
        errors = [item for item in detail["issues"] if item not in allowed]
        if errors:
            raise ValueError("artifact_resource_invalid:" + ",".join(errors))

    def relocate(self, artifact_id: str, actor: str, *, storage_kind: str, entrypoint_path: str,
                 execution: dict[str, Any], shared_with_user_ids: list[str] | None = None,
                 name: str | None = None, description: str | None = None) -> ArtifactRecord:
        record = self.store.get(artifact_id)
        if record is None: raise KeyError("artifact_not_found")
        self.require_manager(record, actor)
        updated = replace(record, storage_kind=storage_kind, entrypoint_path=entrypoint_path,
                          name=name if name is not None else record.name,
                          description=description if description is not None else record.description,
                          execution=dict(execution), shared_with_user_ids=tuple(shared_with_user_ids) if shared_with_user_ids is not None else record.shared_with_user_ids)
        self.validate(updated)
        # Definition edits preserve operating instructions and never touch the
        # executable, environment, configuration or data.
        self.skills.update_artifact_definition(updated.skill_id, updated.artifact_id, updated.name, updated.skill_definition())
        try:
            return self.store.update_location(updated)
        except Exception:
            self.skills.update_artifact_definition(record.skill_id, record.artifact_id, record.name, record.skill_definition())
            raise

    def update_metadata(self, artifact_id: str, actor: str, *, name: str, description: str) -> ArtifactRecord:
        record = self.store.get(artifact_id)
        if record is None:
            raise KeyError("artifact_not_found")
        self.require_manager(record, actor)
        name, description = str(name or "").strip(), str(description or "").strip()
        if not name or not description:
            raise ValueError("artifact_name_and_description_required")
        updated = replace(record, name=name, description=description)
        if record.skill_id:
            self.skills.update_artifact_definition(record.skill_id, artifact_id, name, updated.skill_definition())
        try:
            return self.store.update_metadata(artifact_id, name=name, description=description)
        except Exception:
            if record.skill_id:
                self.skills.update_artifact_definition(record.skill_id, artifact_id, record.name, record.skill_definition())
            raise

    def reconcile(self, record: ArtifactRecord, actor: str) -> dict[str, Any]:
        """Report known locations only. Never search arbitrary files or pick a copy."""
        self.require_manager(record, actor)
        result = self.describe(record, actor, management=True)
        candidates = []
        for kind in ("shared", "skill", "project"):
            candidate = replace(record, storage_kind=kind)
            detail = self.describe(candidate, actor, management=True)
            path = detail.get("resolved_entrypoint_path")
            if path and Path(path).is_file():
                candidates.append({"storage_kind": kind, "entrypoint_path": record.entrypoint_path,
                                   "resolved_entrypoint_path": path})
        result["location_candidates"] = candidates
        result["reconciliation"] = "ambiguous" if len({item["resolved_entrypoint_path"] for item in candidates}) > 1 else ("registered" if result["status"] == "ready" else "needs_manager_review")
        return result
