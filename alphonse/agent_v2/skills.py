"""Installed, reusable skill packages for Alphonse conversations."""

from __future__ import annotations

import os
import re
import json
import shutil
import tempfile
import logging
import unicodedata
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


SKILL_FILE = "SKILL.md"
logger = logging.getLogger(__name__)
_FRONTMATTER_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024
MAX_SKILL_COMPATIBILITY_LENGTH = 500
# Alphonse operational limits, not Agent Skills format constraints.
MAX_SKILL_INSTRUCTIONS_LENGTH = 12_000
MAX_SKILL_FILE_BYTES = 256_000


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    name: str
    description: str
    directory: str
    instructions: str

    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None

    def package_metadata(self) -> dict[str, Any]:
        return {"license": self.license, "compatibility": self.compatibility,
                "metadata": dict(self.metadata), "allowed-tools": self.allowed_tools}

    def candidate(self) -> dict[str, str]:
        return {
            "id": self.skill_id,
            "title": self.name,
            "description": self.description,
            "snippet": self.description,
        }


class SkillStore:
    """Filesystem-backed library of admin-installed SKILL.md packages.

    Skill files are instructions, not executable capabilities. This store never
    executes package scripts or grants tools/permissions described by a skill.
    """

    def __init__(self, skills_dir: str | Path | None = None) -> None:
        self.skills_dir = (Path(skills_dir).expanduser() if skills_dir is not None else default_skills_dir()).resolve()
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics: list[dict[str, str]] = []

    @classmethod
    def default(cls) -> "SkillStore":
        return cls()

    def list_skills(self) -> list[SkillRecord]:
        records: dict[str, SkillRecord] = {}
        self.diagnostics = []
        for directory in (self._bundled_skills_dir(), self.skills_dir):
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
                if path.name.startswith(".skill-") or not (path.is_dir() or path.is_symlink()):
                    continue
                try:
                    record = self._read_skill(path)
                    records[record.skill_id] = record
                except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
                    diagnostic = {"directory": str(path), "error": str(exc),
                                  "category": "operational" if str(exc).startswith("skill_operational_") else "specification" if isinstance(exc, (ValueError, yaml.YAMLError)) else "filesystem"}
                    self.diagnostics.append(diagnostic)
                    logger.warning("Invalid skill package %s: %s", path, exc)
        return sorted(records.values(), key=lambda item: item.name.casefold())

    def get(self, skill_id: str) -> SkillRecord | None:
        name = _validate_skill_name(str(skill_id or "").removeprefix("skill:"))
        path = self.skills_dir / name
        if path.is_dir() and not path.is_symlink():
            return self._read_skill(path)
        bundled = self._bundled_skills_dir() / name
        if not bundled.is_dir() or bundled.is_symlink():
            return None
        return self._read_skill(bundled)

    def files(self, skill_id: str) -> list[dict[str, Any]]:
        record = self.get(skill_id)
        if record is None:
            raise KeyError("skill_not_found")
        root = Path(record.directory)
        result = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            text = False
            if size <= MAX_SKILL_FILE_BYTES:
                try:
                    content = path.read_bytes()
                    content.decode("utf-8")
                    text = b"\x00" not in content
                except UnicodeDecodeError:
                    pass
            result.append({"path": path.relative_to(root).as_posix(), "size_bytes": size, "text_editable": text})
        return result

    def inspect_file(self, skill_id: str, relative_path: str) -> dict[str, Any]:
        record = self.get(skill_id)
        if record is None:
            raise KeyError("skill_not_found")
        root = Path(record.directory)
        path = _skill_path(root, relative_path)
        info = next((item for item in self.files(skill_id) if item["path"] == path.relative_to(root).as_posix()), None)
        if info is None:
            raise ValueError("skill_file_not_found")
        return {**info, "content": self.read_file(skill_id, relative_path) if info["text_editable"] else None}

    def _state_root(self, name: str) -> Path:
        root = self.skills_dir / ".skill-state" / _validate_skill_name(name)
        # Harness state is never a package resource or export input.
        for parent in (root.parent, root):
            if parent.is_symlink():
                raise ValueError("skill_state_symlink_not_allowed")
            parent.mkdir(exist_ok=True)
        return root

    @staticmethod
    def _digest(root: Path) -> str:
        _validate_package_files(root)
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(json.dumps([path.relative_to(root).as_posix(), path.stat().st_mode & 0o777]).encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def _provenance(self, name: str) -> dict[str, Any]:
        path = self._state_root(name) / "provenance.json"
        if path.is_symlink():
            raise ValueError("skill_state_symlink_not_allowed")
        return json.loads(path.read_text()) if path.is_file() else {}

    def _record_install(self, record: SkillRecord, source: Path, *, override: bool = False, owner: str = "") -> None:
        state = self._state_root(record.name)
        digest = self._digest(Path(record.directory))
        value = {"source": str(source), "base_revision": digest, "installed_revision": digest,
                 "override": override, "owner": owner, "audience": "household"}
        self._retain(record, provenance=value)
        temporary = state / (uuid.uuid4().hex + ".json")
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(state / "provenance.json")

    def management_status(self, skill_id: str) -> dict[str, Any]:
        record = self.get(skill_id)
        if record is None:
            raise KeyError("skill_not_found")
        installed = Path(record.directory).parent == self.skills_dir.resolve()
        provenance = self._provenance(record.name) if installed else {}
        validation_error = ""
        try:
            revision = self._digest(Path(record.directory))
        except (OSError, ValueError) as exc:
            # Legacy packages may have linked environments. Keep their management
            # and migration controls visible without treating them as portable.
            revision = ""
            validation_error = str(exc)
        bundled = self._bundled_skills_dir() / record.name
        try:
            upstream = self._digest(bundled) if bundled.is_dir() else None
        except (OSError, ValueError) as exc:
            upstream = None
            validation_error = str(exc)
        return {**provenance, "kind": "household" if installed else "bundled", "local_revision": revision,
                "validation_error": validation_error,
                "recovery_directory": str(self._state_root(record.name)) if installed else None,
                "customized": installed and revision != provenance.get("base_revision"),
                "provenance_known": bool(provenance) or not installed,
                "bundled_update_available": bool(provenance.get("override") and upstream != provenance.get("base_revision"))}

    def _retain(self, record: SkillRecord, *, provenance: dict[str, Any] | None = None) -> Path:
        root = self._state_root(record.name) / uuid.uuid4().hex
        root.mkdir()
        # Keep complete recovery content, including any legacy colocated live data.
        shutil.copytree(record.directory, root / record.name, symlinks=True)
        (root / "provenance.json").write_text(json.dumps(self._provenance(record.name) if provenance is None else provenance))
        return root

    def customize(self, skill_id: str, *, owner: str = "") -> SkillRecord:
        record = self.get(skill_id)
        if record is None:
            raise KeyError("skill_not_found")
        if Path(record.directory).parent == self.skills_dir.resolve():
            raise ValueError("skill_already_customized")
        installed = self.install_directory(record.directory, owner=owner, _allow_bundled_override=True)
        self._record_install(installed, Path(record.directory), override=True, owner=owner)
        return installed

    def read_file(self, skill_id: str, relative_path: str) -> str:
        record = self.get(skill_id)
        if record is None:
            raise KeyError("skill_not_found")
        root = Path(record.directory).resolve()
        path = _skill_path(root, relative_path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SKILL_FILE_BYTES:
            raise ValueError("skill_file_invalid")
        return path.read_text(encoding="utf-8")

    def write_file(self, skill_id: str, relative_path: str, content: str) -> None:
        root = Path(self._editable(skill_id).directory).resolve()
        path = _skill_path(root, relative_path)
        value = str(content)
        if len(value.encode("utf-8")) > MAX_SKILL_FILE_BYTES:
            raise ValueError("skill_operational_file_too_large: file limit is 256000 bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("skill_file_invalid")
        previous = None
        if path.is_file():
            if path.stat().st_size > MAX_SKILL_FILE_BYTES:
                raise ValueError("skill_file_not_text_editable")
            try:
                previous = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise ValueError("skill_file_not_text_editable") from None
            if "\x00" in previous:
                raise ValueError("skill_file_not_text_editable")
        self._retain(self._editable(skill_id))
        path.write_text(value, encoding="utf-8")
        try:
            self._read_skill(root)
        except Exception:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(previous, encoding="utf-8")
            raise

    def delete_file(self, skill_id: str, relative_path: str) -> None:
        root = Path(self._editable(skill_id).directory).resolve()
        path = _skill_path(root, relative_path)
        if path == root / SKILL_FILE:
            raise ValueError("skill_definition_required")
        if path.is_symlink() or not path.is_file():
            raise ValueError("skill_file_not_found")
        self._retain(self._editable(skill_id))
        path.unlink()

    def add_file_from_path(self, skill_id: str, source: str | Path) -> str:
        root = Path(self._editable(skill_id).directory).resolve()
        source_path = Path(source).expanduser()
        if source_path.is_symlink():
            raise ValueError("skill_source_file_symlink_not_allowed")
        source_path = source_path.resolve(strict=True)
        if not source_path.is_file():
            raise ValueError("skill_source_file_required")
        name = source_path.name
        if name == SKILL_FILE:
            raise ValueError("skill_definition_cannot_be_replaced_by_import")
        target = _skill_path(root, name)
        if target.exists() or target.is_symlink():
            raise ValueError("skill_file_already_exists")
        self._retain(self._editable(skill_id))
        shutil.copyfile(source_path, target)
        return name

    def add_artifact_instructions(self, skill_id: str, artifact_id: str, name: str, instructions: str, definition: dict[str, Any] | None = None) -> None:
        record = self._editable(skill_id)
        path = Path(record.directory) / SKILL_FILE
        current = path.read_text(encoding="utf-8")
        manifest = ""
        if definition:
            manifest = "\n\nDefinition:\n\n```json\n" + json.dumps(definition, ensure_ascii=False, indent=2, sort_keys=True) + "\n```"
        section = f"\n\n## Artifact: {name}\n\nArtifact ID: `{artifact_id}`{manifest}\n\nOperating instructions:\n\n{str(instructions).strip()}\n"
        if f"Artifact ID: `{artifact_id}`" in current:
            return
        updated = current.rstrip() + section
        if len(updated.encode("utf-8")) > MAX_SKILL_FILE_BYTES:
            raise ValueError("skill_operational_file_too_large: file limit is 256000 bytes")
        path.write_text(updated, encoding="utf-8")

    def artifact_instructions(self, skill_id: str, artifact_id: str) -> str:
        skill = self.get(skill_id)
        if skill is None:
            return ""
        for section in re.split(r"(?m)^## Artifact: ", skill.instructions)[1:]:
            if f"Artifact ID: `{artifact_id}`" in section:
                return section.partition("Operating instructions:")[2].strip()
        return ""

    def update_artifact_definition(self, skill_id: str, artifact_id: str, name: str, definition: dict[str, Any]) -> None:
        record = self._editable(skill_id)
        path = Path(record.directory) / SKILL_FILE
        current = path.read_text(encoding="utf-8")
        sections = re.split(r"(?m)(?=^## Artifact: )", current)
        for index, section in enumerate(sections):
            if f"Artifact ID: `{artifact_id}`" not in section:
                continue
            detail = section.partition("Operating instructions:")[2].strip()
            if not detail:
                raise ValueError("artifact_instructions_required")
            manifest = json.dumps(definition, ensure_ascii=False, indent=2, sort_keys=True)
            sections[index] = f"## Artifact: {name}\n\nArtifact ID: `{artifact_id}`\n\nDefinition:\n\n```json\n{manifest}\n```\n\nOperating instructions:\n\n{detail}\n\n"
            updated = "".join(sections).rstrip() + "\n"
            if len(updated.encode("utf-8")) > MAX_SKILL_FILE_BYTES:
                raise ValueError("skill_operational_file_too_large: file limit is 256000 bytes")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(updated, encoding="utf-8")
            temporary.replace(path)
            return
        raise ValueError("artifact_skill_definition_missing")
        self._read_skill(Path(record.directory))

    def remove_artifact_instructions(self, skill_id: str, artifact_id: str) -> None:
        record = self._editable(skill_id)
        path = Path(record.directory) / SKILL_FILE
        current = path.read_text(encoding="utf-8")
        marker = f"Artifact ID: `{artifact_id}`"
        marker_index = current.find(marker)
        if marker_index < 0:
            return
        heading_start = current.rfind("\n## Artifact:", 0, marker_index)
        if heading_start < 0:
            return
        next_heading = current.find("\n## ", marker_index)
        section_end = len(current) if next_heading < 0 else next_heading + 1
        updated = (current[:heading_start].rstrip() + "\n\n" + current[section_end:].lstrip()).rstrip() + "\n"
        path.write_text(updated, encoding="utf-8")
        self._read_skill(Path(record.directory))

    def update_description(self, skill_id: str, description: str) -> SkillRecord:
        record = self._editable(skill_id)
        value = _validate_description(description)
        path = Path(record.directory) / SKILL_FILE
        content = path.read_text(encoding="utf-8")
        metadata, instructions = _parse_skill_file(content)
        metadata["description"] = value
        serialized = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
        self.write_file(skill_id, SKILL_FILE, f"---\n{serialized}\n---\n\n{instructions}\n")
        return self._read_skill(Path(record.directory))

    def _editable(self, skill_id: str) -> SkillRecord:
        name = _validate_skill_name(str(skill_id or "").removeprefix("skill:"))
        target = self.skills_dir / name
        if target.is_symlink() or not target.is_dir():
            raise KeyError("installed_skill_not_found")
        return self._read_skill(target)

    @staticmethod
    def _bundled_skills_dir() -> Path:
        return Path(__file__).resolve().parent / "builtin_skills"

    def install_directory(self, source: str | Path, *, owner: str = "", _allow_bundled_override: bool = False) -> SkillRecord:
        """Copy a validated skill package into the local library."""
        if Path(source).expanduser().is_symlink():
            raise ValueError("skill_operational_symlink_not_allowed")
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_dir():
            raise ValueError("skill_source_must_be_directory")
        record = self._read_skill(source_path)
        _validate_package_files(source_path)
        target = self.skills_dir / record.name
        if target.exists() or target.is_symlink():
            raise ValueError("skill_already_installed")
        if not _allow_bundled_override and (self._bundled_skills_dir() / record.name).exists():
            raise ValueError("skill_bundled_name_conflict: use household customization to create an explicit override")
        staging = Path(tempfile.mkdtemp(prefix=".skill-install-", dir=self.skills_dir))
        try:
            shutil.copytree(source_path, staging / record.name, dirs_exist_ok=True, symlinks=False)
            self._read_skill(staging / record.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / record.name).rename(target)
            try:
                installed = self._read_skill(target)
                self._record_install(installed, source_path, owner=owner)
            except Exception:
                shutil.rmtree(target)
                raise
            return installed
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def create_skill(self, name: str, description: str, instructions: str, *, owner: str = "") -> SkillRecord:
        """Create and install a plain-instruction skill package atomically."""
        skill_name = _validate_skill_name(name)
        skill_description = _validate_description(description)
        if not isinstance(instructions, str):
            raise ValueError("skill_operational_instructions_required: authored instructions must be a string")
        skill_instructions = instructions.strip()
        if not skill_instructions:
            raise ValueError("skill_operational_instructions_required: authored skills need instructions")
        if len(skill_instructions) > MAX_SKILL_INSTRUCTIONS_LENGTH:
            raise ValueError("skill_operational_instructions_too_long: authoring limit is 12000 characters")
        if self.get(skill_name) is not None:
            raise ValueError("skill_already_installed")
        staging = Path(tempfile.mkdtemp(prefix=".skill-author-", dir=self.skills_dir))
        source = staging / skill_name
        try:
            source.mkdir()
            metadata = yaml.safe_dump(
                {"name": skill_name, "description": skill_description},
                allow_unicode=True, sort_keys=False,
            ).strip()
            skill_file = source / SKILL_FILE
            skill_file.write_text(f"---\n{metadata}\n---\n\n{skill_instructions}\n", encoding="utf-8")
            self._read_skill(source)
            installed = self.install_directory(source, owner=owner)
            self._record_install(installed, Path("household-authored"), owner=owner)
            return installed
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def replace_directory(self, skill_id: str, source: str | Path, *, resolution: str = "") -> SkillRecord:
        """Atomically replace an installed package with a validated package of the same name."""
        name = _validate_skill_name(str(skill_id or "").removeprefix("skill:"))
        if Path(source).expanduser().is_symlink():
            raise ValueError("skill_operational_symlink_not_allowed")
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_dir():
            raise ValueError("skill_source_must_be_directory")
        source_record = self._read_skill(source_path)
        _validate_package_files(source_path)
        if source_record.name != name:
            raise ValueError("skill_update_name_mismatch")

        target = self.skills_dir / name
        if target.is_symlink():
            raise ValueError("skill_operational_symlink_not_allowed")
        if not target.is_dir():
            raise ValueError("skill_not_installed")
        if _has_private_state(target):
            raise ValueError("skill_update_private_state: reconcile live data, credentials and runtime files outside the package before replacement")
        status = self.management_status(skill_id)
        if resolution not in ("", "accept-upstream"):
            raise ValueError("skill_update_resolution_invalid")
        if status["customized"] and resolution != "accept-upstream":
            raise ValueError("skill_update_conflict: local changes exist; keep local or explicitly accept upstream; recovery copy will be retained")
        self._retain(self._editable(skill_id))

        staging = Path(tempfile.mkdtemp(prefix=".skill-update-", dir=self.skills_dir))
        staged_package = staging / name
        backup = staging / "previous"
        try:
            shutil.copytree(source_path, staged_package, dirs_exist_ok=True, symlinks=False)
            self._read_skill(staged_package)
            target.rename(backup)
            try:
                staged_package.rename(target)
                updated = self._read_skill(target)
                previous_provenance = self._provenance(name)
                self._record_install(updated, source_path, override=bool(previous_provenance.get("override")), owner=previous_provenance.get("owner", ""))
            except Exception:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                backup.rename(target)
                raise
            return updated
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def export_directory(self, skill_id: str, destination: str | Path) -> Path:
        """Export a whole package into a new name-matching directory."""
        record = self.get(skill_id)
        if record is None:
            raise KeyError("skill_not_found")
        source = Path(record.directory)
        _validate_package_files(source)
        if _has_private_state(source):
            raise ValueError("skill_export_private_state: reconcile live data, credentials and runtime files outside the portable package before export")
        parent = Path(destination).expanduser().resolve(strict=True)
        if not parent.is_dir() or parent.is_relative_to(source.resolve()):
            raise ValueError("skill_export_destination_invalid")
        target = parent / record.name
        if target.exists() or target.is_symlink():
            raise ValueError("skill_export_destination_exists")
        staging = Path(tempfile.mkdtemp(prefix=".skill-export-", dir=parent))
        try:
            staged = staging / record.name
            shutil.copytree(source, staged)
            self._read_skill(staged)
            staged.rename(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return target

    def uninstall(self, skill_id: str) -> str:
        """Remove one installed skill package from the local library."""
        name = _validate_skill_name(str(skill_id or "").removeprefix("skill:"))
        target = self.skills_dir / name
        if target.is_symlink():
            raise ValueError("skill_operational_symlink_not_allowed")
        if not target.is_dir():
            raise ValueError("skill_not_installed")
        if not target.resolve().is_relative_to(self.skills_dir.resolve()):
            raise ValueError("skill_package_outside_store")
        self._retain(self._editable(skill_id))
        shutil.rmtree(target)
        return f"skill:{name}"

    def _read_skill(self, directory: Path) -> SkillRecord:
        if directory.is_symlink():
            raise ValueError("skill_operational_symlink_not_allowed")
        path = directory / SKILL_FILE
        if path.is_symlink():
            raise ValueError("skill_operational_symlink_not_allowed")
        if path.stat().st_size > MAX_SKILL_FILE_BYTES:
            raise ValueError("skill_operational_file_too_large: file limit is 256000 bytes")
        content = path.read_text(encoding="utf-8")
        metadata, instructions = _parse_skill_file(content)
        name = _validate_skill_name(metadata.get("name"))
        if unicodedata.normalize("NFKC", directory.name) != name:
            raise ValueError("skill_directory_name_mismatch")
        description = _validate_description(metadata.get("description"))
        _validate_optional_fields(metadata)
        return SkillRecord(
            skill_id=f"skill:{name}", name=name, description=description,
            directory=str(directory.resolve()), instructions=instructions,
            license=metadata.get("license"), compatibility=metadata.get("compatibility"),
            metadata=dict(metadata.get("metadata", {})), allowed_tools=metadata.get("allowed-tools"),
        )


def default_skills_dir() -> Path:
    configured = str(os.getenv("ALPHONSE_SKILLS_DIR") or "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".alphonse" / "skills"


def _parse_skill_file(content: str) -> tuple[dict[str, Any], str]:
    lines = str(content or "").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("skill_frontmatter_required")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("skill_frontmatter_unclosed") from exc
    metadata = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(metadata, dict):
        raise ValueError("skill_frontmatter_invalid")
    instructions = "\n".join(lines[end + 1:]).strip()
    return metadata, instructions


def _validate_skill_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("skill_name_invalid: name must be a non-empty string")
    name = unicodedata.normalize("NFKC", value.strip())
    if (len(name) > MAX_SKILL_NAME_LENGTH or name != name.lower()
            or name.startswith("-") or name.endswith("-") or "--" in name
            or not all(c.isalnum() or c == "-" for c in name)):
        raise ValueError("skill_name_invalid: use 1-64 lowercase letters, digits and single hyphens")
    return name


def _validate_description(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_SKILL_DESCRIPTION_LENGTH:
        raise ValueError("skill_description_invalid: description must be a non-empty string of at most 1024 characters")
    return value


def _validate_optional_fields(metadata: dict[str, Any]) -> None:
    if set(metadata) - _FRONTMATTER_FIELDS:
        raise ValueError("skill_frontmatter_unknown_fields: put custom fields in metadata")
    for key in ("license", "compatibility", "allowed-tools"):
        if key in metadata and not isinstance(metadata[key], str):
            raise ValueError(f"skill_{key}_invalid: {key} must be a string")
    if "compatibility" in metadata and not 1 <= len(metadata["compatibility"]) <= MAX_SKILL_COMPATIBILITY_LENGTH:
        raise ValueError("skill_compatibility_invalid: use 1-500 characters")
    if "metadata" in metadata and (not isinstance(metadata["metadata"], dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata["metadata"].items())):
        raise ValueError("skill_metadata_invalid: metadata must map string keys to string values")


def _validate_package_files(root: Path) -> None:
    # Operational containment policy: never dereference package links on copy.
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"skill_operational_symlink_not_allowed: {path.relative_to(root)}")
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"skill_operational_special_file_not_allowed: {path.relative_to(root)}")


def _has_private_state(root: Path) -> bool:
    private_names = {".env", ".git", ".venv", "__pycache__", "credentials", "runtime", "data"}
    return any(any(part in private_names or part.startswith(".env.") for part in path.relative_to(root).parts)
               for path in root.rglob("*"))


def _skill_path(root: Path, relative_path: str) -> Path:
    value = str(relative_path or "").strip()
    if not value or Path(value).is_absolute():
        raise ValueError("skill_file_path_invalid")
    candidate = root / value
    if any(part.is_symlink() for part in (candidate, *candidate.parents) if part != root and part.is_relative_to(root)):
        raise ValueError("skill_operational_symlink_not_allowed")
    path = candidate.resolve()
    if not path.is_relative_to(root):
        raise ValueError("skill_file_outside_package")
    return path
