"""Installed, reusable skill packages for Alphonse conversations."""

from __future__ import annotations

import os
import re
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SKILL_FILE = "SKILL.md"
_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024
MAX_SKILL_FILE_BYTES = 256_000


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    name: str
    description: str
    directory: str
    instructions: str

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
        self.skills_dir = Path(skills_dir).expanduser() if skills_dir is not None else default_skills_dir()
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "SkillStore":
        return cls()

    def list_skills(self) -> list[SkillRecord]:
        records: dict[str, SkillRecord] = {}
        for directory in (self._bundled_skills_dir(), self.skills_dir):
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
                if not path.is_dir() or path.is_symlink():
                    continue
                try:
                    record = self._read_skill(path)
                    records[record.skill_id] = record
                except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
                    continue
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
            result.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size})
        return result

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
            raise ValueError("skill_file_too_large")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("skill_file_invalid")
        previous = path.read_text(encoding="utf-8") if path.is_file() else None
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
        if relative_path == SKILL_FILE:
            raise ValueError("skill_definition_required")
        root = Path(self._editable(skill_id).directory).resolve()
        path = _skill_path(root, relative_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("skill_file_not_found")
        path.unlink()

    def add_artifact_instructions(self, skill_id: str, artifact_id: str, name: str, instructions: str, definition: dict[str, Any] | None = None) -> None:
        record = self._editable(skill_id)
        path = Path(record.directory) / SKILL_FILE
        current = path.read_text(encoding="utf-8")
        manifest = ""
        if definition:
            manifest = "\n\nDefinition:\n\n```json\n" + json.dumps(definition, ensure_ascii=False, indent=2, sort_keys=True) + "\n```"
        section = f"\n\n## Artifact: {name}\n\nArtifact ID: `{artifact_id}`{manifest}\n\nOperating instructions:\n\n{str(instructions).strip()}\n"
        if f"Artifact ID: `{artifact_id}`" in current:
            raise ValueError("artifact_already_in_skill")
        updated = current.rstrip() + section
        if len(updated.encode("utf-8")) > MAX_SKILL_FILE_BYTES:
            raise ValueError("skill_file_too_large")
        path.write_text(updated, encoding="utf-8")
        self._read_skill(Path(record.directory))

    def update_description(self, skill_id: str, description: str) -> SkillRecord:
        record = self._editable(skill_id)
        value = str(description or "").strip()
        if not value or len(value) > MAX_SKILL_DESCRIPTION_LENGTH:
            raise ValueError("skill_description_invalid")
        path = Path(record.directory) / SKILL_FILE
        content = path.read_text(encoding="utf-8")
        metadata, instructions = _parse_skill_file(content)
        metadata["description"] = value
        serialized = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
        path.write_text(f"---\n{serialized}\n---\n\n{instructions}\n", encoding="utf-8")
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

    def install_directory(self, source: str | Path) -> SkillRecord:
        """Copy a validated skill package into the local library."""
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_dir():
            raise ValueError("skill_source_must_be_directory")
        record = self._read_skill(source_path)
        target = self.skills_dir / record.name
        if target.exists():
            raise ValueError("skill_already_installed")
        staging = Path(tempfile.mkdtemp(prefix=".skill-install-", dir=self.skills_dir))
        try:
            shutil.copytree(source_path, staging / record.name, dirs_exist_ok=True, symlinks=False)
            self._read_skill(staging / record.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / record.name).rename(target)
            return self._read_skill(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def create_skill(self, name: str, description: str, instructions: str) -> SkillRecord:
        """Create and install a plain-instruction skill package atomically."""
        skill_name = _validate_skill_name(name)
        skill_description = str(description or "").strip()
        skill_instructions = str(instructions or "").strip()
        if not skill_description or len(skill_description) > MAX_SKILL_DESCRIPTION_LENGTH:
            raise ValueError("skill_description_invalid")
        if not skill_instructions:
            raise ValueError("skill_instructions_required")
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
            return self.install_directory(source)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def replace_directory(self, skill_id: str, source: str | Path) -> SkillRecord:
        """Atomically replace an installed package with a validated package of the same name."""
        name = _validate_skill_name(str(skill_id or "").removeprefix("skill:"))
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_dir():
            raise ValueError("skill_source_must_be_directory")
        source_record = self._read_skill(source_path)
        if source_record.name != name:
            raise ValueError("skill_update_name_mismatch")

        target = self.skills_dir / name
        if target.is_symlink():
            raise ValueError("skill_package_symlink_not_allowed")
        if not target.is_dir():
            raise ValueError("skill_not_installed")

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
            except Exception:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                backup.rename(target)
                raise
            return updated
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def uninstall(self, skill_id: str) -> str:
        """Remove one installed skill package from the local library."""
        name = _validate_skill_name(str(skill_id or "").removeprefix("skill:"))
        target = self.skills_dir / name
        if target.is_symlink():
            raise ValueError("skill_package_symlink_not_allowed")
        if not target.is_dir():
            raise ValueError("skill_not_installed")
        if not target.resolve().is_relative_to(self.skills_dir.resolve()):
            raise ValueError("skill_package_outside_store")
        shutil.rmtree(target)
        return f"skill:{name}"

    def _read_skill(self, directory: Path) -> SkillRecord:
        if directory.is_symlink():
            raise ValueError("skill_package_symlink_not_allowed")
        path = directory / SKILL_FILE
        if path.is_symlink() or path.stat().st_size > MAX_SKILL_FILE_BYTES:
            raise ValueError("skill_file_invalid")
        content = path.read_text(encoding="utf-8")
        metadata, instructions = _parse_skill_file(content)
        name = _validate_skill_name(metadata.get("name"))
        if directory.name != name:
            raise ValueError("skill_directory_name_mismatch")
        description = str(metadata.get("description") or "").strip()
        if not description or len(description) > MAX_SKILL_DESCRIPTION_LENGTH:
            raise ValueError("skill_description_invalid")
        return SkillRecord(
            skill_id=f"skill:{name}", name=name, description=description,
            directory=str(directory.resolve()), instructions=instructions,
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
    if not instructions:
        raise ValueError("skill_instructions_required")
    return metadata, instructions


def _validate_skill_name(value: Any) -> str:
    name = str(value or "").strip()
    if len(name) > MAX_SKILL_NAME_LENGTH or not _SKILL_NAME.fullmatch(name):
        raise ValueError("skill_name_invalid")
    return name


def _skill_path(root: Path, relative_path: str) -> Path:
    value = str(relative_path or "").strip()
    if not value or Path(value).is_absolute():
        raise ValueError("skill_file_path_invalid")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("skill_file_outside_package")
    return path
