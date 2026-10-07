"""Installed, reusable skill packages for Alphonse conversations."""

from __future__ import annotations

import os
import re
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
        records: list[SkillRecord] = []
        for path in sorted(self.skills_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_dir() or path.is_symlink():
                continue
            try:
                records.append(self._read_skill(path))
            except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
                # Invalid packages stay on disk for the admin to repair, but
                # must not enter the model-facing discovery catalog.
                continue
        return records

    def get(self, skill_id: str) -> SkillRecord | None:
        name = _validate_skill_name(str(skill_id or "").removeprefix("skill:"))
        path = self.skills_dir / name
        if not path.is_dir() or path.is_symlink():
            return None
        return self._read_skill(path)

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
