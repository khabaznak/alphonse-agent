from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillDefinition:
    key: str
    description: str
    aliases: list[str] = field(default_factory=list)
    arg_schema: dict[str, Any] | None = None


class SkillRegistry:
    def __init__(self, skills: list[SkillDefinition]) -> None:
        self._skills = {skill.key: skill for skill in skills}
        self._alias_map: dict[str, str] = {}
        for skill in skills:
            self._register_alias(skill.key)
            for alias in skill.aliases:
                self._register_alias(alias, skill.key)

    def list_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def get_skill(self, key: str) -> SkillDefinition | None:
        return self._skills.get(key)

    def match_alias(self, text: str) -> SkillDefinition | None:
        normalized = _normalize_command(text)
        if not normalized:
            return None
        skill_key = self._alias_map.get(normalized)
        if not skill_key:
            return None
        return self._skills.get(skill_key)

    def _register_alias(self, alias: str, skill_key: str | None = None) -> None:
        normalized = _normalize_command(alias)
        if not normalized:
            return
        self._alias_map.setdefault(normalized, skill_key or alias)


def build_default_registry() -> SkillRegistry:
    return SkillRegistry(
        [
            SkillDefinition(
                key="system.status",
                description="Summarize current system status",
                aliases=["status", "how are you", "system status"],
            ),
            SkillDefinition(
                key="system.joke",
                description="Tell a short, gentle joke",
                aliases=["joke", "tell me a joke"],
            ),
            SkillDefinition(
                key="system.help",
                description="List available commands and skills",
                aliases=["help", "commands", "/help"],
            ),
            SkillDefinition(
                key="conversation.echo",
                description="Echo back a short acknowledgement",
                aliases=["echo"],
            ),
        ]
    )


def _normalize_command(value: str) -> str:
    if not value:
        return ""
    stripped = value.strip().lower()
    if stripped.startswith("/"):
        stripped = stripped[1:]
    return stripped
