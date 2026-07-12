"""Editable daemon-startup configuration for the v2 agent."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from alphonse.agent_v2.core.core import PromptFile

GLOBAL_CONTEXT_FILE = "GlobalContext.md"
PHILOSOPHY_FILE = "Philosophy.md"
EDITABLE_AGENT_CONFIG_FILES = (GLOBAL_CONTEXT_FILE, PHILOSOPHY_FILE)


@dataclass(frozen=True)
class AgentConfigDocument:
    file_name: str
    display_name: str
    content: str
    updated_at: str = ""

    def to_dict(self, *, include_content: bool = True) -> dict[str, str]:
        value = {
            "file_name": self.file_name,
            "display_name": self.display_name,
            "updated_at": self.updated_at,
        }
        if include_content:
            value["content"] = self.content
        return value


class AgentConfigStore:
    """Owns editable local copies of global v2 agent markdown files."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else default_agent_config_dir()
        self._ensure_seeded()

    @classmethod
    def default(cls) -> "AgentConfigStore":
        return cls()

    def list_documents(self) -> list[AgentConfigDocument]:
        return [self.read(file_name) for file_name in EDITABLE_AGENT_CONFIG_FILES]

    def read(self, file_name: str) -> AgentConfigDocument:
        name = _validate_file_name(file_name)
        path = self.config_dir / name
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"agent_config_read_failed: {name}") from exc
        try:
            updated_at = str(path.stat().st_mtime_ns)
        except OSError:
            updated_at = ""
        return AgentConfigDocument(name, _display_name(name), content, updated_at)

    def save(self, file_name: str, content: str) -> AgentConfigDocument:
        name = _validate_file_name(file_name)
        path = self.config_dir / name
        text = str(content)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=self.config_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return self.read(name)

    def _ensure_seeded(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        for file_name in EDITABLE_AGENT_CONFIG_FILES:
            target = self.config_dir / file_name
            if target.exists():
                continue
            source = packaged_agent_config_dir() / file_name
            try:
                content = source.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                content = f"# {_display_name(file_name)}\n"
            self.save(file_name, content)


class AgentConfigPromptLoader:
    """Immutable prompt snapshot loaded once while building a runtime."""

    def __init__(self, documents: dict[str, AgentConfigDocument]) -> None:
        self._documents = dict(documents)

    @classmethod
    def from_store(cls, store: AgentConfigStore) -> "AgentConfigPromptLoader":
        return cls({document.file_name: document for document in store.list_documents()})

    def load(self, name: str) -> PromptFile:
        document = self._documents.get(str(name or "").strip())
        return PromptFile(name=str(name or ""), content=document.content if document is not None else "")


def default_agent_config_dir() -> Path:
    configured = os.getenv("ALPHONSE_V2_AGENT_CONFIG_DIR")
    return Path(configured) if configured else Path.home() / ".alphonse" / "agent-config"


def packaged_agent_config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def _validate_file_name(file_name: str) -> str:
    name = str(file_name or "").strip()
    if name not in EDITABLE_AGENT_CONFIG_FILES:
        raise ValueError(f"agent_config_file_not_allowed: {name}")
    return name


def _display_name(file_name: str) -> str:
    return "Global Context" if file_name == GLOBAL_CONTEXT_FILE else "Philosophy"
