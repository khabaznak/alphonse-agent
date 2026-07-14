"""Per-user, scope-isolated Markdown conversation ledgers."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from alphonse.agent_v2.memory_settings import MemorySettings
from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore


class LedgerMemory:
    """Append-only v2 conversation memory; the latest ledger is the prompt context."""

    def __init__(self, *, users_root: Any, settings_store: SQLiteMemorySettingsStore, summarizer: Callable[[str], str] | None = None) -> None:
        self._users_root = users_root
        self._settings_store = settings_store
        self._summarizer = summarizer

    def start_task(self, task: Any) -> str:
        path = self._current_path(task, rollover=True)
        task_id = str(task.task_id or task.message_id or "task").strip()
        self._append(path, f"\n### Task {task_id}\n- User: {task.user or ''}\n- Project ID: {task.project_id or 'generic'}\n\n#### Conversation\n- User: {task.goal}\n")
        return path.read_text(encoding="utf-8")

    def event(self, task: Any, heading: str, content: Any) -> None:
        path = self._current_path(task, rollover=False)
        text = _render(content)
        self._append(path, f"\n#### {heading}\n{text}\n")

    def finish_task(self, task: Any) -> None:
        outcome = task.outcome if task.outcome is not None else {"status": task.status}
        self.event(task, "Outcome", outcome)

    def latest_content(self, *, user_id: str, project_id: str = "") -> str:
        path = self._latest_path(user_id, project_id)
        return path.read_text(encoding="utf-8") if path is not None else ""

    def ensure_project_scope(self, *, user_id: str, project_id: str) -> Path:
        return self._scope_dir(user_id, project_id)

    def _current_path(self, task: Any, *, rollover: bool) -> Path:
        user_id, project_id = str(task.user or "unknown"), str(task.project_id or "")
        latest = self._latest_path(user_id, project_id)
        if latest is None:
            return self._create_first(user_id, project_id)
        if rollover and latest.stat().st_size >= self._settings_store.get().max_ledger_bytes:
            return self._create_successor(latest, user_id, project_id)
        return latest

    def _scope_dir(self, user_id: str, project_id: str = "") -> Path:
        root = Path(self._users_root()).expanduser().resolve() / str(user_id)
        path = root / "memory" / "generic" if not project_id else root / "projects" / str(project_id) / "memory"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            root = Path(tempfile.gettempdir()) / "alphonse-v2-memory" / str(user_id)
            path = root / "memory" / "generic" if not project_id else root / "projects" / str(project_id) / "memory"
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _latest_path(self, user_id: str, project_id: str = "") -> Path | None:
        files = sorted(self._scope_dir(user_id, project_id).glob("ledger-*.md"))
        return files[-1] if files else None

    def _create_first(self, user_id: str, project_id: str) -> Path:
        path = self._scope_dir(user_id, project_id) / "ledger-0001.md"
        path.write_text("# Memory Ledger\n\n## Memory\n", encoding="utf-8")
        return path

    def _create_successor(self, previous: Path, user_id: str, project_id: str) -> Path:
        sequence = int(previous.stem.rsplit("-", 1)[-1]) + 1
        path = self._scope_dir(user_id, project_id) / f"ledger-{sequence:04d}.md"
        source = previous.read_text(encoding="utf-8")
        generated = ""
        if self._summarizer is not None:
            try: generated = str(self._summarizer(source) or "")
            except Exception: generated = ""
        summary = _summary(generated or source, self._settings_store.get())
        temporary = path.with_suffix(".tmp")
        temporary.write_text(f"# Memory Ledger\n\n## Header\n- Compacted from {previous.name}\n\n## Previous Ledger\n[{previous.name}]({previous.name})\n\n## Compaction Summary\n{summary}\n\n## Memory\n", encoding="utf-8")
        temporary.replace(path)
        return path

    @staticmethod
    def _append(path: Path, content: str) -> None:
        with path.open("a", encoding="utf-8") as handle: handle.write(content)


def _summary(source: str, settings: MemorySettings) -> str:
    words = re.findall(r"\S+", source)
    selected = words[: settings.compaction_summary_max_words]
    return " ".join(selected) or "- (empty ledger)"


def _render(value: Any) -> str:
    if isinstance(value, str): return value
    if isinstance(value, dict): return "\n".join(f"- {key}: {val}" for key, val in value.items())
    if isinstance(value, list): return "\n".join(f"- {item}" for item in value)
    return str(value)
