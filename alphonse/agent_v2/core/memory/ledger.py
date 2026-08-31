"""Per-user, scope-isolated Markdown conversation ledgers."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable

from alphonse.agent_v2.memory_settings import MemorySettings
from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore


logger = logging.getLogger(__name__)
_SCOPE_LOCKS: dict[str, RLock] = {}
_SCOPE_LOCKS_GUARD = Lock()


class LedgerMemory:
    """Append-only v2 conversation memory; the latest ledger is the prompt context."""

    def __init__(self, *, users_root: Any, settings_store: SQLiteMemorySettingsStore, summarizer: Callable[[str], str] | None = None, project_root_provider: Callable[[str], str | Path | None] | None = None) -> None:
        self._users_root = users_root
        self._settings_store = settings_store
        self._summarizer = summarizer
        self._project_root_provider = project_root_provider

    def start_task(self, task: Any) -> str:
        with self._scope_lock(str(task.user or "unknown"), str(task.project_id or "")):
            path = self._current_path(task, rollover=True)
            task_id = str(task.task_id or task.message_id or "task").strip()
            self._append(path, f"\n### Task {task_id}\n- User: {task.user or ''}\n- Project ID: {task.project_id or 'generic'}\n\n#### Conversation\n- User: {task.goal}\n")
            return path.read_text(encoding="utf-8")

    def event(self, task: Any, heading: str, content: Any) -> None:
        with self._scope_lock(str(task.user or "unknown"), str(task.project_id or "")):
            path = self._current_path(task, rollover=False)
            text = _render(content)
            self._append(path, f"\n#### {heading}\n{text}\n")

    def finish_task(self, task: Any) -> None:
        outcome = task.outcome if task.outcome is not None else {"status": task.status}
        self.event(task, "Outcome", outcome)

    def latest_content(self, *, user_id: str, project_id: str = "") -> str:
        with self._scope_lock(user_id, project_id):
            path = self._latest_path(user_id, project_id)
            return path.read_text(encoding="utf-8") if path is not None else ""

    def ensure_project_scope(self, *, user_id: str, project_id: str) -> Path:
        return self._scope_dir(user_id, project_id)

    def migrate_legacy_project_ledgers(self, project_id: str, *, include_generic: bool = False) -> bool:
        """Merge pre-shared per-user ledgers into the canonical project ledger once."""
        project = str(project_id or "").strip()
        if not project or self._project_root_provider is None:
            return False
        root = self._project_root_provider(project)
        if not root:
            return False
        target = Path(root).expanduser().resolve() / ".alphonse" / "memory"
        marker = target / ".legacy-ledgers-migrated"
        if marker.exists():
            return False
        target.mkdir(parents=True, exist_ok=True)
        ledger = target / "ledger-0001.md"
        if not ledger.exists():
            ledger.write_text("# Memory Ledger\n\n## Memory\n", encoding="utf-8")
        legacy_root = Path(self._users_root()).expanduser().resolve()
        sections: list[str] = []
        for source in sorted(legacy_root.glob(f"*/projects/{project}/memory/ledger-*.md")):
            try:
                sections.append(f"\n## Migrated legacy ledger: {source.parent.parent.parent.name}/{source.name}\n{source.read_text(encoding='utf-8')}\n")
            except OSError:
                continue
        if include_generic:
            for source in sorted(legacy_root.glob("*/memory/generic/ledger-*.md")):
                try:
                    sections.append(f"\n## Migrated legacy Home ledger: {source.parent.parent.name}/{source.name}\n{source.read_text(encoding='utf-8')}\n")
                except OSError:
                    continue
        if sections:
            self._append(ledger, "".join(sections))
        marker.write_text("migrated\n", encoding="utf-8")
        return bool(sections)

    def _current_path(self, task: Any, *, rollover: bool) -> Path:
        user_id, project_id = str(task.user or "unknown"), str(task.project_id or "")
        latest = self._latest_path(user_id, project_id)
        if latest is None:
            return self._create_first(user_id, project_id)
        if rollover and latest.stat().st_size >= self._settings_store.get().max_ledger_bytes:
            return self._create_successor(latest, user_id, project_id)
        return latest

    def _scope_dir(self, user_id: str, project_id: str = "") -> Path:
        if project_id and self._project_root_provider is not None:
            root = self._project_root_provider(project_id)
            if root:
                path = Path(root).expanduser().resolve() / ".alphonse" / "memory"
                path.mkdir(parents=True, exist_ok=True)
                return path
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
        logger.info(
            "memory ledger rolled over user_id=%s project_id=%s previous=%s successor=%s previous_bytes=%s",
            user_id,
            project_id,
            previous.name,
            path.name,
            previous.stat().st_size,
        )
        return path

    def _scope_lock(self, user_id: str, project_id: str) -> RLock:
        # Project-backed scopes are shared across collaborators, so the lock key
        # must follow the resolved ledger directory instead of the initiating user.
        key = str(self._scope_dir(user_id, project_id).resolve())
        with _SCOPE_LOCKS_GUARD:
            lock = _SCOPE_LOCKS.get(key)
            if lock is None:
                lock = RLock()
                _SCOPE_LOCKS[key] = lock
            return lock

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
