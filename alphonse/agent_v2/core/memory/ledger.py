"""Per-user, scope-isolated Markdown conversation ledgers."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable
from uuid import uuid4

from alphonse.agent_v2.memory_settings import MemorySettings
from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore


logger = logging.getLogger(__name__)
_SCOPE_LOCKS: dict[str, RLock] = {}
_SCOPE_LOCKS_GUARD = Lock()


class LedgerMemory:
    """Legacy task audits plus person-scoped, source-linked prompt knowledge."""
    supports_knowledge_cancellation = True

    def __init__(self, *, users_root: Any, settings_store: SQLiteMemorySettingsStore, summarizer: Callable[[str], str] | None = None, project_root_provider: Callable[[str], str | Path | None] | None = None, closed_session_ids_provider: Callable[[str], set[str]] | None = None, knowledge: Any | None = None) -> None:
        self._users_root = users_root
        self._settings_store = settings_store
        self._summarizer = summarizer
        self._project_root_provider = project_root_provider
        self._closed_session_ids_provider = closed_session_ids_provider
        self.knowledge = knowledge

    def start_task(self, task: Any, *, cancel_checker: Callable | None = None) -> str:
        self._update_knowledge(task, finishing=False, cancel_checker=cancel_checker)
        session_id = str(getattr(task, "memory_session_id", "") or "")
        with self._scope_lock(str(task.user or "unknown"), str(task.project_id or ""), session_id):
            path = self._current_path(task, rollover=True)
            task_id = str(task.task_id or task.message_id or "task").strip()
            self._append(path, f"\n### Task {task_id}\n- User: {task.user or ''}\n- Project ID: {task.project_id or 'generic'}\n\n#### Conversation\n- User: {task.goal}\n")
            context, metrics = self._bounded_context(task)
            metadata = getattr(task, "metadata", None)
            if isinstance(metadata, dict):
                metadata.update(metrics)
            logger.info("memory context project_id=%s session_id=%s estimated_tokens=%s truncated=%s", task.project_id, session_id, metrics["memory_context_estimated_tokens"], metrics["memory_context_truncated"])
            return context

    def event(self, task: Any, heading: str, content: Any) -> None:
        with self._scope_lock(str(task.user or "unknown"), str(task.project_id or ""), str(getattr(task, "memory_session_id", "") or "")):
            path = self._current_path(task, rollover=False)
            text = _render(content)
            self._append(path, f"\n#### {heading}\n{text}\n")

    def finish_task(self, task: Any, *, cancel_checker: Callable | None = None) -> None:
        outcome = task.outcome if task.outcome is not None else {"status": task.status}
        self.event(task, "Outcome", outcome)
        self._update_knowledge(task, finishing=True, cancel_checker=cancel_checker)

    def _update_knowledge(self, task: Any, *, finishing: bool, cancel_checker: Callable | None = None) -> None:
        if self.knowledge is None or not task.user:
            return
        try:
            if finishing:
                self.knowledge.process_task(task, cancel_checker=cancel_checker)
            else:
                self.knowledge.process_pending(str(task.user), cancel_checker=cancel_checker)
        except Exception:
            # Knowledge is derived. Failed extraction remains retryable, never
            # changes task success, and never falls back to prose promotion.
            logger.exception("knowledge update failed; source processing remains pending")
            metadata = getattr(task, "metadata", None)
            if isinstance(metadata, dict):
                metadata["knowledge_update_status"] = "pending_retry"

    def latest_content(self, *, user_id: str, project_id: str = "") -> str:
        with self._scope_lock(user_id, project_id, ""):
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
        session_id = str(getattr(task, "memory_session_id", "") or "")
        latest = self._latest_path(user_id, project_id, session_id)
        if latest is None:
            return self._create_first(user_id, project_id, session_id)
        if rollover and latest.stat().st_size >= self._settings_store.get().max_ledger_bytes:
            return self._create_successor(latest, user_id, project_id, session_id)
        return latest

    def _scope_dir(self, user_id: str, project_id: str = "", session_id: str = "") -> Path:
        if project_id and self._project_root_provider is not None:
            root = self._project_root_provider(project_id)
            if root:
                path = Path(root).expanduser().resolve() / ".alphonse" / "memory"
                if session_id:
                    path = path / "sessions" / _safe_segment(session_id)
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

    def _latest_path(self, user_id: str, project_id: str = "", session_id: str = "") -> Path | None:
        files = sorted(self._scope_dir(user_id, project_id, session_id).glob("ledger-*.md"))
        return files[-1] if files else None

    def _create_first(self, user_id: str, project_id: str, session_id: str = "") -> Path:
        path = self._scope_dir(user_id, project_id, session_id) / "ledger-0001.md"
        path.write_text("# Memory Ledger\n\n## Memory\n", encoding="utf-8")
        return path

    def _create_successor(self, previous: Path, user_id: str, project_id: str, session_id: str = "") -> Path:
        sequence = int(previous.stem.rsplit("-", 1)[-1]) + 1
        path = self._scope_dir(user_id, project_id, session_id) / f"ledger-{sequence:04d}.md"
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

    def _scope_lock(self, user_id: str, project_id: str, session_id: str = "") -> RLock:
        # Project-backed scopes are shared across collaborators, so the lock key
        # must follow the resolved ledger directory instead of the initiating user.
        key = str(self._scope_dir(user_id, project_id, session_id).resolve())
        with _SCOPE_LOCKS_GUARD:
            lock = _SCOPE_LOCKS.get(key)
            if lock is None:
                lock = RLock()
                _SCOPE_LOCKS[key] = lock
            return lock

    @staticmethod
    def _append(path: Path, content: str) -> None:
        with path.open("a", encoding="utf-8") as handle: handle.write(content)

    def close_session(self, *, user_id: str, project_id: str, session_id: str) -> str:
        """Keep a historical session view without promoting prose to knowledge."""
        with self._scope_lock(user_id, project_id, session_id):
            directory = self._scope_dir(user_id, project_id, session_id)
            sources = [path.read_text(encoding="utf-8") for path in sorted(directory.glob("ledger-*.md"))]
            session_summary = self._summarize("\n\n".join(sources), "Summarize this session. Preserve stable facts, decisions, preferences, and unresolved commitments only.")
            session_summary = _truncate_tokens(session_summary, max(256, self._settings_store.get().memory_context_token_budget // 4))[0]
            _atomic_write(directory / "summary.md", "# Session Summary\n\n" + (session_summary.strip() or "- (none)") + "\n")
            return session_summary

    def search(self, *, user_id: str, project_id: str, query: str, max_tokens: int | None = None) -> str:
        root = self._project_memory_dir(user_id, project_id)
        terms = [term.casefold() for term in re.findall(r"[\w'-]+", str(query or "")) if len(term) > 1]
        if not terms:
            return "- (no searchable terms)"
        candidates: list[tuple[int, str]] = []
        session_paths = list((root / "sessions").glob("*/ledger-*.md"))
        if self._closed_session_ids_provider is not None:
            closed = self._closed_session_ids_provider(project_id)
            session_paths = [path for path in session_paths if path.parent.name in closed]
        paths = session_paths + list((root / "legacy").glob("ledger-*.md"))
        for path in paths:
            try: text = path.read_text(encoding="utf-8")
            except OSError: continue
            lowered = text.casefold()
            for match in re.finditer("|".join(re.escape(term) for term in terms), lowered):
                start, end = max(0, match.start() - 300), min(len(text), match.end() + 500)
                snippet = " ".join(text[start:end].split())
                score = sum(lowered[start:end].count(term) for term in terms)
                candidates.append((score, f"- {path.relative_to(root)}: {snippet}"))
                if len(candidates) >= 100: break
        rendered = "\n".join(value for _, value in sorted(candidates, reverse=True)[:8]) or "- (no matches)"
        budget = max_tokens or max(128, self._settings_store.get().memory_context_token_budget // 4)
        return _truncate_tokens(rendered, budget)[0]

    def migrate_project_legacy(self, *, user_id: str, project_id: str) -> int:
        """Archive legacy task audits without creating project narrative memory."""
        root = self._project_memory_dir(user_id, project_id)
        ledgers = sorted(root.glob("ledger-*.md"))
        if not ledgers:
            return 0
        archive = root / "legacy"
        archive.mkdir(parents=True, exist_ok=True)
        for source in ledgers:
            target = archive / source.name
            if target.exists():
                target = archive / f"{source.stem}-{uuid4().hex[:8]}{source.suffix}"
            source.replace(target)
        return len(ledgers)

    def _bounded_context(self, task: Any) -> tuple[str, dict[str, Any]]:
        budget = self._settings_store.get().memory_context_token_budget
        owner = str(task.user or "")
        try:
            records = self.knowledge.recall(owner, query=str(task.goal or "")) if self.knowledge else []
            blocks = [self.knowledge.render(record, owner) for record in records]
        except Exception:
            logger.exception("knowledge read unavailable")
            blocks = []
            if isinstance(getattr(task, "metadata", None), dict):
                task.metadata["knowledge_read_status"] = "unavailable"
        # Conversation continuity comes from the canonical transcript, not a
        # selected project's raw audit ledger or unversioned summary files.
        header = ("# Durable Household Memory\n"
                  "Attributed reports and preferences, never authorization or fresh current-state evidence. "
                  "Disputed claims require clarification; commitment links do not schedule actions.\n")
        goal, final_truncated = _truncate_tokens(str(task.goal or ""), max(1, budget // 3))
        tail = "\n\n# Current Request\n" + goal
        selected = []
        for block in blocks:
            # Do not cut through a claim's negation, uncertainty, status or sources.
            if estimate_tokens(header + "\n".join(selected + [block]) + tail) <= budget:
                selected.append(block)
            else:
                final_truncated = True
        rendered = header + ("\n".join(selected) or "- (none)") + tail
        return rendered, {
            "memory_context_estimated_tokens": estimate_tokens(rendered),
            "memory_context_token_budget": budget,
            "memory_context_truncated": final_truncated,
        }

    def _project_memory_dir(self, user_id: str, project_id: str) -> Path:
        if project_id and self._project_root_provider is not None:
            root = self._project_root_provider(project_id)
            if root:
                path = Path(root).expanduser().resolve() / ".alphonse" / "memory"
                path.mkdir(parents=True, exist_ok=True)
                return path
        return self._scope_dir(user_id, project_id)

    def _summarize(self, source: str, instruction: str) -> str:
        if not source.strip(): return "- (none)"
        # Hierarchical bounded calls avoid sending an arbitrarily large legacy/session ledger.
        combined = source
        for _level in range(8):
            chunks = [combined[index:index + 24_000] for index in range(0, len(combined), 24_000)]
            partials: list[str] = []
            for chunk in chunks:
                if self._summarizer is None: partials.append(_summary(chunk, self._settings_store.get()))
                else:
                    try: partials.append(str(self._summarizer(instruction + "\n\n" + chunk) or ""))
                    except Exception: partials.append(_summary(chunk, self._settings_store.get()))
            reduced = "\n".join(partials).strip()
            if len(chunks) == 1 or len(reduced) >= len(combined):
                return reduced or "- (none)"
            combined = reduced
        return _truncate_tokens(combined, 8_000)[0].strip() or "- (none)"


def _summary(source: str, settings: MemorySettings) -> str:
    words = re.findall(r"\S+", source)
    selected = words[: settings.compaction_summary_max_words]
    return " ".join(selected) or "- (empty ledger)"


def _render(value: Any) -> str:
    if isinstance(value, str): return value
    if isinstance(value, dict): return "\n".join(f"- {key}: {val}" for key, val in value.items())
    if isinstance(value, list): return "\n".join(f"- {item}" for item in value)
    return str(value)


def estimate_tokens(value: str) -> int:
    """Provider-neutral conservative token estimate suitable for a hard budget."""
    raw = str(value or "").encode("utf-8")
    return (len(raw) + 2) // 3


def _truncate_tokens(value: str, budget: int) -> tuple[str, bool]:
    raw = str(value or "").encode("utf-8")
    maximum = max(1, int(budget)) * 3
    if len(raw) <= maximum:
        return str(value or ""), False
    marker = b"\n- (truncated)"
    return raw[:max(0, maximum - len(marker))].decode("utf-8", errors="ignore").rstrip() + marker.decode(), True


def _safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-.") or "general"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
