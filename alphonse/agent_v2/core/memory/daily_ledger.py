"""Suspended daily Markdown projection of the canonical conversation log.

Existing files may be household-edited and another process may be writing them.
Do not mutate them until an ownership and cross-writer protocol is implemented.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any, Callable


class DailyLedgerProjector:
    """Expose the projection boundary without modifying household files."""

    available = False
    projection_enabled = False

    def __init__(self, *, conversation_store: Any, users_root: Callable[[], str | Path],
                 timezone_provider: Callable[..., str] | None = None,
                 summarizer: Callable[[str], str] | None = None) -> None:
        self._lock = RLock()
        self._users_root = users_root

    def refresh_for_event(self, event: Any) -> Path | None:
        """No projection is written; canonical events remain in SQLite."""
        return None

    def mutation_guard(self):
        """Serialize local callers; this does not fence other processes."""
        return self._lock

    def _daily_root(self, owner: str) -> Path:
        base = Path(self._users_root()).expanduser().resolve()
        current = base
        for part in (_safe_segment(owner), "memory", "daily"):
            current = current / part
            if current.is_symlink():
                raise ValueError("daily_ledger_reconciliation_required")
        if not current.resolve().is_relative_to(base):
            raise ValueError("daily_ledger_reconciliation_required")
        return current

    def invalidate_owner(self, owner: str) -> None:
        """Refuse erasure when a daily directory could contain retained views."""
        with self._lock:
            root = self._daily_root(owner)
            if root.exists():
                raise ValueError("daily_ledger_reconciliation_required")

    def rebuild_owner(self, owner: str) -> list[Path]:
        """Rebuild is unavailable while file ownership cannot be established."""
        raise RuntimeError("daily_ledger_projection_unavailable")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Shared writer for explicit offline migration receipts, not daily views."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


def _safe_segment(value: str) -> str:
    rendered = str(value or "")
    if (not rendered or len(rendered) > 120 or rendered.strip("._") != rendered
            or re.fullmatch(r"[A-Za-z0-9._-]+", rendered) is None):
        raise ValueError("invalid_ledger_member")
    return rendered
