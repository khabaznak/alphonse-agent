"""Generated per-member daily Markdown views of the canonical conversation log."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo


class DailyLedgerProjector:
    """Materialize daily conversation ledgers from SQLite events."""

    def __init__(
        self,
        *,
        conversation_store: Any,
        users_root: Callable[[], str | Path],
        timezone_provider: Callable[[], str] | None = None,
        summarizer: Callable[[str], str] | None = None,
    ) -> None:
        self._conversations = conversation_store
        self._users_root = users_root
        self._timezone_provider = timezone_provider or (lambda: "UTC")
        self._summarizer = summarizer

    def refresh_for_event(self, event: Any) -> Path | None:
        """Refresh today's file and ensure the previous day's summary exists."""
        try:
            zone = ZoneInfo(str(self._timezone_provider() or "UTC"))
        except Exception:
            zone = ZoneInfo("UTC")
        owner = str(getattr(event, "owner_user_id", "") or "").strip()
        if not owner:
            return None
        try:
            instant = datetime.fromisoformat(str(event.created_at).replace("Z", "+00:00"))
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            instant = datetime.now(timezone.utc)
        local_date = instant.astimezone(zone).date()
        root = Path(self._users_root()).expanduser().resolve() / _safe_segment(owner) / "memory" / "daily"
        previous_date = local_date - timedelta(days=1)
        previous_events = self._events_for_date(owner, previous_date, zone)
        previous_summary = self._summary_for(
            root / f"{previous_date.isoformat()}.md", previous_events, previous_date, zone,
        )
        today_events = self._events_for_date(owner, local_date, zone)
        target = root / f"{local_date.isoformat()}.md"
        self._atomic_write(target, _render_ledger(local_date, previous_summary, today_events, zone))
        return target

    def _events_for_date(self, owner: str, day: date, zone: ZoneInfo) -> list[Any]:
        start = datetime.combine(day, time.min, tzinfo=zone).astimezone(timezone.utc)
        end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=zone).astimezone(timezone.utc)
        return self._conversations.list_between(
            owner_user_id=owner,
            start_utc=start.isoformat(),
            end_utc=end.isoformat(),
        )

    def _summary_for(self, path: Path, events: list[Any], day: date, zone: ZoneInfo) -> str:
        if not events:
            return "No conversation was recorded the previous day."
        try:
            summary_path = path.with_suffix(".summary.md")
            if summary_path.exists():
                existing = summary_path.read_text(encoding="utf-8").strip()
                if existing:
                    return existing
        except OSError:
            pass
        transcript_rows = [
            f"{getattr(item, 'role', 'unknown')}: {str(getattr(item, 'content', ''))[:800]}"
            for item in events[-60:]
        ]
        transcript = "\n".join(transcript_rows)[-16000:]
        summary = ""
        if self._summarizer is not None:
            try:
                summary = str(self._summarizer(transcript) or "").strip()
            except Exception:
                summary = ""
        if not summary:
            excerpts = [str(getattr(item, "content", "")).strip() for item in events]
            excerpts = [item for item in excerpts if item]
            summary = "The previous day included: " + "; ".join(item[:180] for item in excerpts[:5])
            if len(excerpts) > 5:
                summary += f"; and {len(excerpts) - 5} more conversation entries"
            summary += "."
        self._atomic_write(path.with_suffix(".summary.md"), summary + "\n")
        if not path.exists():
            self._atomic_write(path, _render_ledger(day, "Summary narrative will be generated in the next day's ledger.", events, zone))
        return summary

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
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


def _render_ledger(day: date, summary: str, events: list[Any], zone: ZoneInfo) -> str:
    lines = [
        f"# Memory Ledger — {day.isoformat()}",
        "",
        "## Summary narrative",
        summary.strip() or "No summary is available.",
        "",
        "## Day session conversation",
    ]
    for event in events:
        try:
            instant = datetime.fromisoformat(str(event.created_at).replace("Z", "+00:00"))
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=timezone.utc)
            local_time = instant.astimezone(zone).strftime("%H:%M")
        except (TypeError, ValueError):
            local_time = "time unknown"
        role = "Alphonse" if event.role == "assistant" else "Member"
        project = str(getattr(event, "project_id", "") or "").strip()
        suffix = f" · {project}" if project else ""
        lines.extend((f"### {local_time} — {role}{suffix}", str(event.content).strip(), ""))
    if not events:
        lines.extend(("No conversation entries were recorded.", ""))
    return "\n".join(lines).rstrip() + "\n"


def _safe_segment(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
    if not rendered or rendered in {".", ".."}:
        raise ValueError("invalid_ledger_member")
    return rendered[:120]
