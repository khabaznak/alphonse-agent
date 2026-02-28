from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from alphonse.agent.cognition.memory.paths import resolve_memory_root
from alphonse.agent.cognition.memory.paths import sanitize_segment
from alphonse.agent.cognition.memory.paths import user_root

_DEFAULT_MAX_LINES = 20_000
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_DAILY_RETENTION_DAYS = 30
_DEFAULT_WEEKLY_RETENTION_DAYS = 90
_ISO_DATE_FMT = "%Y-%m-%d"
_EPISODE_TS_FMT = "%Y-%m-%d %H:%M:%S %z"


@dataclass(frozen=True)
class TimeRange:
    start: datetime | None = None
    end: datetime | None = None


class MemoryService:
    def __init__(
        self,
        *,
        root_dir: Path | None = None,
        max_lines_per_file: int = _DEFAULT_MAX_LINES,
        max_bytes_per_file: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self._root = root_dir.resolve() if isinstance(root_dir, Path) else resolve_memory_root()
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_lines = max(1, int(max_lines_per_file))
        self._max_bytes = max(1024, int(max_bytes_per_file))

    def append_episode(
        self,
        user_id: str,
        mission_id: str,
        event_type: str,
        payload: dict[str, Any],
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now().astimezone()
        episodic_dir = self._episodic_month_dir(user_id=user_id, when=now)
        episode_file = self._pick_episode_file(episodic_dir=episodic_dir, when=now)
        artifact_rows = [dict(item) for item in (artifacts or []) if isinstance(item, dict)]
        rendered = _render_episode_entry(
            timestamp=now,
            mission_id=str(mission_id or "").strip() or "unknown_mission",
            event_type=str(event_type or "").strip() or "event",
            payload=dict(payload or {}),
            artifacts=artifact_rows,
        )
        with episode_file.open("a", encoding="utf-8") as handle:
            handle.write(rendered)
        return {
            "path": str(episode_file),
            "timestamp": now.isoformat(),
            "mission_id": str(mission_id or "").strip(),
            "event_type": str(event_type or "").strip(),
        }

    def put_artifact(
        self,
        mission_id: str,
        content: bytes | dict[str, Any] | str,
        mime: str,
        name_hint: str,
    ) -> dict[str, Any]:
        mission_key = str(mission_id or "").strip() or "unknown_mission"
        owner_user = self._lookup_user_for_mission(mission_key)
        owner_segment = sanitize_segment(owner_user) if owner_user else "_unscoped"
        now = datetime.now(timezone.utc)
        ext = _guess_extension(mime=str(mime or ""), name_hint=str(name_hint or "artifact"))
        artifact_name = (
            f"{now.strftime('%Y%m%d_%H%M%S')}_{sanitize_segment(name_hint) or 'artifact'}{ext}"
        )
        artifact_dir = self._root / owner_segment / "artifacts" / sanitize_segment(mission_key)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / artifact_name
        raw = _to_bytes(content=content, mime=mime)
        artifact_path.write_bytes(raw)
        sha256 = hashlib.sha256(raw).hexdigest()
        ref = {
            "artifact_id": f"{sanitize_segment(mission_key)}:{artifact_name}",
            "mission_id": mission_key,
            "user_id": owner_user,
            "path": str(artifact_path),
            "relative_path": str(artifact_path.relative_to(self._root)),
            "mime": str(mime or "application/octet-stream"),
            "size_bytes": len(raw),
            "sha256": sha256,
            "created_at": now.isoformat(),
        }
        return ref

    def upsert_workspace_pointer(self, user_id: str, key: str, value: Any) -> dict[str, Any]:
        data = self._read_json(self._workspace_pointer_path(user_id), default={})
        pointers = data if isinstance(data, dict) else {}
        pointers[str(key)] = value
        self._write_json(self._workspace_pointer_path(user_id), pointers)
        return {"user_id": user_id, "key": key, "value": value}

    def mission_upsert(self, user_id: str, mission_id: str, **fields: Any) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        path = self._mission_path(user_id=user_id, mission_id=mission_id)
        existing = self._read_json(path, default={})
        mission = existing if isinstance(existing, dict) else {}
        mission.setdefault("mission_id", mission_id)
        mission.setdefault("user_id", user_id)
        mission.setdefault("created_at", now)
        mission.setdefault("steps", [])
        mission.update({k: v for k, v in fields.items() if k not in {"steps", "mission_id", "user_id"}})
        mission["updated_at"] = now
        self._write_json(path, mission)
        self._link_mission_to_user(user_id=user_id, mission_id=mission_id)
        return mission

    def mission_step_update(
        self,
        user_id: str,
        mission_id: str,
        step_id: str,
        status: str,
        **fields: Any,
    ) -> dict[str, Any]:
        mission = self.get_mission(user_id=user_id, mission_id=mission_id) or self.mission_upsert(
            user_id=user_id,
            mission_id=mission_id,
        )
        steps = mission.get("steps") if isinstance(mission.get("steps"), list) else []
        found = None
        for item in steps:
            if not isinstance(item, dict):
                continue
            if str(item.get("step_id") or "").strip() == str(step_id):
                found = item
                break
        now = datetime.now(timezone.utc).isoformat()
        if found is None:
            found = {"step_id": step_id, "created_at": now}
            steps.append(found)
        found["status"] = status
        found["updated_at"] = now
        for key, value in fields.items():
            found[key] = value
        mission["steps"] = steps
        mission["updated_at"] = now
        self._write_json(self._mission_path(user_id=user_id, mission_id=mission_id), mission)
        return mission

    def search_episodes(
        self,
        user_id: str,
        query: str,
        mission_id: str | None = None,
        time_range: TimeRange | tuple[str | datetime | None, str | datetime | None] | None = None,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        base_dir = self._user_episodic_root(user_id=user_id)
        if not base_dir.exists():
            return []
        q = str(query or "").strip()
        if not q:
            return []
        rg = shutil.which("rg")
        if not rg:
            return self._search_fallback(
                base_dir=base_dir,
                query=q,
                mission_id=mission_id,
                time_range=time_range,
                limit=limit,
            )
        cmd = [
            rg,
            "--line-number",
            "--no-heading",
            "--with-filename",
            "--color",
            "never",
            q,
            str(base_dir),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        except Exception:
            return self._search_fallback(
                base_dir=base_dir,
                query=q,
                mission_id=mission_id,
                time_range=time_range,
                limit=limit,
            )
        if proc.returncode not in {0, 1}:
            return []
        mission_filter = str(mission_id or "").strip()
        normalized_range = _coerce_time_range(time_range)
        rows: list[dict[str, Any]] = []
        for line in (proc.stdout or "").splitlines():
            parsed = _parse_rg_line(line)
            if not parsed:
                continue
            file_path = Path(parsed["path"])
            if mission_filter and not _file_contains_mission(file_path, mission_filter):
                continue
            if normalized_range and not _file_in_time_range(file_path=file_path, time_range=normalized_range):
                continue
            rows.append(parsed)
            if len(rows) >= max(1, int(limit)):
                break
        return rows

    def get_mission(self, user_id: str, mission_id: str) -> dict[str, Any] | None:
        payload = self._read_json(self._mission_path(user_id=user_id, mission_id=mission_id), default=None)
        return payload if isinstance(payload, dict) else None

    def list_active_missions(self, user_id: str) -> list[dict[str, Any]]:
        root = self._missions_dir(user_id=user_id)
        if not root.exists():
            return []
        active: list[dict[str, Any]] = []
        for file in sorted(root.glob("*.json")):
            payload = self._read_json(file, default=None)
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status") or "active").strip().lower()
            if status in {"active", "running", "pending"}:
                active.append(payload)
        return active

    def get_workspace_pointer(self, user_id: str, key: str) -> Any:
        payload = self._read_json(self._workspace_pointer_path(user_id), default={})
        if not isinstance(payload, dict):
            return None
        return payload.get(str(key))

    def generate_weekly_summary(
        self,
        user_id: str,
        *,
        reference: datetime | None = None,
    ) -> dict[str, Any] | None:
        now = (reference or datetime.now()).astimezone()
        week_start = (now - timedelta(days=now.weekday())).date()
        week_end = week_start + timedelta(days=6)
        files = self._episodic_files_for_range(
            user_id=user_id,
            start=week_start,
            end=week_end,
        )
        if not files:
            return None
        mission_counts: dict[str, int] = {}
        event_counts: dict[str, int] = {}
        total_entries = 0
        for file in files:
            for line in _read_lines(file):
                if not line.startswith("### "):
                    continue
                total_entries += 1
                mission = _extract_between(line, "| mission:", "| event:")
                event = _extract_after(line, "| event:")
                if mission:
                    mission_counts[mission] = mission_counts.get(mission, 0) + 1
                if event:
                    event_counts[event] = event_counts.get(event, 0) + 1
        summary = self._weekly_summary_path(user_id=user_id, week_start=week_start)
        summary.parent.mkdir(parents=True, exist_ok=True)
        rendered = _render_weekly_summary_markdown(
            user_id=user_id,
            week_start=week_start,
            week_end=week_end,
            total_entries=total_entries,
            mission_counts=mission_counts,
            event_counts=event_counts,
            files=files,
        )
        summary.write_text(rendered, encoding="utf-8")
        return {"path": str(summary), "entries": total_entries}

    def apply_retention(
        self,
        *,
        user_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, int]:
        current = (now or datetime.now()).astimezone()
        users = [user_id] if user_id else self._list_users()
        daily_days = _env_int("ALPHONSE_MEMORY_DAILY_RETENTION_DAYS", _DEFAULT_DAILY_RETENTION_DAYS)
        weekly_days = _env_int("ALPHONSE_MEMORY_WEEKLY_RETENTION_DAYS", _DEFAULT_WEEKLY_RETENTION_DAYS)
        deleted_daily = 0
        deleted_weekly = 0
        for uid in users:
            if daily_days > 0:
                daily_cutoff = current.date() - timedelta(days=daily_days)
                for file in self._all_episodic_files(user_id=uid):
                    file_date = _date_from_episode_filename(file.name)
                    if file_date and file_date < daily_cutoff:
                        file.unlink(missing_ok=True)
                        deleted_daily += 1
            if weekly_days > 0:
                weekly_cutoff = current.date() - timedelta(days=weekly_days)
                for file in self._all_weekly_summary_files(user_id=uid):
                    week_start = _week_start_from_summary_filename(file.name)
                    if week_start and week_start < weekly_cutoff:
                        file.unlink(missing_ok=True)
                        deleted_weekly += 1
        return {"deleted_daily": deleted_daily, "deleted_weekly": deleted_weekly}

    def run_maintenance(
        self,
        *,
        user_id: str | None = None,
        now: datetime | None = None,
        generate_weekly: bool = True,
    ) -> dict[str, Any]:
        current = (now or datetime.now()).astimezone()
        users = [user_id] if user_id else self._list_users()
        summaries_written = 0
        if generate_weekly and current.weekday() == 0:
            for uid in users:
                result = self.generate_weekly_summary(user_id=uid, reference=current - timedelta(days=1))
                if result:
                    summaries_written += 1
        retention = self.apply_retention(user_id=user_id, now=current)
        return {
            "users_scanned": len(users),
            "summaries_written": summaries_written,
            **retention,
        }

    def _pick_episode_file(self, *, episodic_dir: Path, when: datetime) -> Path:
        day = when.strftime(_ISO_DATE_FMT)
        candidates = sorted(episodic_dir.glob(f"{day}*.md"))
        if not candidates:
            return episodic_dir / f"{day}.md"
        for file in reversed(candidates):
            if _line_count(file) < self._max_lines and file.stat().st_size < self._max_bytes:
                return file
        part = len(candidates) + 1
        return episodic_dir / f"{day}__part{part:02d}.md"

    def _episodic_month_dir(self, *, user_id: str, when: datetime) -> Path:
        root = self._user_episodic_root(user_id=user_id)
        path = root / f"{when.year:04d}" / f"{when.month:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _user_episodic_root(self, *, user_id: str) -> Path:
        path = user_root(user_id=user_id, root=self._root) / "episodic"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _missions_dir(self, *, user_id: str) -> Path:
        path = user_root(user_id=user_id, root=self._root) / "missions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _mission_path(self, *, user_id: str, mission_id: str) -> Path:
        return self._missions_dir(user_id=user_id) / f"{sanitize_segment(mission_id)}.json"

    def _workspace_pointer_path(self, user_id: str) -> Path:
        return user_root(user_id=user_id, root=self._root) / "workspace_pointers.json"

    def _weekly_summary_path(self, *, user_id: str, week_start: datetime.date) -> Path:
        iso_year, iso_week, _ = week_start.isocalendar()
        path = user_root(user_id=user_id, root=self._root) / "summaries" / "weekly" / f"{iso_year:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"week_{week_start.isoformat()}_W{iso_week:02d}.md"

    def _all_episodic_files(self, *, user_id: str) -> list[Path]:
        root = self._user_episodic_root(user_id=user_id)
        return sorted(root.glob("*/*/*.md"))

    def _all_weekly_summary_files(self, *, user_id: str) -> list[Path]:
        root = user_root(user_id=user_id, root=self._root) / "summaries" / "weekly"
        if not root.exists():
            return []
        return sorted(root.glob("*/*.md"))

    def _episodic_files_for_range(
        self,
        *,
        user_id: str,
        start: datetime.date,
        end: datetime.date,
    ) -> list[Path]:
        rows: list[Path] = []
        for file in self._all_episodic_files(user_id=user_id):
            day = _date_from_episode_filename(file.name)
            if not day:
                continue
            if start <= day <= end:
                rows.append(file)
        return rows

    def _search_fallback(
        self,
        *,
        base_dir: Path,
        query: str,
        mission_id: str | None,
        time_range: TimeRange | tuple[str | datetime | None, str | datetime | None] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        needle = query.lower()
        mission_filter = str(mission_id or "").strip()
        normalized_range = _coerce_time_range(time_range)
        for file in sorted(base_dir.glob("*/*/*.md")):
            if mission_filter and not _file_contains_mission(file, mission_filter):
                continue
            if normalized_range and not _file_in_time_range(file_path=file, time_range=normalized_range):
                continue
            for idx, text in enumerate(_read_lines(file), start=1):
                if needle in text.lower():
                    rows.append(
                        {
                            "path": str(file),
                            "line": idx,
                            "text": text.strip(),
                        }
                    )
                    if len(rows) >= max(1, int(limit)):
                        return rows
        return rows

    def _list_users(self) -> list[str]:
        users: list[str] = []
        for item in self._root.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                users.append(item.name)
        return sorted(users)

    def _mission_index_path(self) -> Path:
        index_dir = self._root / "_indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        return index_dir / "mission_to_user.json"

    def _link_mission_to_user(self, *, user_id: str, mission_id: str) -> None:
        payload = self._read_json(self._mission_index_path(), default={})
        table = payload if isinstance(payload, dict) else {}
        table[str(mission_id)] = str(user_id)
        self._write_json(self._mission_index_path(), table)

    def _lookup_user_for_mission(self, mission_id: str) -> str | None:
        payload = self._read_json(self._mission_index_path(), default={})
        if not isinstance(payload, dict):
            return None
        user_id = payload.get(str(mission_id))
        if isinstance(user_id, str) and user_id.strip():
            return user_id.strip()
        return None

    def _read_json(self, path: Path, *, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_episode(
    user_id: str,
    mission_id: str,
    event_type: str,
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return MemoryService().append_episode(user_id, mission_id, event_type, payload, artifacts=artifacts)


def put_artifact(
    mission_id: str,
    content: bytes | dict[str, Any] | str,
    mime: str,
    name_hint: str,
) -> dict[str, Any]:
    return MemoryService().put_artifact(mission_id, content, mime, name_hint)


def upsert_workspace_pointer(user_id: str, key: str, value: Any) -> dict[str, Any]:
    return MemoryService().upsert_workspace_pointer(user_id, key, value)


def mission_upsert(user_id: str, mission_id: str, **fields: Any) -> dict[str, Any]:
    return MemoryService().mission_upsert(user_id, mission_id, **fields)


def mission_step_update(
    user_id: str,
    mission_id: str,
    step_id: str,
    status: str,
    **fields: Any,
) -> dict[str, Any]:
    return MemoryService().mission_step_update(user_id, mission_id, step_id, status, **fields)


def search_episodes(
    user_id: str,
    query: str,
    mission_id: str | None = None,
    time_range: TimeRange | tuple[str | datetime | None, str | datetime | None] | None = None,
) -> list[dict[str, Any]]:
    return MemoryService().search_episodes(user_id, query, mission_id=mission_id, time_range=time_range)


def get_mission(user_id: str, mission_id: str) -> dict[str, Any] | None:
    return MemoryService().get_mission(user_id, mission_id)


def list_active_missions(user_id: str) -> list[dict[str, Any]]:
    return MemoryService().list_active_missions(user_id)


def get_workspace_pointer(user_id: str, key: str) -> Any:
    return MemoryService().get_workspace_pointer(user_id, key)


def _render_episode_entry(
    *,
    timestamp: datetime,
    mission_id: str,
    event_type: str,
    payload: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> str:
    heading = f"### {timestamp.strftime(_EPISODE_TS_FMT)} | mission: {mission_id} | event: {event_type}"
    lines = [heading]
    merged = dict(payload or {})
    if artifacts:
        commit = merged.get("commit")
        if not isinstance(commit, dict):
            commit = {}
        existing_artifacts = commit.get("artifacts")
        artifact_rows = list(existing_artifacts) if isinstance(existing_artifacts, list) else []
        artifact_rows.extend(artifacts)
        commit["artifacts"] = artifact_rows
        merged["commit"] = commit
    for key, value in merged.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        lines.extend(_render_field(key=key_text, value=value, indent=0))
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_field(*, key: str, value: Any, indent: int) -> list[str]:
    pad = "  " * indent
    if isinstance(value, dict):
        lines = [f"{pad}- {key}:"]
        for child_key, child_value in value.items():
            lines.extend(_render_field(key=str(child_key), value=child_value, indent=indent + 1))
        return lines
    if isinstance(value, list):
        lines = [f"{pad}- {key}:"]
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{pad}  -")
                for child_key, child_value in item.items():
                    child_lines = _render_field(key=str(child_key), value=child_value, indent=indent + 2)
                    lines.extend(child_lines)
            elif isinstance(item, list):
                lines.append(f"{pad}  - {json.dumps(item, ensure_ascii=False)}")
            else:
                lines.append(f"{pad}  - {item}")
        return lines
    return [f"{pad}- {key}: {value}"]


def _guess_extension(*, mime: str, name_hint: str) -> str:
    hint = str(name_hint or "").strip()
    if "." in hint:
        ext = "." + hint.split(".")[-1]
        if 1 <= len(ext) <= 10:
            return ext.lower()
    lower = str(mime or "").strip().lower()
    mapping = {
        "application/json": ".json",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "application/pdf": ".pdf",
    }
    return mapping.get(lower, ".bin")


def _to_bytes(*, content: bytes | dict[str, Any] | str, mime: str) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, dict):
        return (json.dumps(content, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    text = str(content or "")
    if str(mime or "").strip().lower() == "application/base64":
        try:
            return base64.b64decode(text.encode("utf-8"), validate=True)
        except Exception:
            return text.encode("utf-8")
    return text.encode("utf-8")


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return 0


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []


def _parse_rg_line(line: str) -> dict[str, Any] | None:
    parts = line.split(":", 2)
    if len(parts) != 3:
        return None
    path, line_no, text = parts
    try:
        num = int(line_no)
    except ValueError:
        return None
    return {"path": path, "line": num, "text": text}


def _file_contains_mission(path: Path, mission_id: str) -> bool:
    needle = f"| mission: {mission_id} |"
    try:
        return needle in path.read_text(encoding="utf-8")
    except Exception:
        return False


def _date_from_episode_filename(filename: str) -> datetime.date | None:
    text = str(filename or "").strip()
    if len(text) < 10:
        return None
    chunk = text[:10]
    try:
        return datetime.strptime(chunk, _ISO_DATE_FMT).date()
    except ValueError:
        return None


def _coerce_time_range(
    value: TimeRange | tuple[str | datetime | None, str | datetime | None] | None,
) -> TimeRange | None:
    if value is None:
        return None
    if isinstance(value, TimeRange):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        start = _coerce_datetime(value[0])
        end = _coerce_datetime(value[1])
        return TimeRange(start=start, end=end)
    return None


def _coerce_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _file_in_time_range(*, file_path: Path, time_range: TimeRange) -> bool:
    file_date = _date_from_episode_filename(file_path.name)
    if not file_date:
        return False
    if time_range.start and file_date < time_range.start.astimezone(timezone.utc).date():
        return False
    if time_range.end and file_date > time_range.end.astimezone(timezone.utc).date():
        return False
    return True


def _extract_between(text: str, left: str, right: str) -> str:
    start = text.find(left)
    if start < 0:
        return ""
    start += len(left)
    end = text.find(right, start)
    if end < 0:
        return ""
    return text[start:end].strip()


def _extract_after(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        return ""
    return text[start + len(marker) :].strip()


def _render_weekly_summary_markdown(
    *,
    user_id: str,
    week_start: datetime.date,
    week_end: datetime.date,
    total_entries: int,
    mission_counts: dict[str, int],
    event_counts: dict[str, int],
    files: list[Path],
) -> str:
    lines = [
        f"# Weekly Memory Summary: {week_start.isoformat()} to {week_end.isoformat()}",
        "",
        f"- user_id: {user_id}",
        f"- entries: {total_entries}",
        f"- source_files: {len(files)}",
        "",
        "## Mission Counts",
    ]
    if mission_counts:
        for mission, count in sorted(mission_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {mission}: {count}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Event Counts"])
    if event_counts:
        for event, count in sorted(event_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {event}: {count}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Source Files"])
    for file in files:
        lines.append(f"- {file}")
    if not files:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def _week_start_from_summary_filename(filename: str) -> datetime.date | None:
    text = str(filename or "").strip()
    if not text.startswith("week_"):
        return None
    parts = text.split("_")
    if len(parts) < 2:
        return None
    date_text = parts[1]
    try:
        return datetime.strptime(date_text, _ISO_DATE_FMT).date()
    except ValueError:
        return None


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
