from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dateutil.rrule import rrulestr

from alphonse.agent.services.job_models import JobExecution, JobSpec


class JobStore:
    def __init__(self, *, root: str | Path | None = None) -> None:
        base = Path(root) if root is not None else Path("data/jobs")
        self._root = base.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def list_user_ids(self) -> list[str]:
        values: list[str] = []
        for path in self._root.iterdir():
            if path.is_dir():
                values.append(path.name)
        return sorted(values)

    def create_job(self, *, user_id: str, payload: dict[str, Any]) -> JobSpec:
        now = _now_utc()
        job = JobSpec(
            job_id=_new_job_id(),
            name=str(payload.get("name") or "").strip(),
            description=str(payload.get("description") or "").strip(),
            enabled=bool(payload.get("enabled", True)),
            schedule=dict(payload.get("schedule") or {}),
            timezone=str(payload.get("timezone") or "UTC"),
            payload_type=str(payload.get("payload_type") or "internal_event"),  # type: ignore[arg-type]
            payload=dict(payload.get("payload") or {}),
            domain_tags=[str(item).strip() for item in (payload.get("domain_tags") or []) if str(item).strip()],
            safety_level=str(payload.get("safety_level") or "low"),
            requires_confirmation=bool(payload.get("requires_confirmation", False)),
            retry_policy=dict(payload.get("retry_policy") or {"max_retries": 0, "backoff_seconds": 60}),
            idempotency=dict(payload.get("idempotency") or {"strategy": "none"}),
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )
        if not job.name:
            raise ValueError("name is required")
        if not isinstance(job.schedule, dict):
            raise ValueError("schedule is required")
        job.next_run_at = compute_next_run_at(
            schedule=job.schedule,
            timezone_name=job.timezone,
            after=now,
        )
        data = self._read_jobs(user_id)
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            jobs = {}
            data["jobs"] = jobs
        jobs[job.job_id] = job.to_dict()
        self._write_jobs(user_id, data)
        return job

    def list_jobs(
        self,
        *,
        user_id: str,
        enabled: bool | None = None,
        domain_tag: str | None = None,
        limit: int = 50,
    ) -> list[JobSpec]:
        data = self._read_jobs(user_id)
        jobs = data.get("jobs") if isinstance(data.get("jobs"), dict) else {}
        rows: list[JobSpec] = []
        tag_filter = str(domain_tag or "").strip().lower()
        for value in jobs.values():
            if not isinstance(value, dict):
                continue
            spec = JobSpec.from_dict(value)
            if enabled is not None and bool(spec.enabled) != bool(enabled):
                continue
            if tag_filter and tag_filter not in {item.lower() for item in spec.domain_tags}:
                continue
            rows.append(spec)
        rows.sort(key=lambda item: str(item.next_run_at or item.updated_at))
        return rows[: max(1, min(int(limit), 500))]

    def get_job(self, *, user_id: str, job_id: str) -> JobSpec:
        jobs = self._read_jobs(user_id).get("jobs")
        if not isinstance(jobs, dict):
            raise ValueError("job_not_found")
        payload = jobs.get(str(job_id))
        if not isinstance(payload, dict):
            raise ValueError("job_not_found")
        return JobSpec.from_dict(payload)

    def save_job(self, *, user_id: str, job: JobSpec) -> JobSpec:
        data = self._read_jobs(user_id)
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            jobs = {}
            data["jobs"] = jobs
        job.updated_at = _now_utc().isoformat()
        jobs[job.job_id] = job.to_dict()
        self._write_jobs(user_id, data)
        return job

    def pause_job(self, *, user_id: str, job_id: str) -> JobSpec:
        job = self.get_job(user_id=user_id, job_id=job_id)
        job.enabled = False
        return self.save_job(user_id=user_id, job=job)

    def resume_job(self, *, user_id: str, job_id: str) -> JobSpec:
        job = self.get_job(user_id=user_id, job_id=job_id)
        job.enabled = True
        now = _now_utc()
        job.next_run_at = compute_next_run_at(schedule=job.schedule, timezone_name=job.timezone, after=now)
        return self.save_job(user_id=user_id, job=job)

    def delete_job(self, *, user_id: str, job_id: str) -> bool:
        data = self._read_jobs(user_id)
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            return False
        if str(job_id) not in jobs:
            return False
        jobs.pop(str(job_id), None)
        self._write_jobs(user_id, data)
        return True

    def due_jobs(self, *, user_id: str, now: datetime | None = None) -> list[JobSpec]:
        current = now or _now_utc()
        rows: list[JobSpec] = []
        for spec in self.list_jobs(user_id=user_id, enabled=True, limit=1000):
            if not spec.next_run_at:
                continue
            try:
                due_at = datetime.fromisoformat(spec.next_run_at)
            except Exception:
                continue
            if due_at.tzinfo is None:
                due_at = due_at.replace(tzinfo=timezone.utc)
            if due_at <= current:
                rows.append(spec)
        return rows

    def list_executions(self, *, user_id: str, job_id: str | None = None, limit: int = 200) -> list[JobExecution]:
        data = self._read_executions(user_id)
        rows = data.get("executions") if isinstance(data.get("executions"), list) else []
        executions: list[JobExecution] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            execution = JobExecution.from_dict(item)
            if job_id and execution.job_id != str(job_id):
                continue
            executions.append(execution)
        executions.sort(key=lambda item: item.started_at, reverse=True)
        return executions[: max(1, min(int(limit), 2000))]

    def append_execution(self, *, user_id: str, execution: JobExecution) -> JobExecution:
        data = self._read_executions(user_id)
        rows = data.get("executions")
        if not isinstance(rows, list):
            rows = []
            data["executions"] = rows
        rows.append(execution.to_dict())
        data["executions"] = rows[-5000:]
        self._write_executions(user_id, data)
        return execution

    def payload_hash(self, payload: dict[str, Any]) -> str:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _user_dir(self, user_id: str) -> Path:
        safe = _safe_name(user_id)
        target = (self._root / safe).resolve()
        if not _is_subpath(target, self._root):
            raise ValueError("invalid_user_id")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _jobs_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "jobs.json"

    def _executions_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "executions.json"

    def _read_jobs(self, user_id: str) -> dict[str, Any]:
        path = self._jobs_path(user_id)
        if not path.exists():
            payload = {"jobs": {}}
            self._write_json_atomic(path, payload)
            return payload
        return _read_json(path, fallback={"jobs": {}})

    def _write_jobs(self, user_id: str, payload: dict[str, Any]) -> None:
        self._write_json_atomic(self._jobs_path(user_id), payload)

    def _read_executions(self, user_id: str) -> dict[str, Any]:
        path = self._executions_path(user_id)
        if not path.exists():
            payload = {"executions": []}
            self._write_json_atomic(path, payload)
            return payload
        return _read_json(path, fallback={"executions": []})

    def _write_executions(self, user_id: str, payload: dict[str, Any]) -> None:
        self._write_json_atomic(self._executions_path(user_id), payload)

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.parent / f".{path.name}.{secrets.token_hex(4)}.tmp"
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)


def compute_next_run_at(*, schedule: dict[str, Any], timezone_name: str, after: datetime) -> str | None:
    if str(schedule.get("type") or "").strip().lower() != "rrule":
        return None
    rrule_value = str(schedule.get("rrule") or "").strip()
    if not rrule_value:
        return None
    dtstart_raw = str(schedule.get("dtstart") or "").strip()
    if not dtstart_raw:
        return None
    try:
        dtstart = datetime.fromisoformat(dtstart_raw)
    except Exception:
        return None
    tz = _resolve_tz(timezone_name)
    if dtstart.tzinfo is None:
        dtstart = dtstart.replace(tzinfo=tz)
    dtstart_local = dtstart.astimezone(tz)
    after_local = after.astimezone(tz)
    rule = rrulestr(rrule_value, dtstart=dtstart_local)
    candidate = rule.after(after_local, inc=True)
    if candidate is None:
        return None
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=tz)
    return candidate.astimezone(timezone.utc).isoformat()


def _resolve_tz(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(str(name or "UTC"))
    except Exception:
        return ZoneInfo("UTC")


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value or "").strip())
    return text or "default"


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _read_json(path: Path, *, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(fallback)
    return payload if isinstance(payload, dict) else dict(fallback)


def _new_job_id() -> str:
    return f"job_{secrets.token_hex(6)}"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def compute_retry_time(*, now: datetime, backoff_seconds: int) -> str:
    return (now + timedelta(seconds=max(1, int(backoff_seconds or 1)))).isoformat()

