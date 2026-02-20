from __future__ import annotations

import hashlib
import json
import secrets
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dateutil.rrule import rrulestr

from alphonse.agent.nervous_system.timed_store import insert_timed_signal
from alphonse.agent.services.job_models import JobExecution, JobSpec

logger = logging.getLogger(__name__)
VALID_PAYLOAD_TYPES = {"job_ability", "tool_call", "prompt_to_brain", "internal_event"}


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
        payload_type = _normalize_payload_type(str(payload.get("payload_type") or "internal_event"))
        job = JobSpec(
            job_id=_new_job_id(),
            name=str(payload.get("name") or "").strip(),
            description=str(payload.get("description") or "").strip(),
            enabled=bool(payload.get("enabled", True)),
            schedule=dict(payload.get("schedule") or {}),
            timezone=str(payload.get("timezone") or "UTC"),
            payload_type=payload_type,  # type: ignore[arg-type]
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
        if str(job.payload_type or "").strip() not in VALID_PAYLOAD_TYPES:
            raise ValueError("invalid_payload_type")
        _normalize_job_schedule(job=job, now=now)
        if not job.next_run_at:
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
        try:
            self._sync_job_timed_signal(user_id=user_id, job=job)
        except Exception as exc:
            logger.warning(
                "JobStore create_job sync_failed user_id=%s job_id=%s error=%s",
                user_id,
                job.job_id,
                type(exc).__name__,
            )
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
        job.payload_type = _normalize_payload_type(str(job.payload_type or "internal_event"))  # type: ignore[assignment]
        if str(job.payload_type or "").strip() not in VALID_PAYLOAD_TYPES:
            raise ValueError("invalid_payload_type")
        _normalize_job_schedule(job=job, now=_now_utc())
        if not job.next_run_at and bool(job.enabled):
            job.next_run_at = compute_next_run_at(
                schedule=job.schedule,
                timezone_name=job.timezone,
                after=_now_utc(),
            )
        data = self._read_jobs(user_id)
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            jobs = {}
            data["jobs"] = jobs
        job.updated_at = _now_utc().isoformat()
        jobs[job.job_id] = job.to_dict()
        self._write_jobs(user_id, data)
        try:
            self._sync_job_timed_signal(user_id=user_id, job=job)
        except Exception as exc:
            logger.warning(
                "JobStore save_job sync_failed user_id=%s job_id=%s error=%s",
                user_id,
                job.job_id,
                type(exc).__name__,
            )
        return job

    def backfill_and_sync_jobs(self, *, user_id: str | None = None) -> dict[str, int]:
        user_ids = [str(user_id)] if user_id else self.list_user_ids()
        scanned = 0
        updated = 0
        for uid in user_ids:
            data = self._read_jobs(uid)
            jobs_raw = data.get("jobs")
            if not isinstance(jobs_raw, dict):
                continue
            user_changed = False
            for job_id, payload in jobs_raw.items():
                if not isinstance(payload, dict):
                    continue
                scanned += 1
                spec = JobSpec.from_dict(payload)
                original_payload_type = str(spec.payload_type or "").strip()
                normalized_payload_type = _normalize_payload_type(original_payload_type)
                spec.payload_type = normalized_payload_type  # type: ignore[assignment]
                changed = _normalize_job_schedule(job=spec, now=_now_utc())
                if normalized_payload_type != original_payload_type:
                    changed = True
                if str(spec.payload_type or "").strip() not in VALID_PAYLOAD_TYPES:
                    spec.payload_type = "prompt_to_brain"  # type: ignore[assignment]
                    changed = True
                if not spec.next_run_at and bool(spec.enabled):
                    spec.next_run_at = compute_next_run_at(
                        schedule=spec.schedule,
                        timezone_name=spec.timezone,
                        after=_now_utc(),
                    )
                    changed = True
                if changed:
                    spec.updated_at = _now_utc().isoformat()
                    jobs_raw[str(job_id)] = spec.to_dict()
                    updated += 1
                    user_changed = True
                try:
                    self._sync_job_timed_signal(user_id=uid, job=spec)
                except Exception as exc:
                    logger.warning(
                        "JobStore backfill sync_failed user_id=%s job_id=%s error=%s",
                        uid,
                        spec.job_id,
                        type(exc).__name__,
                    )
            if user_changed:
                self._write_jobs(uid, data)
        return {"scanned": scanned, "updated": updated}

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
        try:
            self._delete_job_timed_signal(job_id=str(job_id))
        except Exception:
            pass
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

    def _sync_job_timed_signal(self, *, user_id: str, job: JobSpec) -> None:
        if not job.enabled:
            self._delete_job_timed_signal(job_id=job.job_id)
            return
        dtstart_raw = str((job.schedule or {}).get("dtstart") or "").strip()
        if not dtstart_raw:
            return
        trigger_at = dtstart_raw
        next_trigger = str(job.next_run_at or "").strip() or None
        signal_payload = {
            "job_id": job.job_id,
            "user_id": str(user_id),
            "payload_type": job.payload_type,
            "payload": job.payload,
            "timezone": job.timezone,
            "name": job.name,
            "description": job.description,
        }
        self._delete_job_timed_signal(job_id=job.job_id)
        insert_timed_signal(
            signal_id=f"job_trigger:{job.job_id}",
            trigger_at=trigger_at,
            next_trigger_at=next_trigger,
            rrule=str((job.schedule or {}).get("rrule") or "").strip() or None,
            timezone=job.timezone,
            signal_type="job_trigger",
            mind_layer="conscious",
            dispatch_mode="graph",
            job_id=job.job_id,
            payload=signal_payload,
            target=str(user_id),
            origin="job_store",
            correlation_id=job.job_id,
        )

    def _delete_job_timed_signal(self, *, job_id: str) -> None:
        from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
        import sqlite3

        signal_id = f"job_trigger:{str(job_id)}"
        with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
            try:
                conn.execute("DELETE FROM timed_signals WHERE id = ? OR job_id = ?", (signal_id, str(job_id)))
            except sqlite3.OperationalError:
                conn.execute("DELETE FROM timed_signals WHERE id = ?", (signal_id,))
            conn.commit()


def compute_next_run_at(*, schedule: dict[str, Any], timezone_name: str, after: datetime) -> str | None:
    schedule_type = str(schedule.get("type") or "").strip().lower()
    rrule_value = str(schedule.get("rrule") or "").strip()
    if schedule_type and schedule_type != "rrule":
        return None
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


def _normalize_job_schedule(*, job: JobSpec, now: datetime) -> bool:
    changed = False
    schedule = dict(job.schedule or {})
    rrule_value = str(schedule.get("rrule") or "").strip()
    if not rrule_value:
        job.schedule = schedule
        return changed
    schedule_type = str(schedule.get("type") or "").strip().lower()
    if not schedule_type:
        schedule["type"] = "rrule"
        changed = True
    tz_in_schedule = str(schedule.get("timezone") or "").strip()
    if tz_in_schedule and str(job.timezone or "").strip().upper() == "UTC":
        job.timezone = tz_in_schedule
        changed = True
    dtstart_raw = str(schedule.get("dtstart") or "").strip()
    if not dtstart_raw:
        fallback = str(job.next_run_at or job.last_run_at or job.created_at or "").strip()
        if not fallback:
            fallback = now.isoformat()
        schedule["dtstart"] = fallback
        changed = True
    job.schedule = schedule
    return changed


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


def _normalize_payload_type(value: str) -> str:
    raw = str(value or "").strip()
    if raw == "prompt":
        return "prompt_to_brain"
    return raw or "internal_event"


def compute_retry_time(*, now: datetime, backoff_seconds: int) -> str:
    return (now + timedelta(seconds=max(1, int(backoff_seconds or 1)))).isoformat()
