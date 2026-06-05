from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from alphonse.agent.observability.log_manager import get_component_logger
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dateutil.rrule import rrulestr

from alphonse.agent.nervous_system.sandbox_dirs import get_sandbox_alias
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.services.job_models import JobExecution, JobSpec

logger = get_component_logger("services.job_store")
VALID_PAYLOAD_TYPES = {"prompt_to_brain"}
_EXECUTION_FILE_LOCK = threading.RLock()


class JobStore:
    def __init__(self, *, root: str | Path | None = None) -> None:
        base = Path(root) if root is not None else _default_jobs_root()
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
        payload_type = _normalize_payload_type(str(payload.get("payload_type") or "prompt_to_brain"))
        payload_data = _normalize_job_payload_for_type(
            payload_type=payload_type,
            payload=dict(payload.get("payload") or {}),
        )
        job = JobSpec(
            job_id=_new_job_id(),
            name=str(payload.get("name") or "").strip(),
            description=str(payload.get("description") or "").strip(),
            enabled=bool(payload.get("enabled", True)),
            schedule=dict(payload.get("schedule") or {}),
            timezone=str(payload.get("timezone") or "UTC"),
            payload_type=payload_type,  # type: ignore[arg-type]
            payload=payload_data,
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
            self._sync_scheduled_job_row(user_id=user_id, job=job)
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
        job.payload_type = _normalize_payload_type(str(job.payload_type or "prompt_to_brain"))  # type: ignore[assignment]
        job.payload = _normalize_job_payload_for_type(
            payload_type=str(job.payload_type or ""),
            payload=dict(job.payload or {}),
        )
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
            self._sync_scheduled_job_row(user_id=user_id, job=job)
        except Exception as exc:
            logger.warning(
                "JobStore save_job sync_failed user_id=%s job_id=%s error=%s",
                user_id,
                job.job_id,
                type(exc).__name__,
            )
        return job

    def backfill_and_sync_jobs(self, *, user_id: str | None = None) -> dict[str, int]:
        owner_migration = self.migrate_legacy_channel_user_job_dirs(user_id=user_id)
        migrated_users = {
            str(item).strip()
            for item in owner_migration.get("canonical_user_ids", [])
            if str(item).strip()
        }
        requested_user = str(user_id or "").strip()
        user_ids = sorted({requested_user, *migrated_users}) if requested_user else self.list_user_ids()
        scanned = 0
        updated = 0
        deleted = 0
        deleted_ids: list[str] = []
        for uid in user_ids:
            data = self._read_jobs(uid)
            jobs_raw = data.get("jobs")
            if not isinstance(jobs_raw, dict):
                continue
            user_changed = False
            for job_id, payload in list(jobs_raw.items()):
                if not isinstance(payload, dict):
                    continue
                scanned += 1
                spec = JobSpec.from_dict(payload)
                changed = False
                original_payload_type = str(spec.payload_type or "").strip()
                normalized_payload_type = _normalize_payload_type(original_payload_type)
                if normalized_payload_type not in VALID_PAYLOAD_TYPES:
                    jobs_raw.pop(str(job_id), None)
                    deleted += 1
                    deleted_ids.append(f"{uid}:{job_id}")
                    user_changed = True
                    try:
                        self._delete_scheduled_job_row(job_id=str(job_id))
                        self._delete_job_timed_signal_row(job_id=str(job_id))
                    except Exception:
                        pass
                    logger.warning(
                        "JobStore deleted_non_conscious_job user_id=%s job_id=%s payload_type=%s",
                        uid,
                        str(job_id),
                        normalized_payload_type,
                    )
                    continue
                spec.payload_type = normalized_payload_type  # type: ignore[assignment]
                normalized_payload = _normalize_job_payload_for_type(
                    payload_type=normalized_payload_type,
                    payload=dict(spec.payload or {}),
                )
                if normalized_payload != spec.payload:
                    spec.payload = normalized_payload
                    changed = True
                changed = _normalize_job_schedule(job=spec, now=_now_utc()) or changed
                if normalized_payload_type != original_payload_type:
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
                    self._sync_scheduled_job_row(user_id=uid, job=spec)
                except Exception as exc:
                    logger.warning(
                        "JobStore backfill sync_failed user_id=%s job_id=%s error=%s",
                        uid,
                        spec.job_id,
                        type(exc).__name__,
                    )
            if user_changed:
                self._write_jobs(uid, data)
        return {
            "scanned": scanned,
            "updated": updated,
            "deleted": deleted,
            "deleted_sample_count": len(deleted_ids[:10]),
            "legacy_owner_dirs_migrated": int(owner_migration.get("dirs_migrated") or 0),
            "legacy_owner_jobs_migrated": int(owner_migration.get("jobs_migrated") or 0),
            "legacy_owner_jobs_merged": int(owner_migration.get("jobs_merged") or 0),
        }

    def migrate_legacy_channel_user_job_dirs(self, *, user_id: str | None = None) -> dict[str, Any]:
        mappings = self._legacy_channel_user_job_dir_mappings(user_id=user_id)
        dirs_migrated = 0
        jobs_migrated = 0
        jobs_merged = 0
        canonical_user_ids: set[str] = set()
        for legacy_user_id, canonical_user_id in mappings.items():
            legacy_data = self._read_jobs(legacy_user_id)
            legacy_jobs = legacy_data.get("jobs")
            if not isinstance(legacy_jobs, dict) or not legacy_jobs:
                continue
            canonical_data = self._read_jobs(canonical_user_id)
            canonical_jobs = canonical_data.get("jobs")
            if not isinstance(canonical_jobs, dict):
                canonical_jobs = {}
                canonical_data["jobs"] = canonical_jobs
            legacy_changed = False
            canonical_changed = False
            for job_id, payload in list(legacy_jobs.items()):
                rendered_job_id = str(job_id or "").strip()
                if not rendered_job_id or not isinstance(payload, dict):
                    continue
                if rendered_job_id not in canonical_jobs:
                    canonical_jobs[rendered_job_id] = _job_payload_with_delivery_target(
                        payload=payload,
                        delivery_target=legacy_user_id,
                    )
                    jobs_migrated += 1
                    canonical_changed = True
                else:
                    jobs_merged += 1
                legacy_jobs.pop(job_id, None)
                legacy_changed = True
            if canonical_changed:
                self._write_jobs(canonical_user_id, canonical_data)
            if legacy_changed:
                self._write_jobs(legacy_user_id, legacy_data)
                dirs_migrated += 1
                canonical_user_ids.add(canonical_user_id)
                logger.info(
                    "JobStore migrated_legacy_owner_jobs legacy_user_id=%s canonical_user_id=%s migrated=%s merged=%s",
                    legacy_user_id,
                    canonical_user_id,
                    jobs_migrated,
                    jobs_merged,
                )
        return {
            "dirs_migrated": dirs_migrated,
            "jobs_migrated": jobs_migrated,
            "jobs_merged": jobs_merged,
            "canonical_user_ids": sorted(canonical_user_ids),
        }

    def migration_report_tool_call_contract(
        self,
        *,
        user_id: str | None = None,
        sample_limit: int = 20,
    ) -> dict[str, Any]:
        user_ids = [str(user_id)] if user_id else self.list_user_ids()
        scanned = 0
        canonical = 0
        non_canonical = 0
        samples: list[str] = []
        for uid in user_ids:
            data = self._read_jobs(uid)
            jobs_raw = data.get("jobs")
            if not isinstance(jobs_raw, dict):
                continue
            for job_id, payload in jobs_raw.items():
                if not isinstance(payload, dict):
                    continue
                spec = JobSpec.from_dict(payload)
                if str(spec.payload_type or "").strip() in VALID_PAYLOAD_TYPES:
                    continue
                scanned += 1
                non_canonical += 1
                if len(samples) < max(1, int(sample_limit)):
                    samples.append(f"{uid}:{job_id}")
        return {
            "scanned": scanned,
            "canonical": canonical,
            "non_canonical": non_canonical,
            "sample_ids": samples,
        }

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
            self._delete_scheduled_job_row(job_id=str(job_id))
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
        with _EXECUTION_FILE_LOCK:
            data = self._read_executions(user_id)
            rows = data.get("executions")
            if not isinstance(rows, list):
                rows = []
                data["executions"] = rows
            rows.append(execution.to_dict())
            data["executions"] = rows[-5000:]
            self._write_executions(user_id, data)
        return execution

    def finalize_execution(
        self,
        *,
        user_id: str,
        execution_id: str,
        status: str,
        output_summary: str,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobExecution:
        normalized_execution_id = str(execution_id or "").strip()
        if not normalized_execution_id:
            raise ValueError("execution_id is required")
        ended_at = _now_utc()
        with _EXECUTION_FILE_LOCK:
            data = self._read_executions(user_id)
            rows = data.get("executions")
            if not isinstance(rows, list):
                raise ValueError("execution_not_found")
            for index in range(len(rows) - 1, -1, -1):
                item = rows[index]
                if not isinstance(item, dict):
                    continue
                if str(item.get("execution_id") or "").strip() != normalized_execution_id:
                    continue
                execution = JobExecution.from_dict(item)
                try:
                    started_at = datetime.fromisoformat(execution.started_at)
                except Exception:
                    started_at = ended_at
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
                execution.status = str(status or "").strip() or "error"
                execution.ended_at = ended_at.isoformat()
                execution.duration_ms = max(
                    int((ended_at - started_at.astimezone(timezone.utc)).total_seconds() * 1000),
                    0,
                )
                execution.error = dict(error) if isinstance(error, dict) else None
                execution.output_summary = str(output_summary or "").strip() or None
                execution.metadata = {
                    **dict(execution.metadata or {}),
                    **dict(metadata or {}),
                }
                rows[index] = execution.to_dict()
                self._write_executions(user_id, data)
                return execution
        raise ValueError("execution_not_found")

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

    def _legacy_channel_user_job_dir_mappings(self, *, user_id: str | None = None) -> dict[str, str]:
        existing_user_dirs = set(self.list_user_ids())
        if not existing_user_dirs:
            return {}
        requested_user = str(user_id or "").strip()
        db_path = resolve_nervous_system_db_path()
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT cu.channel_user_id, cu.user_id
                    FROM channels_users cu
                    JOIN users u ON u.user_id = cu.user_id
                    WHERE cu.is_active = 1
                      AND u.is_active = 1
                    ORDER BY cu.updated_at DESC
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("JobStore legacy_owner_mapping_failed error=%s", type(exc).__name__)
            return {}
        mappings: dict[str, str] = {}
        for row in rows:
            legacy_user_id = str(row[0] or "").strip()
            canonical_user_id = str(row[1] or "").strip()
            if not legacy_user_id or not canonical_user_id:
                continue
            if legacy_user_id == canonical_user_id:
                continue
            if legacy_user_id not in existing_user_dirs:
                continue
            if requested_user and requested_user not in {legacy_user_id, canonical_user_id}:
                continue
            mappings[legacy_user_id] = canonical_user_id
        return mappings

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.parent / f".{path.name}.{secrets.token_hex(4)}.tmp"
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _sync_scheduled_job_row(self, *, user_id: str, job: JobSpec) -> None:
        if not job.enabled:
            self._delete_scheduled_job_row(job_id=job.job_id)
            self._delete_job_timed_signal_row(job_id=job.job_id)
            return
        prompt = _extract_job_prompt(job.payload)
        now_text = _now_utc().isoformat()
        rrule_value = str((job.schedule or {}).get("rrule") or "").strip() or None
        status = "active" if bool(job.enabled) else "paused"
        next_run_at = str(job.next_run_at or "").strip() or None
        retries = int((job.retry_policy or {}).get("max_retries") or 0)
        db_path = resolve_nervous_system_db_path()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO scheduled_jobs (
                  id, name, prompt, owner_id, rrule, retries, status, next_run_at, timezone, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  name = excluded.name,
                  prompt = excluded.prompt,
                  owner_id = excluded.owner_id,
                  rrule = excluded.rrule,
                  retries = excluded.retries,
                  status = excluded.status,
                  next_run_at = excluded.next_run_at,
                  timezone = excluded.timezone,
                  updated_at = excluded.updated_at
                """,
                (
                    str(job.job_id),
                    str(job.name or ""),
                    prompt,
                    str(user_id),
                    rrule_value,
                    retries,
                    status,
                    next_run_at,
                    str(job.timezone or "UTC"),
                    str(job.created_at or now_text),
                    now_text,
                ),
            )
            conn.commit()
        self._sync_job_timed_signal_row(user_id=user_id, job=job)

    def _delete_scheduled_job_row(self, *, job_id: str) -> None:
        db_path = resolve_nervous_system_db_path()
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM scheduled_jobs WHERE id = ?", (str(job_id),))
            conn.commit()

    def _sync_job_timed_signal_row(self, *, user_id: str, job: JobSpec) -> None:
        next_run_at = str(job.next_run_at or "").strip()
        if not next_run_at:
            self._delete_job_timed_signal_row(job_id=job.job_id)
            return
        db_path = resolve_nervous_system_db_path()
        signal_id = _job_timed_signal_id(job.job_id)
        now_text = _now_utc().isoformat()
        target = str(job.payload.get("delivery_target") or user_id or "").strip() or str(user_id)
        origin = str(job.payload.get("origin_channel") or "assistant").strip().lower() or "assistant"
        correlation_id = str(job.payload.get("correlation_id") or job.job_id).strip() or job.job_id
        signal_payload = _job_trigger_payload(job=job, user_id=user_id)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO timed_signals
                  (id, trigger_at, timezone, status, fired_at, signal_type, payload, target, origin, correlation_id, created_at, updated_at)
                VALUES
                  (?, ?, ?, 'pending', NULL, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  trigger_at = excluded.trigger_at,
                  timezone = excluded.timezone,
                  status = 'pending',
                  fired_at = NULL,
                  signal_type = excluded.signal_type,
                  payload = excluded.payload,
                  target = excluded.target,
                  origin = excluded.origin,
                  correlation_id = excluded.correlation_id,
                  updated_at = excluded.updated_at
                """,
                (
                    signal_id,
                    next_run_at,
                    str(job.timezone or "UTC"),
                    "timed_signal",
                    json.dumps(signal_payload, ensure_ascii=False, separators=(",", ":")),
                    target,
                    origin,
                    correlation_id,
                    now_text,
                    now_text,
                ),
            )
            conn.commit()

    def _delete_job_timed_signal_row(self, *, job_id: str) -> None:
        db_path = resolve_nervous_system_db_path()
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM timed_signals WHERE id = ?", (_job_timed_signal_id(job_id),))
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
    return raw or "prompt_to_brain"


def _extract_job_prompt(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("prompt_text", "prompt", "message", "text"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalize_job_payload_for_type(
    *,
    payload_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(payload or {})
    if str(payload_type or "").strip() not in VALID_PAYLOAD_TYPES:
        raise ValueError("jobs_conscious_only_payload_type")
    prompt_text = _extract_job_prompt(normalized)
    if not prompt_text:
        raise ValueError("missing_prompt_text")
    normalized.setdefault("prompt_text", prompt_text)
    return normalized


def _job_timed_signal_id(job_id: str) -> str:
    return f"job_trigger:{str(job_id or '').strip()}"


def _job_trigger_payload(*, job: JobSpec, user_id: str) -> dict[str, Any]:
    payload = dict(job.payload or {})
    return {
        "kind": "job_trigger",
        "job_id": str(job.job_id or "").strip(),
        "user_id": str(user_id or "").strip(),
        "job_name": str(job.name or "").strip(),
        "payload_type": str(job.payload_type or "").strip(),
        "payload": payload,
        "mind_layer": "conscious",
    }


def _job_payload_with_delivery_target(*, payload: dict[str, Any], delivery_target: str) -> dict[str, Any]:
    copied = dict(payload)
    rendered_target = str(delivery_target or "").strip()
    if not rendered_target:
        return copied
    inner = copied.get("payload")
    if not isinstance(inner, dict):
        return copied
    copied_inner = dict(inner)
    copied_inner.setdefault("delivery_target", rendered_target)
    copied["payload"] = copied_inner
    return copied


def compute_retry_time(*, now: datetime, backoff_seconds: int) -> str:
    return (now + timedelta(seconds=max(1, int(backoff_seconds or 1)))).isoformat()


def _default_jobs_root() -> Path:
    configured = str(os.getenv("ALPHONSE_JOBS_ROOT") or "").strip()
    if configured:
        return Path(configured)
    # Keep jobs in the shared workdir sandbox when available.
    for alias in ("main", "dumpster"):
        record = get_sandbox_alias(alias)
        if isinstance(record, dict) and bool(record.get("enabled")):
            base_path = str(record.get("base_path") or "").strip()
            if base_path:
                return Path(base_path) / "jobs"
    return Path("data/jobs")
