from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from alphonse.agent.services.job_models import JobExecution, JobSpec
from alphonse.agent.services.job_store import JobStore, compute_next_run_at, compute_retry_time


@dataclass(frozen=True)
class JobRouteDecision:
    route: str
    reason: str


class JobRunner:
    def __init__(
        self,
        *,
        job_store: JobStore,
        tool_registry: Any | None = None,
        brain_event_sink: Callable[[dict[str, Any]], None] | None = None,
        tick_seconds: float = 45.0,
        auto_execute_high_risk: bool = False,
    ) -> None:
        self._job_store = job_store
        self._tool_registry = tool_registry
        self._brain_event_sink = brain_event_sink
        self._tick_seconds = max(float(tick_seconds), 5.0)
        self._auto_execute_high_risk = bool(auto_execute_high_risk)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def set_tool_registry(self, tool_registry: Any) -> None:
        self._tool_registry = tool_registry

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def run_due_once(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        current = now or datetime.now(timezone.utc)
        outcomes: list[dict[str, Any]] = []
        for user_id in self._job_store.list_user_ids():
            due = self._job_store.due_jobs(user_id=user_id, now=current)
            for job in due:
                outcomes.append(self.run_job_now(user_id=user_id, job_id=job.job_id, now=current))
        return outcomes

    def run_job_now(
        self,
        *,
        user_id: str,
        job_id: str,
        now: datetime | None = None,
        brain_event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        current = now or datetime.now(timezone.utc)
        job = self._job_store.get_job(user_id=user_id, job_id=job_id)
        route = route_job(job=job, auto_execute_high_risk=self._auto_execute_high_risk)
        execution_id = f"exec_{int(time.time() * 1000)}_{job.job_id}"
        started_at = datetime.now(timezone.utc)
        input_hash = self._job_store.payload_hash(job.payload)
        if self._is_idempotent_skip(user_id=user_id, job=job, input_hash=input_hash, now=current):
            execution = JobExecution(
                execution_id=execution_id,
                job_id=job.job_id,
                user_id=user_id,
                status="ok",
                route=route.route,
                started_at=started_at.isoformat(),
                ended_at=started_at.isoformat(),
                duration_ms=0,
                output_summary="skipped: idempotency",
                input_hash=input_hash,
                metadata={"idempotency_skip": True, "reason": route.reason},
            )
            self._job_store.append_execution(user_id=user_id, execution=execution)
            self._update_job_after_execution(user_id=user_id, job=job, now=current, success=True, retried=False)
            return {"execution_id": execution.execution_id, "status": execution.status, "route": route.route}

        status = "ok"
        error: dict[str, Any] | None = None
        output_summary = ""
        retried = False
        try:
            if route.route == "needs_confirmation":
                status = "needs_confirmation"
                output_summary = "blocked: confirmation required"
            elif route.route == "brain":
                payload = _brain_payload(job=job, user_id=user_id, current=current)
                sink = brain_event_sink if callable(brain_event_sink) else self._brain_event_sink
                if callable(sink):
                    sink(payload)
                output_summary = "queued_to_brain"
            else:
                output = self._execute_unconscious(job=job, user_id=user_id)
                output_summary = _summarize_output(output)
        except Exception as exc:
            status = "error"
            error = {"type": type(exc).__name__, "message": str(exc)}
            output_summary = str(exc)

        ended_at = datetime.now(timezone.utc)
        execution = JobExecution(
            execution_id=execution_id,
            job_id=job.job_id,
            user_id=user_id,
            status=status,
            route=route.route,
            started_at=started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            duration_ms=max(int((ended_at - started_at).total_seconds() * 1000), 0),
            error=error,
            output_summary=output_summary,
            input_hash=input_hash,
            metadata={"reason": route.reason},
        )
        self._job_store.append_execution(user_id=user_id, execution=execution)
        if status == "error":
            retried = self._apply_retry(user_id=user_id, job=job, now=current)
        self._update_job_after_execution(user_id=user_id, job=job, now=current, success=(status != "error"), retried=retried)
        return {"execution_id": execution.execution_id, "status": execution.status, "route": route.route}

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_due_once()
            except Exception:
                pass
            self._stop.wait(self._tick_seconds)

    def _execute_unconscious(self, *, job: JobSpec, user_id: str) -> Any:
        if job.payload_type == "internal_event":
            payload = dict(job.payload or {})
            return {"event_name": payload.get("event_name"), "data": payload.get("data")}
        if job.payload_type == "tool_call":
            tool_key = str(job.payload.get("tool_key") or "").strip()
            args = dict(job.payload.get("args") or {})
            if not tool_key:
                raise ValueError("missing_tool_key")
            tool = self._tool_registry.get(tool_key) if hasattr(self._tool_registry, "get") else None
            if tool is None or not callable(getattr(tool, "execute", None)):
                raise ValueError("tool_not_found")
            execute = getattr(tool, "execute")
            signature = inspect.signature(execute)
            state = {"actor_person_id": user_id, "incoming_user_id": user_id, "channel_target": user_id}
            if "state" in signature.parameters:
                return execute(**args, state=state)
            return execute(**args)
        raise ValueError("unsupported_unconscious_payload")

    def _apply_retry(self, *, user_id: str, job: JobSpec, now: datetime) -> bool:
        max_retries = int(job.retry_policy.get("max_retries") or 0)
        backoff = int(job.retry_policy.get("backoff_seconds") or 60)
        if max_retries <= 0:
            return False
        recent = self._job_store.list_executions(user_id=user_id, job_id=job.job_id, limit=max_retries + 2)
        error_count = len([item for item in recent if item.status == "error"])
        if error_count > max_retries:
            return False
        job.next_run_at = compute_retry_time(now=now, backoff_seconds=backoff)
        self._job_store.save_job(user_id=user_id, job=job)
        return True

    def _update_job_after_execution(self, *, user_id: str, job: JobSpec, now: datetime, success: bool, retried: bool) -> None:
        job.last_run_at = now.isoformat()
        if not retried:
            job.next_run_at = compute_next_run_at(schedule=job.schedule, timezone_name=job.timezone, after=now)
        self._job_store.save_job(user_id=user_id, job=job)

    def _is_idempotent_skip(self, *, user_id: str, job: JobSpec, input_hash: str, now: datetime) -> bool:
        strategy = str(job.idempotency.get("strategy") or "none").strip().lower()
        if strategy == "none":
            return False
        latest = self._job_store.list_executions(user_id=user_id, job_id=job.job_id, limit=20)
        if not latest:
            return False
        if strategy == "by_input_hash":
            for item in latest:
                if item.status != "ok":
                    continue
                if item.input_hash and item.input_hash == input_hash:
                    return True
            return False
        if strategy == "by_window":
            minutes = int(job.idempotency.get("window_minutes") or 60)
            cutoff = now - timedelta(minutes=max(1, minutes))
            for item in latest:
                if item.status != "ok":
                    continue
                if not item.started_at:
                    continue
                try:
                    started = datetime.fromisoformat(item.started_at)
                except Exception:
                    continue
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                if started >= cutoff:
                    return True
            return False
        return False

def route_job(*, job: JobSpec, auto_execute_high_risk: bool = False) -> JobRouteDecision:
    safety = str(job.safety_level or "low").strip().lower()
    if job.requires_confirmation:
        return JobRouteDecision(route="needs_confirmation", reason="requires_confirmation")
    if safety in {"critical", "high"} and not auto_execute_high_risk:
        return JobRouteDecision(route="brain", reason="high_risk_requires_brain")
    if job.payload_type in {"job_ability", "prompt_to_brain"}:
        return JobRouteDecision(route="brain", reason="conscious_payload")
    if job.payload_type in {"tool_call", "internal_event"}:
        return JobRouteDecision(route="unconscious", reason="deterministic_payload")
    return JobRouteDecision(route="brain", reason="unknown_payload")


def _brain_payload(*, job: JobSpec, user_id: str, current: datetime) -> dict[str, Any]:
    return {
        "type": "SYSTEM_EVENT",
        "message": f"Scheduled job fired (job_id={job.job_id}, name={job.name})",
        "job_id": job.job_id,
        "job_name": job.name,
        "user_id": user_id,
        "payload_type": job.payload_type,
        "payload": job.payload,
        "fired_at": current.isoformat(),
    }


def _summarize_output(output: Any) -> str:
    if isinstance(output, dict):
        if output.get("status") == "failed":
            return str(output.get("error") or "tool_failed")
        text = output.get("result")
        return str(text)[:240]
    return str(output)[:240]
