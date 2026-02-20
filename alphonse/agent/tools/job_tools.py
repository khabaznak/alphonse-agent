from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.nervous_system.prompt_artifacts import create_prompt_artifact
from alphonse.agent.services.job_runner import JobRunner
from alphonse.agent.services.job_store import JobStore


@dataclass(frozen=True)
class JobToolServices:
    jobs: JobStore
    runner: JobRunner


@dataclass(frozen=True)
class JobToolContext:
    user_id: str
    services: JobToolServices


class JobCreateTool:
    def __init__(self, store: JobStore) -> None:
        self._store = store

    def execute(
        self,
        *,
        name: str,
        description: str,
        schedule: dict[str, Any],
        payload_type: str,
        payload: dict[str, Any],
        timezone: str = "UTC",
        domain_tags: list[str] | None = None,
        safety_level: str = "low",
        requires_confirmation: bool = False,
        retry_policy: dict[str, Any] | None = None,
        idempotency: dict[str, Any] | None = None,
        enabled: bool = True,
        ctx: JobToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_user = _resolve_user_id(user_id=user_id, state=state, ctx=ctx)
        try:
            source_instruction = _build_job_source_instruction(
                name=name,
                description=description,
                schedule=schedule,
                payload_type=payload_type,
                payload=payload,
            )
            llm_client = state.get("_llm_client") if isinstance(state, dict) else None
            internal_prompt = _you_just_remembered_paraphrase(
                llm_client=llm_client,
                source_instruction=source_instruction,
            )
            artifact_id = create_prompt_artifact(
                user_id=resolved_user,
                source_instruction=source_instruction,
                agent_internal_prompt=internal_prompt,
                language=None,
                artifact_kind="job",
            )
            payload_with_prompt = dict(payload or {})
            payload_with_prompt.setdefault("source_instruction", source_instruction)
            payload_with_prompt.setdefault("agent_internal_prompt", internal_prompt)
            payload_with_prompt.setdefault("prompt_artifact_id", artifact_id)
            job = self._store.create_job(
                user_id=resolved_user,
                payload={
                    "name": name,
                    "description": description,
                    "schedule": schedule,
                    "payload_type": payload_type,
                    "payload": payload_with_prompt,
                    "timezone": timezone,
                    "domain_tags": domain_tags or [],
                    "safety_level": safety_level,
                    "requires_confirmation": requires_confirmation,
                    "retry_policy": retry_policy or {"max_retries": 0, "backoff_seconds": 60},
                    "idempotency": idempotency or {"strategy": "none"},
                    "enabled": enabled,
                },
            )
            return _ok(
                result={"job_id": job.job_id, "next_run_at": job.next_run_at},
                metadata={"tool": "job_create"},
            )
        except Exception as exc:
            return _failed(code=_err_code(exc), message=str(exc), metadata={"tool": "job_create"})


class JobListTool:
    def __init__(self, store: JobStore) -> None:
        self._store = store

    def execute(
        self,
        *,
        enabled: bool | None = None,
        domain_tag: str | None = None,
        limit: int = 25,
        ctx: JobToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_user = _resolve_user_id(user_id=user_id, state=state, ctx=ctx)
        try:
            jobs = self._store.list_jobs(user_id=resolved_user, enabled=enabled, domain_tag=domain_tag, limit=limit)
            rows = [
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "schedule_summary": str(job.schedule.get("rrule") or ""),
                    "next_run_at": job.next_run_at,
                    "enabled": job.enabled,
                }
                for job in jobs
            ]
            return _ok(result={"jobs": rows}, metadata={"tool": "job_list"})
        except Exception as exc:
            return _failed(code=_err_code(exc), message=str(exc), metadata={"tool": "job_list"})


class JobPauseTool:
    def __init__(self, store: JobStore) -> None:
        self._store = store

    def execute(
        self,
        *,
        job_id: str,
        ctx: JobToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_user = _resolve_user_id(user_id=user_id, state=state, ctx=ctx)
        try:
            job = self._store.pause_job(user_id=resolved_user, job_id=job_id)
            return _ok(result={"job_id": job.job_id, "enabled": job.enabled, "next_run_at": job.next_run_at}, metadata={"tool": "job_pause"})
        except Exception as exc:
            return _failed(code=_err_code(exc), message=str(exc), metadata={"tool": "job_pause"})


class JobResumeTool:
    def __init__(self, store: JobStore) -> None:
        self._store = store

    def execute(
        self,
        *,
        job_id: str,
        ctx: JobToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_user = _resolve_user_id(user_id=user_id, state=state, ctx=ctx)
        try:
            job = self._store.resume_job(user_id=resolved_user, job_id=job_id)
            return _ok(result={"job_id": job.job_id, "enabled": job.enabled, "next_run_at": job.next_run_at}, metadata={"tool": "job_resume"})
        except Exception as exc:
            return _failed(code=_err_code(exc), message=str(exc), metadata={"tool": "job_resume"})


class JobDeleteTool:
    def __init__(self, store: JobStore) -> None:
        self._store = store

    def execute(
        self,
        *,
        job_id: str,
        ctx: JobToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_user = _resolve_user_id(user_id=user_id, state=state, ctx=ctx)
        try:
            deleted = self._store.delete_job(user_id=resolved_user, job_id=job_id)
            return _ok(result={"job_id": str(job_id), "deleted": bool(deleted)}, metadata={"tool": "job_delete"})
        except Exception as exc:
            return _failed(code=_err_code(exc), message=str(exc), metadata={"tool": "job_delete"})


class JobRunNowTool:
    def __init__(self, runner: JobRunner) -> None:
        self._runner = runner

    def execute(
        self,
        *,
        job_id: str,
        ctx: JobToolContext | None = None,
        user_id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_user = _resolve_user_id(user_id=user_id, state=state, ctx=ctx)
        try:
            outcome = self._runner.run_job_now(user_id=resolved_user, job_id=job_id)
            return _ok(
                result={
                    "execution_id": outcome.get("execution_id"),
                    "status": outcome.get("status"),
                },
                metadata={"tool": "job_run_now"},
            )
        except Exception as exc:
            return _failed(code=_err_code(exc), message=str(exc), metadata={"tool": "job_run_now"})


def _resolve_user_id(*, user_id: str | None, state: dict[str, Any] | None, ctx: JobToolContext | None) -> str:
    if isinstance(ctx, JobToolContext):
        return str(ctx.user_id)
    value = str(user_id or "").strip()
    if value:
        return value
    if isinstance(state, dict):
        for key in ("actor_person_id", "incoming_user_id", "channel_target", "chat_id"):
            candidate = str(state.get(key) or "").strip()
            if candidate:
                return candidate
    return "default"


def _err_code(exc: Exception) -> str:
    text = str(exc or "").strip()
    if not text:
        return "job_tool_failed"
    return text.split(":", 1)[0]


def _ok(*, result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "result": result,
        "error": None,
        "metadata": metadata,
    }


def _failed(*, code: str, message: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {"code": code, "message": message},
        "metadata": metadata,
    }


def _build_job_source_instruction(
    *,
    name: str,
    description: str,
    schedule: dict[str, Any],
    payload_type: str,
    payload: dict[str, Any],
) -> str:
    rrule = str((schedule or {}).get("rrule") or "").strip()
    dtstart = str((schedule or {}).get("dtstart") or "").strip()
    return (
        f"Create scheduled job '{str(name or '').strip()}'. "
        f"Description: {str(description or '').strip()}. "
        f"Schedule dtstart={dtstart}, rrule={rrule}. "
        f"Payload type={str(payload_type or '').strip()}. "
        f"Payload={payload}."
    ).strip()


def _you_just_remembered_paraphrase(*, llm_client: Any | None, source_instruction: str) -> str:
    source = str(source_instruction or "").strip()
    if not source:
        return "You just remembered a scheduled job."
    complete = getattr(llm_client, "complete", None) if llm_client is not None else None
    if not callable(complete):
        return f"You just remembered: {source}"
    system_prompt = (
        "Rewrite the source instruction as a first-person internal reminder.\n"
        "Technique: 'You Just Remembered'.\n"
        "Keep the same language as source text.\n"
        "One concise sentence. Plain text only."
    )
    user_prompt = f"Source instruction:\n{source}"
    try:
        rendered = str(complete(system_prompt=system_prompt, user_prompt=user_prompt)).strip()
    except TypeError:
        try:
            rendered = str(complete(system_prompt, user_prompt)).strip()
        except Exception:
            rendered = ""
    except Exception:
        rendered = ""
    return rendered or f"You just remembered: {source}"
