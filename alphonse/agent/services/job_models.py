from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


PayloadType = Literal["job_ability", "tool_call", "prompt_to_brain", "internal_event"]
IdempotencyStrategy = Literal["none", "by_window", "by_input_hash"]


@dataclass
class JobSpec:
    job_id: str
    name: str
    description: str
    enabled: bool
    schedule: dict[str, Any]
    timezone: str
    payload_type: PayloadType
    payload: dict[str, Any]
    domain_tags: list[str] = field(default_factory=list)
    safety_level: str = "low"
    requires_confirmation: bool = False
    retry_policy: dict[str, Any] = field(default_factory=lambda: {"max_retries": 0, "backoff_seconds": 60})
    idempotency: dict[str, Any] = field(default_factory=lambda: {"strategy": "none"})
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())
    last_run_at: str | None = None
    next_run_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobSpec":
        return cls(
            job_id=str(payload.get("job_id") or ""),
            name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
            enabled=bool(payload.get("enabled", True)),
            schedule=dict(payload.get("schedule") or {}),
            timezone=str(payload.get("timezone") or "UTC"),
            payload_type=str(payload.get("payload_type") or "internal_event"),  # type: ignore[arg-type]
            payload=dict(payload.get("payload") or {}),
            domain_tags=[str(item) for item in (payload.get("domain_tags") or []) if str(item).strip()],
            safety_level=str(payload.get("safety_level") or "low"),
            requires_confirmation=bool(payload.get("requires_confirmation", False)),
            retry_policy=dict(payload.get("retry_policy") or {"max_retries": 0, "backoff_seconds": 60}),
            idempotency=dict(payload.get("idempotency") or {"strategy": "none"}),
            created_at=str(payload.get("created_at") or _now_iso()),
            updated_at=str(payload.get("updated_at") or _now_iso()),
            last_run_at=_optional_text(payload.get("last_run_at")),
            next_run_at=_optional_text(payload.get("next_run_at")),
        )


@dataclass
class JobExecution:
    execution_id: str
    job_id: str
    user_id: str
    status: str
    route: str
    started_at: str
    ended_at: str | None
    duration_ms: int | None
    error: dict[str, Any] | None = None
    output_summary: str | None = None
    input_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobExecution":
        return cls(
            execution_id=str(payload.get("execution_id") or ""),
            job_id=str(payload.get("job_id") or ""),
            user_id=str(payload.get("user_id") or ""),
            status=str(payload.get("status") or "error"),
            route=str(payload.get("route") or "brain"),
            started_at=str(payload.get("started_at") or _now_iso()),
            ended_at=_optional_text(payload.get("ended_at")),
            duration_ms=_optional_int(payload.get("duration_ms")),
            error=dict(payload.get("error")) if isinstance(payload.get("error"), dict) else None,
            output_summary=_optional_text(payload.get("output_summary")),
            input_hash=_optional_text(payload.get("input_hash")),
            metadata=dict(payload.get("metadata") or {}),
        )


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
