from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
import logging
import re
from typing import Any
from zoneinfo import ZoneInfo

from alphonse.agent.nervous_system.prompt_artifacts import create_prompt_artifact
from alphonse.agent.services.scheduler_service import SchedulerService
from alphonse.config import settings

logger = logging.getLogger(__name__)


class SchedulerToolError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code or "scheduler_error")
        self.message = str(message or "Scheduler operation failed")
        self.retryable = bool(retryable)
        self.details = dict(details or {})

    def as_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


@dataclass(frozen=True)
class SchedulerTool:
    llm_client: Any | None = None
    scheduler: SchedulerService | None = None

    def execute(self, *, state: dict[str, Any] | None = None, **args: Any) -> dict[str, Any]:
        tool: SchedulerTool = self
        llm_client = (state or {}).get("_llm_client") if isinstance(state, dict) else None
        if tool.llm_client is None and llm_client is not None:
            tool = SchedulerTool(llm_client=llm_client, scheduler=self.scheduler)
        for_whom = str(args.get("ForWhom") or args.get("for_whom") or args.get("To") or "").strip()
        time_value = str(args.get("Time") or args.get("time") or "").strip()
        message_value = str(args.get("Message") or args.get("message") or "").strip()
        timezone_name = args.get("timezone_name") or args.get("TimezoneName")
        correlation_id = args.get("correlation_id") or args.get("CorrelationId")
        if not correlation_id and isinstance(state, dict):
            correlation_id = state.get("correlation_id")
        origin_channel_value = str(
            args.get("origin_channel") or args.get("channel") or args.get("from_") or args.get("from") or ""
        ).strip()
        if not origin_channel_value and isinstance(state, dict):
            origin_channel_value = str(state.get("channel_type") or state.get("channel") or "").strip()
        if not origin_channel_value:
            origin_channel_value = "api"
        channel_target = args.get("channel_target")
        reminder = tool.create_reminder(
            for_whom=for_whom,
            time=time_value,
            message=message_value,
            timezone_name=str(timezone_name or ""),
            correlation_id=str(correlation_id).strip() if correlation_id is not None else None,
            origin_channel=origin_channel_value,
            channel_target=str(channel_target or ""),
        )
        return {
            "status": "ok",
            "result": reminder,
            "error": None,
            "metadata": {"tool": "createReminder"},
        }

    def create_reminder(
        self,
        *,
        for_whom: str,
        time: str,
        message: str,
        timezone_name: str | None,
        correlation_id: str | None = None,
        origin_channel: str | None = None,
        channel_target: str | None = None,
    ) -> dict[str, str]:
        whom_raw = str(for_whom or "").strip()
        trigger_expr = str(time or "").strip()
        reminder_message = str(message or "").strip()
        if not whom_raw:
            raise SchedulerToolError(
                code="missing_for_whom",
                message="for_whom is required",
                retryable=False,
            )
        if not trigger_expr:
            raise SchedulerToolError(
                code="missing_time",
                message="time is required",
                retryable=False,
            )
        if not reminder_message:
            raise SchedulerToolError(
                code="missing_message",
                message="message is required",
                retryable=False,
            )
        resolved_timezone = _resolve_timezone_name(timezone_name)
        fire_at = _normalize_time_expression_to_iso(
            expression=trigger_expr,
            timezone_name=resolved_timezone,
            llm_client=self.llm_client,
        )
        _ = channel_target
        delivery_target = whom_raw
        resolved_origin_channel = str(origin_channel or "").strip().lower() or "api"
        trigger_time = fire_at
        source_instruction = str(reminder_message or "").strip()
        message_mode, message_text = _build_reminder_message_payload(
            llm_client=self.llm_client,
            source_instruction=source_instruction,
        )
        payload = {
            "mind_layer": "conscious",
            "message": message_text,
            "message_text": message_text,
            "message_mode": message_mode,
            "reminder_text_raw": source_instruction,
            "speaker": "alphonse",
            "requested_by": delivery_target,
            "origin_channel": resolved_origin_channel,
            "created_at": datetime.now(dt_timezone.utc).isoformat(),
            "trigger_at": trigger_time,
            "fire_at": trigger_time,
            "delivery_target": delivery_target,
            "event_trigger": {
                "type": "time",
                "time": trigger_time,
                "original_time_expression": trigger_expr,
            },
        }
        internal_prompt = message_text
        artifact_id = create_prompt_artifact(
            user_id=str(delivery_target or "default"),
            source_instruction=source_instruction,
            agent_internal_prompt=internal_prompt,
            language=None,
            artifact_kind="reminder",
        )
        payload["source_instruction"] = source_instruction
        payload["agent_internal_prompt"] = internal_prompt
        payload["prompt"] = internal_prompt
        payload["prompt_artifact_id"] = artifact_id
        schedule_id = self._schedule_timed_signal(
            trigger_time=trigger_time,
            timezone_name=resolved_timezone,
            payload=payload,
            target=delivery_target,
            origin=resolved_origin_channel,
            correlation_id=correlation_id,
        )
        return {
            "reminder_id": schedule_id,
            "fire_at": fire_at,
            "delivery_target": delivery_target,
            "message": reminder_message,
            "original_time_expression": trigger_expr,
        }

    def _schedule_timed_signal(
        self,
        *,
        trigger_time: str,
        timezone_name: str,
        payload: dict[str, Any] | None = None,
        target: str | None = None,
        origin: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        service = self.scheduler or SchedulerService()
        return service.schedule_event(
            trigger_time=trigger_time,
            timezone_name=timezone_name,
            payload=payload,
            target=target,
            origin=origin,
            correlation_id=correlation_id,
        )


def _resolve_timezone_name(timezone_name: str | None) -> str:
    candidate = str(timezone_name or "").strip()
    if candidate:
        try:
            ZoneInfo(candidate)
            return candidate
        except Exception:
            pass
    fallback = str(settings.get_timezone() or "").strip()
    if fallback:
        try:
            ZoneInfo(fallback)
            return fallback
        except Exception:
            pass
    return "UTC"

def _normalize_time_expression_to_iso(*, expression: str, timezone_name: str, llm_client: Any | None = None) -> str:
    raw = str(expression or "").strip()
    if not raw:
        raise SchedulerToolError(
            code="missing_time_expression",
            message="time is required",
            retryable=False,
        )
    iso_direct = _try_parse_iso(raw, timezone_name=timezone_name)
    if iso_direct is not None:
        return iso_direct
    llm_iso = _normalize_with_llm(
        expression=raw,
        timezone_name=timezone_name,
        llm_client=llm_client,
    )
    if llm_iso is not None:
        return llm_iso
    raise SchedulerToolError(
        code="time_expression_unresolvable",
        message="time expression could not be normalized",
        retryable=True,
        details={"expression": raw, "timezone": timezone_name},
    )


def _build_reminder_message_payload(*, llm_client: Any | None, source_instruction: str) -> tuple[str, str]:
    source = str(source_instruction or "").strip()
    logger.info(
        "scheduler_tool paraphrase input source=%r source_instructions=%r",
        source,
        source_instruction,
    )
    if not source:
        logger.info(
            "scheduler_tool paraphrase output mode=paraphrased prompt=%r",
            "You just remembered something important.",
        )
        return "paraphrased", "You just remembered something important."
    quoted = _extract_quoted_text(source)
    if quoted:
        logger.info(
            "scheduler_tool paraphrase output mode=verbatim prompt=%r",
            quoted,
        )
        return "verbatim", quoted
    rewritten = _paraphrase_reminder_message(llm_client=llm_client, source_instruction=source)
    logger.info(
        "scheduler_tool paraphrase output mode=paraphrased prompt=%r",
        rewritten or source,
    )
    return "paraphrased", rewritten or source


def _extract_quoted_text(text: str) -> str:
    for pattern in (r'"([^"]+)"', r"'([^']+)'"):
        matched = re.search(pattern, str(text or ""))
        if not matched:
            continue
        value = str(matched.group(1) or "").strip()
        if value:
            return value
    return ""


def _paraphrase_reminder_message(*, llm_client: Any | None, source_instruction: str) -> str:
    source = str(source_instruction or "").strip()
    if not source:
        logger.info("scheduler_tool _paraphrase_reminder_message source=%r result=%r", source, "")
        return ""
    if llm_client is None:
        logger.info(
            "scheduler_tool _paraphrase_reminder_message source=%r result=%r",
            source,
            source,
        )
        return source
    system_prompt = (
        "You are Alphonse the family's genius butler and virtual assistant.\n"
        "Rewrite reminder content as a clear execution cue for future trigger time.\n"
        "Keep the same language as the source text only if between quotes; otherwise rely the message content.\n"    
    )
    user_prompt = (
        "Source reminder content:\n"
        f"{source}\n\n"
        "Rewrite only the reminder payload text."
    )
    rendered = _llm_complete_text(client=llm_client, system_prompt=system_prompt, user_prompt=user_prompt).strip()
    logger.info(
        "scheduler_tool _paraphrase_reminder_message source=%r result=%r",
        source,
        rendered or source,
    )
    return rendered or source


def _try_parse_iso(raw: str, *, timezone_name: str) -> str | None:
    candidate = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(candidate)
    except Exception:
        return None
    tz = ZoneInfo(timezone_name)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(dt_timezone.utc).isoformat()


def _normalize_with_llm(*, expression: str, timezone_name: str, llm_client: Any | None) -> str | None:
    client = llm_client or _build_scheduler_llm_client()
    if client is None:
        return None
    now_local = datetime.now(ZoneInfo(timezone_name))
    system_prompt = (
        "# Role\n"
        "You convert natural language time expressions into a single ISO timestamp.\n"
        "# Output Contract\n"
        "- Return ONLY one line.\n"
        "- Either a valid ISO-8601 datetime in UTC (offset +00:00) or UNRESOLVABLE.\n"
        "- No markdown, no extra words."
    )
    user_prompt = (
        "## Input\n"
        f"- timezone: {timezone_name}\n"
        f"- now_local: {now_local.isoformat()}\n"
        f"- expression: {expression}\n\n"
        "## Task\n"
        "Convert expression to exact datetime and output UTC ISO only."
    )
    raw = _llm_complete_text(client=client, system_prompt=system_prompt, user_prompt=user_prompt)
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.upper() == "UNRESOLVABLE":
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("text"):
            candidate = candidate[4:].strip()
    parsed = _try_parse_iso(candidate, timezone_name=timezone_name)
    return parsed


def _llm_complete_text(*, client: Any, system_prompt: str, user_prompt: str) -> str:
    complete = getattr(client, "complete", None)
    if not callable(complete):
        return ""
    try:
        return str(complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except TypeError:
        try:
            return str(complete(system_prompt, user_prompt))
        except Exception:
            return ""
    except Exception:
        return ""


def _build_scheduler_llm_client() -> Any | None:
    try:
        from alphonse.agent.cognition.providers.factory import build_llm_client

        return build_llm_client()
    except Exception:
        return None
