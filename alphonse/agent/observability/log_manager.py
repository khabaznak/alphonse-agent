from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any

from alphonse.agent.observability.store import write_task_event

_DEFAULT_LOGGER_NAME = "alphonse.agent.observability"


class LogManager:
    """Centralized structured logging + observability sink."""

    def __init__(self, logger_name: str = _DEFAULT_LOGGER_NAME) -> None:
        self._logger = logging.getLogger(logger_name)

    def emit(
        self,
        *,
        level: str = "info",
        event: str,
        message: str | None = None,
        component: str | None = None,
        correlation_id: str | None = None,
        channel: str | None = None,
        user_id: str | None = None,
        node: str | None = None,
        cycle: int | None = None,
        status: str | None = None,
        tool: str | None = None,
        error_code: str | None = None,
        latency_ms: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        normalized_level = str(level or "info").lower()
        event_payload: dict[str, Any] = {
            "level": normalized_level,
            "event": str(event or "unknown_event"),
            "component": component,
            "correlation_id": correlation_id,
            "channel": channel,
            "user_id": user_id,
            "node": node,
            "cycle": cycle,
            "status": status,
            "tool": tool,
            "error_code": error_code,
            "latency_ms": latency_ms,
            "message": message,
        }
        if isinstance(payload, dict) and payload:
            event_payload.update(payload)

        self._log_text_line(level=normalized_level, payload=event_payload)
        self._write_observability_event(event_payload)

    def emit_exception(
        self,
        *,
        event: str,
        exc: BaseException,
        message: str | None = None,
        component: str | None = None,
        correlation_id: str | None = None,
        channel: str | None = None,
        user_id: str | None = None,
        node: str | None = None,
        cycle: int | None = None,
        status: str | None = None,
        tool: str | None = None,
        error_code: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        stack_excerpt = traceback.format_exc(limit=10)
        merged_payload = dict(payload or {})
        merged_payload.update(
            {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "stack_excerpt": stack_excerpt,
            }
        )
        self.emit(
            level="error",
            event=event,
            message=message or str(exc),
            component=component,
            correlation_id=correlation_id,
            channel=channel,
            user_id=user_id,
            node=node,
            cycle=cycle,
            status=status,
            tool=tool,
            error_code=error_code or type(exc).__name__,
            payload=merged_payload,
        )

    def _log_text_line(self, *, level: str, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        if level == "debug":
            self._logger.debug("event %s", line)
        elif level in {"warning", "warn"}:
            self._logger.warning("event %s", line)
        elif level == "error":
            self._logger.error("event %s", line)
        else:
            self._logger.info("event %s", line)

    def _write_observability_event(self, payload: dict[str, Any]) -> None:
        try:
            write_task_event(payload)
        except Exception:
            return


class StructuredLoggerAdapter:
    """Drop-in logger-style adapter that writes via LogManager."""

    def __init__(self, *, manager: LogManager, component: str) -> None:
        self._manager = manager
        self._component = component

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(level="debug", msg=msg, args=args, kwargs=kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(level="info", msg=msg, args=args, kwargs=kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(level="warning", msg=msg, args=args, kwargs=kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(level="error", msg=msg, args=args, kwargs=kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        exc = kwargs.get("exc_info")
        text = self._format(msg, args)
        context = self._extract_context(text=text, kwargs=kwargs)
        if isinstance(exc, BaseException):
            self._manager.emit_exception(
                event=context["event"],
                exc=exc,
                component=self._component,
                correlation_id=context["correlation_id"],
                channel=context["channel"],
                user_id=context["user_id"],
                node=context["node"],
                cycle=context["cycle"],
                status=context["status"],
                tool=context["tool"],
                error_code=context["error_code"],
                latency_ms=context["latency_ms"],
                message=text,
                payload=context["payload"],
            )
            return
        self._manager.emit(
            level="error",
            event=context["event"],
            component=self._component,
            correlation_id=context["correlation_id"],
            channel=context["channel"],
            user_id=context["user_id"],
            node=context["node"],
            cycle=context["cycle"],
            status=context["status"],
            tool=context["tool"],
            error_code=context["error_code"],
            latency_ms=context["latency_ms"],
            message=text,
            payload={
                **context["payload"],
                "stack_excerpt": traceback.format_exc(limit=10),
            },
        )

    def _emit(self, *, level: str, msg: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        text = self._format(msg, args)
        context = self._extract_context(text=text, kwargs=kwargs)
        self._manager.emit(
            level=level,
            event=context["event"],
            component=self._component,
            correlation_id=context["correlation_id"],
            channel=context["channel"],
            user_id=context["user_id"],
            node=context["node"],
            cycle=context["cycle"],
            status=context["status"],
            tool=context["tool"],
            error_code=context["error_code"],
            latency_ms=context["latency_ms"],
            message=text,
            payload=context["payload"],
        )

    @staticmethod
    def _format(msg: str, args: tuple[Any, ...]) -> str:
        if not args:
            return str(msg)
        try:
            return str(msg) % args
        except Exception:
            arg_text = ", ".join(str(v) for v in args)
            return f"{msg} | args={arg_text}"

    def _extract_context(self, *, text: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        extra = kwargs.get("extra")
        extra_map = extra if isinstance(extra, dict) else {}
        kv_from_text = _extract_kv_pairs(text)
        merged: dict[str, Any] = {**kv_from_text, **extra_map}
        event = str(merged.get("event") or f"{self._component}.log")
        return {
            "event": event,
            "correlation_id": _as_text_or_none(merged.get("correlation_id")),
            "channel": _as_text_or_none(merged.get("channel")),
            "user_id": _as_text_or_none(merged.get("user_id")),
            "node": _as_text_or_none(merged.get("node")),
            "cycle": _as_int_or_none(merged.get("cycle")),
            "status": _as_text_or_none(merged.get("status")),
            "tool": _as_text_or_none(merged.get("tool")),
            "error_code": _as_text_or_none(merged.get("error_code")),
            "latency_ms": _as_int_or_none(merged.get("latency_ms")),
            "payload": {
                "logger_kwargs": kwargs or None,
                "parsed_fields": merged or None,
            },
        }


_DEFAULT_MANAGER: LogManager | None = None


def get_log_manager() -> LogManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = LogManager()
    return _DEFAULT_MANAGER


def get_component_logger(component: str) -> StructuredLoggerAdapter:
    return StructuredLoggerAdapter(manager=get_log_manager(), component=component)


_KEY_VALUE_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_.-]*)=([^\s]+)")


def _extract_kv_pairs(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, raw_value in _KEY_VALUE_PATTERN.findall(str(text or "")):
        value = raw_value.strip().strip(",")
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        result[key] = value
    return result


def _as_text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
