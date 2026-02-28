from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any

from alphonse.integrations.domotics import (
    ActionRequest,
    QuerySpec,
    SubscribeSpec,
    get_domotics_facade,
)


class DomoticsQueryTool:
    def execute(
        self,
        *,
        kind: str,
        entity_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        facade = _safe_facade()
        if facade is None:
            return _failed("domotics_not_configured", "Domotics integration is not configured")

        spec = QuerySpec(
            kind=str(kind or "").strip().lower(),
            entity_id=str(entity_id).strip() if entity_id else None,
            filters=dict(filters or {}),
        )
        result = facade.query(spec)
        if not result.ok:
            return _failed(
                str(result.error_code or "domotics_query_failed"),
                str(result.error_detail or "domotics query failed"),
            )
        return _ok(
            {
                "kind": spec.kind,
                "entity_id": spec.entity_id,
                "item": result.item,
                "items": result.items,
            },
            tool="domotics.query",
        )


class DomoticsExecuteTool:
    def execute(
        self,
        *,
        domain: str,
        service: str,
        data: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
        readback: bool = True,
        readback_entity_id: str | None = None,
        expected_state: str | None = None,
        expected_attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        facade = _safe_facade()
        if facade is None:
            return _failed("domotics_not_configured", "Domotics integration is not configured")

        request = ActionRequest(
            action_type="call_service",
            domain=str(domain or "").strip(),
            service=str(service or "").strip(),
            data=dict(data or {}),
            target=dict(target or {}) if isinstance(target, dict) else None,
            readback=bool(readback),
            readback_entity_id=str(readback_entity_id).strip() if readback_entity_id else None,
            expected_state=str(expected_state).strip() if expected_state else None,
            expected_attributes=dict(expected_attributes or {}),
        )
        result = facade.execute(request)
        payload = asdict(result)
        if not result.transport_ok:
            error_code = str(result.error_code or "domotics_transport_failed")
            return {
                "status": "failed",
                "result": payload,
                "error": {
                    "code": error_code,
                    "message": str(result.error_detail or "domotics transport failed"),
                    "retryable": _is_retryable_domotics_error(error_code),
                    "details": payload,
                },
                "metadata": {"tool": "domotics.execute"},
            }

        return _ok(payload, tool="domotics.execute")


class DomoticsSubscribeTool:
    def execute(
        self,
        *,
        event_type: str = "state_changed",
        duration_seconds: float = 10.0,
        filters: dict[str, Any] | None = None,
        max_events: int = 200,
    ) -> dict[str, Any]:
        facade = _safe_facade()
        if facade is None:
            return _failed("domotics_not_configured", "Domotics integration is not configured")

        event_name = str(event_type or "state_changed").strip() or "state_changed"
        duration = max(0.5, min(120.0, float(duration_seconds or 10.0)))
        cap = max(1, min(1000, int(max_events or 200)))

        lock = threading.Lock()
        events: list[dict[str, Any]] = []

        def _on_event(event) -> None:
            serialized = asdict(event)
            with lock:
                if len(events) >= cap:
                    return
                events.append(serialized)

        handle = facade.subscribe(
            SubscribeSpec(event_type=event_name, filters=dict(filters or {})),
            _on_event,
        )
        try:
            time.sleep(duration)
        finally:
            handle.unsubscribe()

        with lock:
            snapshot = list(events)

        return _ok(
            {
                "event_type": event_name,
                "duration_seconds": duration,
                "event_count": len(snapshot),
                "events": snapshot,
            },
            tool="domotics.subscribe",
        )


def _safe_facade():
    try:
        return get_domotics_facade()
    except Exception:
        return None


def _ok(result: dict[str, Any], *, tool: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "result": result,
        "error": None,
        "metadata": {"tool": tool},
    }


def _failed(code: str, message: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "result": None,
        "error": {
            "code": str(code),
            "message": str(message),
            "retryable": False,
            "details": {},
        },
        "metadata": {"tool": "domotics"},
    }


def _is_retryable_domotics_error(error_code: str) -> bool:
    normalized = str(error_code or "").strip().lower()
    if normalized in {"entity_unavailable", "unsupported_action_type"}:
        return False
    return True
