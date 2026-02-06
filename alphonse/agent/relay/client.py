from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from alphonse.agent.lan.store import get_latest_paired_device, get_paired_device

logger = logging.getLogger(__name__)


@dataclass
class RelayConfig:
    supabase_url: str
    service_key: str
    alphonse_id: str
    heartbeat_secs: int = 15
    coalesce_ms: int = 500
    poll_interval_secs: float = 1.0


class RelayClient:
    def __init__(self, config: RelayConfig) -> None:
        self._config = config
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_heartbeat = 0.0
        self._last_status_sent: dict[str, str] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        logger.info("RelayClient started alphonse_id=%s", self._config.alphonse_id)
        while not self._stop.is_set():
            try:
                self._poll_messages()
                self._emit_heartbeat_if_needed()
                self._emit_status_if_needed()
            except Exception as exc:
                logger.warning("RelayClient loop error: %s", exc)
            self._stop.wait(self._config.poll_interval_secs)

    def _poll_messages(self) -> None:
        url = f"{self._rest_base()}/relay_messages"
        params = {
            "alphonse_id": f"eq.{self._config.alphonse_id}",
            "sender": "eq.mobile",
            "type": "eq.command",
            "delivered_to_alphonse": "is.false",
            "order": "ts.asc",
            "limit": "50",
        }
        resp = requests.get(url, headers=self._headers(), params=params, timeout=5)
        if resp.status_code >= 400:
            logger.warning("RelayClient poll failed status=%s body=%s", resp.status_code, resp.text)
            return
        messages = resp.json() if isinstance(resp.json(), list) else []
        for msg in messages:
            self._handle_message(msg)

    def _handle_message(self, msg: dict[str, Any]) -> None:
        message_id = msg.get("id")
        if not message_id:
            return
        payload = msg.get("payload") or {}
        device_id = msg.get("device_id")
        command_type = payload.get("type")
        correlation_id = msg.get("correlation_id") or str(uuid.uuid4())
        response_payload: dict[str, Any] = {}

        if command_type == "request_status":
            response_payload = self._build_status_payload(device_id)
        else:
            response_payload = {"error": "unsupported_command", "type": command_type}

        self._mark_delivered(message_id)
        self._insert_response(
            channel_id=msg.get("channel_id"),
            device_id=device_id,
            correlation_id=str(correlation_id),
            payload=response_payload,
        )

    def _build_status_payload(self, device_id: str | None) -> dict[str, Any]:
        device = get_paired_device(device_id) if device_id else get_latest_paired_device()
        if not device:
            return {"status": "not_paired"}
        return {
            "status": "ok",
            "device_id": device.device_id,
            "device_name": device.device_name,
            "armed": device.armed,
            "armed_at": device.armed_at.isoformat() if device.armed_at else None,
            "armed_by": device.armed_by,
            "armed_until": device.armed_until.isoformat() if device.armed_until else None,
            "last_status": device.last_status,
            "last_status_at": device.last_status_at.isoformat() if device.last_status_at else None,
        }

    def _mark_delivered(self, message_id: str) -> None:
        url = f"{self._rest_base()}/relay_messages"
        params = {"id": f"eq.{message_id}"}
        data = {"delivered_to_alphonse": True}
        resp = requests.patch(url, headers=self._headers(), params=params, json=data, timeout=5)
        if resp.status_code >= 400:
            logger.warning("RelayClient mark delivered failed status=%s body=%s", resp.status_code, resp.text)

    def _insert_response(
        self,
        *,
        channel_id: str | None,
        device_id: str | None,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> None:
        url = f"{self._rest_base()}/relay_messages"
        message = {
            "id": str(uuid.uuid4()),
            "channel_id": channel_id,
            "sender": "alphonse",
            "type": "response",
            "ts": _now_iso(),
            "correlation_id": correlation_id,
            "device_id": device_id or "",
            "alphonse_id": self._config.alphonse_id,
            "payload": payload,
            "schema_version": 1,
            "delivered_to_alphonse": True,
            "delivered_to_device": False,
        }
        resp = requests.post(url, headers=self._headers(), json=message, timeout=5)
        if resp.status_code >= 400:
            logger.warning("RelayClient insert response failed status=%s body=%s", resp.status_code, resp.text)

    def _emit_heartbeat_if_needed(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < self._config.heartbeat_secs:
            return
        self._last_heartbeat = now
        self._insert_event({"kind": "heartbeat"})

    def _emit_status_if_needed(self) -> None:
        device = get_latest_paired_device()
        if not device:
            return
        snapshot = self._build_status_payload(device.device_id)
        key = device.device_id
        serialized = str(snapshot)
        last_sent = self._last_status_sent.get(key)
        if last_sent == serialized:
            return
        # coalesce by time
        now = time.time()
        last_ts = float(self._last_status_sent.get(f"{key}__ts", "0"))
        if now - last_ts < (self._config.coalesce_ms / 1000.0):
            return
        self._last_status_sent[key] = serialized
        self._last_status_sent[f"{key}__ts"] = str(now)
        self._insert_event({"kind": "status", "payload": snapshot})

    def _insert_event(self, payload: dict[str, Any]) -> None:
        url = f"{self._rest_base()}/relay_messages"
        message = {
            "id": str(uuid.uuid4()),
            "channel_id": None,
            "sender": "alphonse",
            "type": "event",
            "ts": _now_iso(),
            "correlation_id": None,
            "device_id": "",
            "alphonse_id": self._config.alphonse_id,
            "payload": payload,
            "schema_version": 1,
            "delivered_to_alphonse": True,
            "delivered_to_device": False,
        }
        resp = requests.post(url, headers=self._headers(), json=message, timeout=5)
        if resp.status_code >= 400:
            logger.warning("RelayClient insert event failed status=%s body=%s", resp.status_code, resp.text)

    def _rest_base(self) -> str:
        return f"{self._config.supabase_url.rstrip('/')}/rest/v1"

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self._config.service_key,
            "Authorization": f"Bearer {self._config.service_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }


def build_relay_client_from_env() -> RelayClient | None:
    if os.getenv("RELAY_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    alphonse_id = os.getenv("ALPHONSE_ID")
    if not supabase_url or not service_key or not alphonse_id:
        logger.warning("RelayClient disabled: missing SUPABASE_URL/SUPABASE_SERVICE_KEY/ALPHONSE_ID")
        return None
    heartbeat = _as_int(os.getenv("RELAY_HEARTBEAT_SECS"), default=15)
    coalesce_ms = _as_int(os.getenv("RELAY_COALESCE_MS"), default=500)
    return RelayClient(
        RelayConfig(
            supabase_url=supabase_url,
            service_key=service_key,
            alphonse_id=alphonse_id,
            heartbeat_secs=heartbeat,
            coalesce_ms=coalesce_ms,
        )
    )


def _as_int(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
