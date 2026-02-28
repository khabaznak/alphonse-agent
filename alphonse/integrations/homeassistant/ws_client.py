from __future__ import annotations

import json
import queue
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.integrations.homeassistant.config import HomeAssistantConfig

try:  # pragma: no cover - import guard
    import websocket
except Exception:  # pragma: no cover - import guard
    websocket = None


logger = get_component_logger("integrations.homeassistant.ws_client")


class HomeAssistantWsError(RuntimeError):
    pass


@dataclass
class _PendingRequest:
    response_queue: queue.Queue[dict[str, Any] | None]


@dataclass
class _Subscription:
    local_id: str
    event_type: str
    callback: Callable[[dict[str, Any]], None]
    ha_subscription_id: int | None = None


class HomeAssistantWsClient:
    def __init__(self, config: HomeAssistantConfig) -> None:
        if websocket is None:
            raise HomeAssistantWsError("websocket-client package is not installed")
        self._config = config
        self._ws_url = _build_ws_url(config.base_url)

        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self._state_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._ws: Any = None
        self._request_id = 0

        self._pending_requests: dict[int, _PendingRequest] = {}
        self._subscriptions: dict[str, _Subscription] = {}
        self._subscriptions_by_ha_id: dict[int, str] = {}

    def connect(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=self._config.ws.open_timeout_sec):
            raise HomeAssistantWsError("WS connect/auth timeout")

    def stop(self) -> None:
        self._shutdown.set()
        self._ready.clear()
        self._close_ws()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._state_lock:
            for pending in self._pending_requests.values():
                pending.response_queue.put(None)
            self._pending_requests.clear()
            self._subscriptions_by_ha_id.clear()
            for subscription in self._subscriptions.values():
                subscription.ha_subscription_id = None

    def subscribe_events(
        self,
        event_type: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> str:
        local_id = f"sub-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
        sub = _Subscription(local_id=local_id, event_type=str(event_type), callback=callback)
        with self._state_lock:
            self._subscriptions[local_id] = sub
        self.connect()
        self._register_subscription(local_id)
        return local_id

    def unsubscribe(self, local_subscription_id: str) -> None:
        with self._state_lock:
            sub = self._subscriptions.pop(local_subscription_id, None)
        if not sub:
            return
        ha_id = sub.ha_subscription_id
        if ha_id is not None:
            with self._state_lock:
                self._subscriptions_by_ha_id.pop(ha_id, None)
            try:
                self._send_request({"type": "unsubscribe_events", "subscription": ha_id})
            except Exception:
                return

    def _run(self) -> None:
        attempt = 0
        while not self._shutdown.is_set():
            if not self._connect_and_auth():
                delay = _backoff_delay(
                    attempt=attempt,
                    minimum=self._config.ws.min_backoff_sec,
                    maximum=self._config.ws.max_backoff_sec,
                    jitter_ratio=self._config.ws.jitter_ratio,
                )
                attempt += 1
                self._shutdown.wait(timeout=delay)
                continue

            attempt = 0
            self._ready.set()
            self._resubscribe_all()
            while not self._shutdown.is_set():
                try:
                    message = self._ws.recv()
                except Exception:
                    self._ready.clear()
                    self._mark_disconnected()
                    break
                payload = _parse_message(message)
                if not payload:
                    continue
                self._handle_message(payload)

    def _connect_and_auth(self) -> bool:
        self._ready.clear()
        try:
            ws = websocket.create_connection(
                self._ws_url,
                timeout=self._config.ws.open_timeout_sec,
            )
            ws.settimeout(self._config.ws.recv_timeout_sec)
        except Exception as exc:
            logger.warning("HomeAssistant WS connection failed: %s", exc)
            return False

        with self._state_lock:
            self._ws = ws

        first = _parse_message(_safe_recv(ws))
        if not first or first.get("type") != "auth_required":
            logger.warning("HomeAssistant WS expected auth_required, got=%s", first)
            self._mark_disconnected()
            return False

        try:
            ws.send(json.dumps({"type": "auth", "access_token": self._config.token}))
        except Exception:
            self._mark_disconnected()
            return False

        second = _parse_message(_safe_recv(ws))
        if not second or second.get("type") != "auth_ok":
            logger.warning("HomeAssistant WS auth failed payload=%s", second)
            self._mark_disconnected()
            return False
        return True

    def _resubscribe_all(self) -> None:
        with self._state_lock:
            local_ids = list(self._subscriptions.keys())
            self._subscriptions_by_ha_id.clear()
            for sub in self._subscriptions.values():
                sub.ha_subscription_id = None
        for local_id in local_ids:
            try:
                self._register_subscription(local_id)
            except Exception as exc:
                logger.warning("HomeAssistant WS resubscribe failed local_id=%s error=%s", local_id, exc)

    def _register_subscription(self, local_id: str) -> None:
        with self._state_lock:
            sub = self._subscriptions.get(local_id)
        if not sub:
            return
        response = self._send_request({"type": "subscribe_events", "event_type": sub.event_type})
        if not response or not response.get("success"):
            raise HomeAssistantWsError(f"subscribe_events failed local_id={local_id} response={response}")
        ha_id = int(response.get("id"))
        with self._state_lock:
            current = self._subscriptions.get(local_id)
            if not current:
                return
            current.ha_subscription_id = ha_id
            self._subscriptions_by_ha_id[ha_id] = local_id

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._ready.wait(timeout=self._config.ws.open_timeout_sec):
            raise HomeAssistantWsError("WS is not ready")

        req_id = self._next_request_id()
        packet = dict(payload)
        packet["id"] = req_id
        response_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1)
        with self._state_lock:
            self._pending_requests[req_id] = _PendingRequest(response_queue=response_queue)
            ws = self._ws
        if ws is None:
            with self._state_lock:
                self._pending_requests.pop(req_id, None)
            raise HomeAssistantWsError("WS is disconnected")

        try:
            ws.send(json.dumps(packet))
        except Exception as exc:
            with self._state_lock:
                pending = self._pending_requests.pop(req_id, None)
            if pending:
                pending.response_queue.put(None)
            self._mark_disconnected()
            raise HomeAssistantWsError(f"WS send failed: {exc}") from exc

        try:
            response = response_queue.get(timeout=self._config.ws.open_timeout_sec)
        except queue.Empty as exc:
            with self._state_lock:
                self._pending_requests.pop(req_id, None)
            raise HomeAssistantWsError(f"WS request timeout id={req_id}") from exc

        if response is None:
            raise HomeAssistantWsError("WS disconnected while waiting for response")
        return response

    def _handle_message(self, payload: dict[str, Any]) -> None:
        response_id = payload.get("id")
        if isinstance(response_id, int):
            with self._state_lock:
                pending = self._pending_requests.pop(response_id, None)
            if pending:
                pending.response_queue.put(payload)
                return

        if payload.get("type") != "event":
            return
        event = payload.get("event") if isinstance(payload.get("event"), dict) else None
        if not event:
            return

        ha_subscription_id = payload.get("id")
        if not isinstance(ha_subscription_id, int):
            return
        with self._state_lock:
            local_id = self._subscriptions_by_ha_id.get(ha_subscription_id)
            sub = self._subscriptions.get(local_id) if local_id else None
        if not sub:
            return
        try:
            sub.callback(event)
        except Exception as exc:
            logger.warning("HomeAssistant WS callback failed local_id=%s error=%s", local_id, exc)

    def _next_request_id(self) -> int:
        with self._state_lock:
            self._request_id += 1
            return self._request_id

    def _mark_disconnected(self) -> None:
        self._close_ws()
        with self._state_lock:
            for pending in self._pending_requests.values():
                pending.response_queue.put(None)
            self._pending_requests.clear()
            self._subscriptions_by_ha_id.clear()
            for sub in self._subscriptions.values():
                sub.ha_subscription_id = None

    def _close_ws(self) -> None:
        with self._state_lock:
            ws = self._ws
            self._ws = None
        if ws is None:
            return
        try:
            ws.close()
        except Exception:
            return


def _build_ws_url(base_url: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :].rstrip("/") + "/api/websocket"
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :].rstrip("/") + "/api/websocket"
    return base_url.rstrip("/") + "/api/websocket"


def _safe_recv(ws: Any) -> str:
    try:
        message = ws.recv()
    except Exception as exc:
        raise HomeAssistantWsError(f"WS receive failed: {exc}") from exc
    if not isinstance(message, str):
        return ""
    return message


def _parse_message(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _backoff_delay(*, attempt: int, minimum: float, maximum: float, jitter_ratio: float) -> float:
    base = min(maximum, minimum * (2 ** max(0, attempt)))
    jitter = base * max(0.0, jitter_ratio)
    return max(0.05, base + random.uniform(-jitter, jitter))
