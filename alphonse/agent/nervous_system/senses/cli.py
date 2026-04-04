from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone

from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system.seed import (
    BOOTSTRAP_ADMIN_DISPLAY_NAME,
    BOOTSTRAP_ADMIN_USER_ID,
    BOOTSTRAP_CLI_SERVICE_USER_ID,
)
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.io import CliSenseAdapter

logger = get_component_logger("senses.cli")


def build_cli_user_message_signal(
    *,
    text: str,
    correlation_id: str | None = None,
    user_name: str | None = None,
    channel_target: str = "cli",
    metadata: dict[str, object] | None = None,
    external_user_id: str | None = None,
    person_id: str | None = None,
) -> Signal:
    identity = _resolve_cli_identity(
        user_name=user_name,
        external_user_id=external_user_id,
        person_id=person_id,
    )
    sense_adapter = CliSenseAdapter()
    normalized = sense_adapter.normalize(
        {
            "text": str(text or "").strip(),
            "origin": "cli",
            "user_id": identity["external_user_id"],
            "user_name": identity["display_name"],
            "person_id": identity["person_id"],
            "timestamp": time.time(),
            "correlation_id": correlation_id or str(uuid.uuid4()),
        }
    )
    occurred_at = datetime.fromtimestamp(float(normalized.timestamp), tz=timezone.utc).isoformat()
    payload = build_incoming_message_envelope(
        message_id=str(normalized.correlation_id or normalized.timestamp),
        channel_type=normalized.channel_type,
        channel_target=str(normalized.channel_target or channel_target),
        provider="cli",
        text=normalized.text,
        occurred_at=occurred_at,
        correlation_id=normalized.correlation_id,
        actor_external_user_id=identity["external_user_id"],
        actor_display_name=identity["display_name"],
        actor_person_id=identity["person_id"],
        metadata={
            "normalized_metadata": normalized.metadata,
            "service_key": "cli",
            "service_user_id": identity["external_user_id"],
            "bootstrap_admin_user_id": identity["person_id"],
            **dict(metadata or {}),
        },
    )
    return Signal(
        type="sense.cli.message.user.received",
        payload=payload,
        source="cli",
        correlation_id=normalized.correlation_id,
    )


class CliSense(Sense):
    key = "cli"
    name = "CLI Sense"
    description = "Reads CLI input and emits sense.cli.message.user.received"
    source_type = "system"
    signals = [
        SignalSpec(key="sense.cli.message.user.received", name="CLI User Message Received"),
    ]

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._bus: Bus | None = None
        self._sense_adapter = CliSenseAdapter()

    def start(self, bus: Bus) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._bus = bus
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("CliSense started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("CliSense stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = input("alphonse> ").strip()
            except EOFError:
                self._stop_event.set()
                return
            if not text:
                continue
            if not self._bus:
                continue
            signal = build_cli_user_message_signal(
                text=text,
                metadata={"source": "cli.sense"},
            )
            self._bus.emit(signal)


def _resolve_cli_identity(
    *,
    user_name: str | None,
    external_user_id: str | None,
    person_id: str | None,
) -> dict[str, str | None]:
    try:
        admin = users_store.get_active_admin_user()
    except Exception:
        admin = None
    resolved_person_id = str(person_id or "").strip() or None
    if resolved_person_id is None and isinstance(admin, dict):
        resolved_person_id = str(admin.get("user_id") or "").strip() or None
    display_name = str(user_name or "").strip() or None
    if display_name is None and isinstance(admin, dict):
        display_name = str(admin.get("display_name") or "").strip() or None
    external_id = str(external_user_id or "").strip() or BOOTSTRAP_CLI_SERVICE_USER_ID
    return {
        "external_user_id": external_id,
        "person_id": resolved_person_id or BOOTSTRAP_ADMIN_USER_ID,
        "display_name": display_name or BOOTSTRAP_ADMIN_DISPLAY_NAME,
    }
