from __future__ import annotations

from alphonse.agent.observability.log_manager import get_component_logger
import os
from typing import Any

from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.agent.extremities.telegram_config import build_telegram_adapter_config
from alphonse.agent.lan.pairing_store import create_delivery_receipt, append_audit

logger = get_component_logger("lan.pairing_notify")


def notify_pairing_request(pairing_id: str, device_name: str | None, otp: str, expires_at: str) -> None:
    _notify_cli(pairing_id, device_name, otp, expires_at)
    _notify_telegram(pairing_id, device_name, otp, expires_at)


def _notify_cli(pairing_id: str, device_name: str | None, otp: str, expires_at: str) -> None:
    name = device_name or "Unknown device"
    print("\nPairing request received:")
    print(f"- device: {name}")
    print(f"- pairing_id: {pairing_id}")
    print(f"- otp: {otp}")
    print(f"- expires_at: {expires_at}")
    print("Approve: approve <pairing_id> <otp>")
    print("Deny: deny <pairing_id>\n")
    create_delivery_receipt(
        pairing_id=pairing_id,
        channel="cli",
        status="sent",
        details={"device_name": name},
    )
    append_audit("pairing.notify.cli", pairing_id, {"device_name": name})


def _notify_telegram(pairing_id: str, device_name: str | None, otp: str, expires_at: str) -> None:
    config = build_telegram_adapter_config()
    if not config:
        create_delivery_receipt(
            pairing_id=pairing_id,
            channel="telegram",
            status="failed",
            details={"error": "missing_config"},
        )
        return
    adapter = TelegramAdapter(config)
    chat_ids = config.get("allowed_chat_ids") or []
    if not chat_ids:
        create_delivery_receipt(
            pairing_id=pairing_id,
            channel="telegram",
            status="failed",
            details={"error": "no_allowed_chat_ids"},
        )
        return
    name = device_name or "Unknown device"
    message = (
        "Pairing request:\n"
        f"- device: {name}\n"
        f"- pairing_id: {pairing_id}\n"
        f"- otp: {otp}\n"
        f"- expires_at: {expires_at}\n\n"
        f"Approve: /approve {pairing_id} {otp}\n"
        f"Deny: /deny {pairing_id}"
    )
    try:
        for chat_id in chat_ids:
            adapter.handle_action(
                {
                    "type": "send_message",
                    "payload": {"chat_id": chat_id, "text": message},
                    "target_integration_id": "telegram",
                }
            )
        create_delivery_receipt(
            pairing_id=pairing_id,
            channel="telegram",
            status="sent",
            details={"chat_ids": chat_ids},
        )
        append_audit("pairing.notify.telegram", pairing_id, {"chat_ids": chat_ids})
    except Exception as exc:
        logger.warning("Telegram pairing notify failed: %s", exc)
        create_delivery_receipt(
            pairing_id=pairing_id,
            channel="telegram",
            status="failed",
            details={"error": str(exc)},
        )
