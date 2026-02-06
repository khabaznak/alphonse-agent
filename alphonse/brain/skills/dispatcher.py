from __future__ import annotations

import logging
from typing import Any

from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import (
    TelegramAdapter,
)
from alphonse.agent.extremities.telegram_config import build_telegram_adapter_config

logger = logging.getLogger(__name__)


class _SafeFormat(dict):
    def __missing__(self, key: str) -> str:
        return ""


def dispatch(skill: str, args: dict[str, Any], context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    rendered_args = _render_args(args, context)
    if skill == "notify.cli":
        return _send_cli(rendered_args)
    if skill == "notify.telegram":
        return _send_telegram(rendered_args)
    return "failed", {"reason": "unknown_skill", "skill": skill}


def _render_args(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    payload = dict(context.get("event_payload") or {})
    merged = _SafeFormat({**context, **payload})
    rendered: dict[str, Any] = {}
    for key, value in (args or {}).items():
        if isinstance(value, str):
            try:
                rendered[key] = value.format_map(merged)
            except Exception:
                rendered[key] = value
        else:
            rendered[key] = value
    return rendered


def _send_cli(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    message = str(args.get("message_text") or args.get("message") or "")
    if not message:
        return "failed", {"reason": "missing_message"}
    print(message)
    return "sent", {"channel": "cli"}


def _send_telegram(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    message = str(args.get("message_text") or args.get("message") or "")
    if not message:
        return "failed", {"reason": "missing_message"}
    config = build_telegram_adapter_config()
    if not config:
        logger.warning("Telegram notify skipped: missing config")
        return "failed", {"reason": "telegram_not_configured"}
    adapter = TelegramAdapter(config)
    chat_ids = config.get("allowed_chat_ids") or []
    if not chat_ids:
        return "failed", {"reason": "no_allowed_chat_ids"}
    for chat_id in chat_ids:
        adapter.handle_action(
            {
                "type": "send_message",
                "payload": {"chat_id": chat_id, "text": message},
                "target_integration_id": "telegram",
            }
        )
    return "sent", {"channel": "telegram", "chat_ids": chat_ids}
