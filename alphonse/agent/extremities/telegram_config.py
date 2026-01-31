"""Shared Telegram configuration helpers."""

from __future__ import annotations

import os
from typing import Any


def build_telegram_adapter_config() -> dict[str, Any] | None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return None

    allowed = _parse_allowed_chat_ids(
        os.getenv("TELEGRAM_ALLOWED_CHAT_IDS"),
        os.getenv("TELEGRAM_ALLOWED_CHAT_ID"),
    )
    poll_interval = _parse_float(os.getenv("TELEGRAM_POLL_INTERVAL_SEC"), default=1.0)

    config: dict[str, Any] = {
        "bot_token": bot_token,
        "poll_interval_sec": poll_interval,
    }
    if allowed is not None:
        config["allowed_chat_ids"] = list(allowed)
    return config


def _parse_allowed_chat_ids(primary: str | None, fallback: str | None) -> set[int] | None:
    raw = primary or fallback
    if not raw:
        return None
    ids: set[int] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            ids.add(int(entry))
        except ValueError:
            continue
    return ids or None


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
