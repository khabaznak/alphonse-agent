from __future__ import annotations

import os
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "America/Mexico_City"
DEFAULT_LOCALE = "es-MX"
DEFAULT_TONE = "friendly"
DEFAULT_ADDRESS_STYLE = "tu"
DEFAULT_AUTONOMY_LEVEL = 0.35
DEFAULT_EXECUTION_MODE = "readonly"


def get_timezone() -> str:
    configured = os.getenv("ALPHONSE_TIMEZONE")
    tz_name = (
        configured.strip()
        if isinstance(configured, str) and configured.strip()
        else DEFAULT_TIMEZONE
    )
    try:
        ZoneInfo(tz_name)
    except Exception:
        return DEFAULT_TIMEZONE
    return tz_name


def get_default_locale() -> str:
    configured = os.getenv("ALPHONSE_DEFAULT_LOCALE")
    locale = (
        configured.strip()
        if isinstance(configured, str) and configured.strip()
        else DEFAULT_LOCALE
    )
    return locale


def get_tone() -> str:
    configured = os.getenv("ALPHONSE_TONE")
    tone = (
        configured.strip()
        if isinstance(configured, str) and configured.strip()
        else DEFAULT_TONE
    )
    return tone


def get_address_style() -> str:
    configured = os.getenv("ALPHONSE_ADDRESS_STYLE")
    style = (
        configured.strip().lower()
        if isinstance(configured, str) and configured.strip()
        else DEFAULT_ADDRESS_STYLE
    )
    if style not in {"tu", "usted"}:
        return DEFAULT_ADDRESS_STYLE
    return style


def get_autonomy_level() -> float:
    configured = os.getenv("ALPHONSE_AUTONOMY_LEVEL")
    if configured is None:
        return DEFAULT_AUTONOMY_LEVEL
    try:
        value = float(configured)
    except (TypeError, ValueError):
        return DEFAULT_AUTONOMY_LEVEL
    return min(max(value, 0.0), 1.0)


def get_planning_mode() -> str | None:
    configured = os.getenv("ALPHONSE_PLANNING_MODE")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return None


def get_execution_mode() -> str:
    configured = str(os.getenv("ALPHONSE_EXECUTION_MODE") or "").strip().lower()
    if configured in {"readonly", "dev", "ops"}:
        return configured
    return DEFAULT_EXECUTION_MODE
