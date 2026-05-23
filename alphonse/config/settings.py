from __future__ import annotations

import os
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "America/Mexico_City"
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


def get_execution_mode() -> str:
    configured = str(os.getenv("ALPHONSE_EXECUTION_MODE") or "").strip().lower()
    if configured in {"readonly", "dev", "ops"}:
        return configured
    return DEFAULT_EXECUTION_MODE
