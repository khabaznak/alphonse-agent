from __future__ import annotations

from datetime import datetime


def init_db() -> None:
    return None


def get_timezone() -> str:
    tzinfo = datetime.now().astimezone().tzinfo
    if tzinfo is not None and hasattr(tzinfo, "key"):
        return str(tzinfo.key)
    return "UTC"
