from __future__ import annotations

from datetime import datetime

from alphonse.config import settings


def init_db() -> None:
    return None


def get_timezone() -> str:
    return settings.get_timezone()
