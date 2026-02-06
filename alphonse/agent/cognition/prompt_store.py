from __future__ import annotations

from typing import Protocol


class PromptStore(Protocol):
    def get_template(
        self, key: str, locale: str | None, address_style: str | None, tone: str | None
    ) -> str | None: ...


class NullPromptStore:
    def get_template(
        self, key: str, locale: str | None, address_style: str | None, tone: str | None
    ) -> str | None:
        return None
