"""Sense base types and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from alphonse.senses.bus import Bus


@dataclass(frozen=True)
class SignalSpec:
    key: str
    name: str
    description: str | None = None
    enabled: bool = True
    source: str | None = None


_SENSE_CLASSES: dict[str, type["Sense"]] = {}


def register_sense_class(cls: type["Sense"]) -> None:
    key = getattr(cls, "key", None)
    if not key:
        raise ValueError("Sense classes must define a key")
    existing = _SENSE_CLASSES.get(key)
    if existing and existing is not cls:
        raise ValueError(f"Sense key already registered: {key}")
    _SENSE_CLASSES[key] = cls


def get_registered_sense_classes() -> dict[str, type["Sense"]]:
    return dict(_SENSE_CLASSES)


class Sense(ABC):
    """Background producer for signals."""

    key: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str | None] = None
    source_type: ClassVar[str] = "system"
    owner: ClassVar[str | None] = None
    signals: ClassVar[list[SignalSpec]] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Sense:
            return
        register_sense_class(cls)

    @abstractmethod
    def start(self, bus: Bus) -> None:
        """Start emitting signals via a background producer."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the background producer."""
