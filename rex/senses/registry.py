"""Sense registry and database registration helpers."""

from __future__ import annotations

import importlib
import pkgutil
import sqlite3
from pathlib import Path

from rex.senses.base import SignalSpec, Sense, get_registered_sense_classes


_DISCOVERED = False
_EXCLUDED_MODULES = {"__init__", "base", "bus", "manager", "registry"}


def _discover_sense_modules() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    import rex.senses as senses_pkg

    for module in pkgutil.iter_modules(senses_pkg.__path__):
        if module.name in _EXCLUDED_MODULES:
            continue
        importlib.import_module(f"{senses_pkg.__name__}.{module.name}")
    _DISCOVERED = True


def _resolved_signal_spec(sense_cls: type[Sense], spec: SignalSpec) -> SignalSpec:
    source = spec.source or sense_cls.key
    if source == spec.source:
        return spec
    return SignalSpec(
        key=spec.key,
        name=spec.name,
        description=spec.description,
        enabled=spec.enabled,
        source=source,
    )


def all_senses() -> list[type[Sense]]:
    _discover_sense_modules()
    return list(get_registered_sense_classes().values())


def all_signal_specs() -> list[SignalSpec]:
    specs: list[SignalSpec] = []
    for sense_cls in all_senses():
        for spec in sense_cls.signals:
            specs.append(_resolved_signal_spec(sense_cls, spec))
    return specs


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def register_senses(db_path: str) -> None:
    _discover_sense_modules()
    rows = []
    for sense_cls in all_senses():
        name = getattr(sense_cls, "name", sense_cls.key)
        rows.append(
            {
                "key": sense_cls.key,
                "name": name,
                "description": getattr(sense_cls, "description", None),
                "source_type": getattr(sense_cls, "source_type", "system"),
                "enabled": 1,
                "owner": getattr(sense_cls, "owner", None),
            }
        )
    if not rows:
        return
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO senses
              (key, name, description, source_type, enabled, owner)
            VALUES
              (:key, :name, :description, :source_type, :enabled, :owner)
            """,
            rows,
        )


def register_signals(db_path: str) -> None:
    _discover_sense_modules()
    rows = []
    for sense_cls in all_senses():
        for spec in sense_cls.signals:
            resolved = _resolved_signal_spec(sense_cls, spec)
            rows.append(
                {
                    "key": resolved.key,
                    "name": resolved.name,
                    "source": resolved.source or sense_cls.key,
                    "description": resolved.description,
                    "is_enabled": 1 if resolved.enabled else 0,
                }
            )
    if not rows:
        return
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO signals
              (key, name, source, description, is_enabled)
            VALUES
              (:key, :name, :source, :description, :is_enabled)
            """,
            rows,
        )
