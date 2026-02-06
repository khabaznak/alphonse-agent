from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlotSpec:
    name: str
    type: str
    required: bool
    prompt_key: str
    examples: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    critical: bool = True
    semantic_text: bool = False
    min_length: int | None = None
    reject_if_core_conversational: bool = False


@dataclass(frozen=True)
class IntentSpec:
    intent_name: str
    category: str
    description: str
    examples: list[str]
    required_slots: list[SlotSpec]
    optional_slots: list[SlotSpec]
    default_mode: str
    risk_level: str
    handler: str
    enabled: bool = True


class IntentCatalogStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(resolve_nervous_system_db_path())

    def list_enabled(self) -> list[IntentSpec]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM intent_specs WHERE enabled = 1",
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [self._row_to_spec(row) for row in rows]

    def get(self, intent_name: str) -> IntentSpec | None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM intent_specs WHERE intent_name = ?",
                    (intent_name,),
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        return self._row_to_spec(row) if row else None

    def upsert(self, spec: IntentSpec) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO intent_specs (
                  intent_name, category, description, examples_json,
                  required_slots_json, optional_slots_json, default_mode,
                  risk_level, handler, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(intent_name) DO UPDATE SET
                  category = excluded.category,
                  description = excluded.description,
                  examples_json = excluded.examples_json,
                  required_slots_json = excluded.required_slots_json,
                  optional_slots_json = excluded.optional_slots_json,
                  default_mode = excluded.default_mode,
                  risk_level = excluded.risk_level,
                  handler = excluded.handler,
                  enabled = excluded.enabled
                """,
                (
                    spec.intent_name,
                    spec.category,
                    spec.description,
                    json.dumps(spec.examples),
                    json.dumps([_slot_to_dict(s) for s in spec.required_slots]),
                    json.dumps([_slot_to_dict(s) for s in spec.optional_slots]),
                    spec.default_mode,
                    spec.risk_level,
                    spec.handler,
                    1 if spec.enabled else 0,
                ),
            )

    def _row_to_spec(self, row: sqlite3.Row | None) -> IntentSpec:
        if row is None:
            raise ValueError("intent spec row missing")
        return IntentSpec(
            intent_name=str(row["intent_name"]),
            category=str(row["category"]),
            description=str(row["description"]),
            examples=_loads_list(row["examples_json"]),
            required_slots=_loads_slots(row["required_slots_json"]),
            optional_slots=_loads_slots(row["optional_slots_json"]),
            default_mode=str(row["default_mode"]),
            risk_level=str(row["risk_level"]),
            handler=str(row["handler"]),
            enabled=bool(row["enabled"]),
        )


def seed_default_intents(db_path: str | None = None) -> None:
    store = IntentCatalogStore(db_path=db_path)
    intents = [
        IntentSpec(
            intent_name="timed_signals.create",
            category="task_plane",
            description="Schedule a reminder at a specific time or condition.",
            examples=[
                "Remind me to drink water in fifteen minutes",
                "Recuérdame tomar agua en quince minutos",
            ],
            required_slots=[
                SlotSpec(
                    name="reminder_text",
                    type="string",
                    required=True,
                    prompt_key="clarify.reminder_text",
                    examples=["drink water"],
                    critical=True,
                    semantic_text=True,
                    min_length=4,
                    reject_if_core_conversational=True,
                ),
                SlotSpec(
                    name="trigger_time",
                    type="time_expression",
                    required=True,
                    prompt_key="clarify.trigger_time",
                    examples=["in 15 minutes", "a las 7pm"],
                    critical=True,
                ),
            ],
            optional_slots=[
                SlotSpec(
                    name="trigger_geo",
                    type="geo_expression",
                    required=False,
                    prompt_key="clarify.trigger_geo.stub_setup",
                    examples=["al llegar a casa"],
                    critical=True,
                )
            ],
            default_mode="aventurizacion",
            risk_level="low",
            handler="timed_signals.create",
            enabled=True,
        ),
        IntentSpec(
            intent_name="timed_signals.list",
            category="task_plane",
            description="List scheduled reminders.",
            examples=[
                "What reminders do I have?",
                "Qué recordatorios tengo pendientes?",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="timed_signals.list",
            enabled=True,
        ),
        IntentSpec(
            intent_name="core.identity.query_user_name",
            category="core_conversational",
            description="Tell the user their name if known.",
            examples=["Do you know my name?", "¿Cómo me llamo?"],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="core.identity.query_user_name",
            enabled=True,
        ),
    ]
    for spec in intents:
        try:
            store.upsert(spec)
        except Exception:
            logger.exception("intent catalog seed failed intent=%s", spec.intent_name)


def _slot_to_dict(slot: SlotSpec) -> dict[str, Any]:
    return {
        "name": slot.name,
        "type": slot.type,
        "required": slot.required,
        "prompt_key": slot.prompt_key,
        "examples": slot.examples,
        "constraints": slot.constraints,
        "critical": slot.critical,
        "semantic_text": slot.semantic_text,
        "min_length": slot.min_length,
        "reject_if_core_conversational": slot.reject_if_core_conversational,
    }


def _loads_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _loads_slots(raw: str | None) -> list[SlotSpec]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    slots: list[SlotSpec] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        slots.append(
            SlotSpec(
                name=str(item.get("name") or ""),
                type=str(item.get("type") or ""),
                required=bool(item.get("required", True)),
                prompt_key=str(item.get("prompt_key") or ""),
                examples=[str(x) for x in item.get("examples") or []],
                constraints=item.get("constraints") or {},
                critical=bool(item.get("critical", True)),
                semantic_text=bool(item.get("semantic_text", False)),
                min_length=item.get("min_length"),
                reject_if_core_conversational=bool(
                    item.get("reject_if_core_conversational", False)
                ),
            )
        )
    return slots
