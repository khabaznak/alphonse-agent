from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

logger = logging.getLogger(__name__)


class CatalogUnavailable(RuntimeError):
    pass


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
    intent_version: str = "1.0.0"
    origin: str = "factory"
    parent_intent: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class IntentCatalogStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(resolve_nervous_system_db_path())
        self.available = True

    def is_available(self) -> bool:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='intent_specs'"
                ).fetchone()
                if not row:
                    return False
                columns = {col[1] for col in conn.execute("PRAGMA table_info(intent_specs)")}
            required = {
                "intent_name",
                "category",
                "description",
                "examples_json",
                "required_slots_json",
                "optional_slots_json",
                "default_mode",
                "risk_level",
                "handler",
                "enabled",
                "intent_version",
                "origin",
                "parent_intent",
                "created_at",
                "updated_at",
            }
            return required.issubset(columns)
        except sqlite3.Error:
            return False

    def list_enabled(self) -> list[IntentSpec]:
        if not self.is_available():
            self.available = False
            logger.error(
                "intent catalog unavailable db_path=%s error=missing intent_specs table or columns",
                self._db_path,
            )
            if _is_dev():
                raise CatalogUnavailable(
                    "intent_specs missing. Run apply_schema() or migrate nerve-db."
                )
            return []
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM intent_specs WHERE enabled = 1",
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if _is_schema_error(exc):
                self.available = False
                logger.error(
                    "intent catalog unavailable db_path=%s error=%s",
                    self._db_path,
                    exc,
                )
                if _is_dev():
                    raise CatalogUnavailable(str(exc)) from exc
                return []
            raise
        return [self._row_to_spec(row) for row in rows]

    def list_all(self) -> list[IntentSpec]:
        if not self.is_available():
            self.available = False
            logger.error(
                "intent catalog unavailable db_path=%s error=missing intent_specs table or columns",
                self._db_path,
            )
            if _is_dev():
                raise CatalogUnavailable(
                    "intent_specs missing. Run apply_schema() or migrate nerve-db."
                )
            return []
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM intent_specs").fetchall()
        except sqlite3.OperationalError as exc:
            if _is_schema_error(exc):
                self.available = False
                logger.error(
                    "intent catalog unavailable db_path=%s error=%s",
                    self._db_path,
                    exc,
                )
                if _is_dev():
                    raise CatalogUnavailable(str(exc)) from exc
                return []
            raise
        return [self._row_to_spec(row) for row in rows]

    def get(self, intent_name: str) -> IntentSpec | None:
        if not self.is_available():
            self.available = False
            logger.error(
                "intent catalog unavailable db_path=%s error=missing intent_specs table",
                self._db_path,
            )
            if _is_dev():
                raise CatalogUnavailable(
                    "intent_specs missing. Run apply_schema() or migrate nerve-db."
                )
            return None
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM intent_specs WHERE intent_name = ?",
                    (intent_name,),
                ).fetchone()
        except sqlite3.OperationalError as exc:
            if _is_schema_error(exc):
                self.available = False
                logger.error(
                    "intent catalog unavailable db_path=%s error=%s",
                    self._db_path,
                    exc,
                )
                if _is_dev():
                    raise CatalogUnavailable(str(exc)) from exc
                return None
            raise
        return self._row_to_spec(row) if row else None

    def upsert(self, spec: IntentSpec) -> None:
        now = _now_iso()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO intent_specs (
                  intent_name, category, description, examples_json,
                  required_slots_json, optional_slots_json, default_mode,
                  risk_level, handler, enabled, intent_version, origin,
                  parent_intent, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(intent_name) DO UPDATE SET
                  category = excluded.category,
                  description = excluded.description,
                  examples_json = excluded.examples_json,
                  required_slots_json = excluded.required_slots_json,
                  optional_slots_json = excluded.optional_slots_json,
                  default_mode = excluded.default_mode,
                  risk_level = excluded.risk_level,
                  handler = excluded.handler,
                  enabled = excluded.enabled,
                  intent_version = excluded.intent_version,
                  origin = excluded.origin,
                  parent_intent = excluded.parent_intent,
                  updated_at = excluded.updated_at
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
                    spec.intent_version,
                    spec.origin,
                    spec.parent_intent,
                    spec.created_at or now,
                    spec.updated_at or now,
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
            intent_version=str(row["intent_version"]) if "intent_version" in row.keys() else "1.0.0",
            origin=str(row["origin"]) if "origin" in row.keys() else "factory",
            parent_intent=str(row["parent_intent"]) if "parent_intent" in row.keys() and row["parent_intent"] is not None else None,
            created_at=str(row["created_at"]) if "created_at" in row.keys() else None,
            updated_at=str(row["updated_at"]) if "updated_at" in row.keys() else None,
        )


@dataclass
class CatalogDiagnostics:
    enabled_count: int
    total_count: int
    categories: dict[str, int]
    last_refresh_at: str | None
    db_path: str
    available: bool


class IntentCatalogService:
    def __init__(
        self,
        *,
        store: IntentCatalogStore | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._store = store or IntentCatalogStore()
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else 60
        self._cache: list[IntentSpec] | None = None
        self._last_refresh_at: datetime | None = None

    @property
    def store(self) -> IntentCatalogStore:
        return self._store

    def refresh(self) -> list[IntentSpec]:
        intents = self._store.list_enabled()
        self._cache = intents
        self._last_refresh_at = datetime.now(timezone.utc)
        return intents

    def load_enabled_intents(self) -> list[IntentSpec]:
        if self._cache is None:
            return self.refresh()
        if self._ttl_seconds <= 0:
            return self._cache
        if not self._last_refresh_at:
            return self.refresh()
        age = (datetime.now(timezone.utc) - self._last_refresh_at).total_seconds()
        if age >= self._ttl_seconds:
            return self.refresh()
        return self._cache

    def get_intent(self, intent_name: str) -> IntentSpec | None:
        for spec in self.load_enabled_intents():
            if spec.intent_name == intent_name:
                return spec
        spec = self._store.get(intent_name)
        if spec and not spec.enabled:
            return None
        return spec

    def diagnostics(self) -> CatalogDiagnostics:
        enabled = self.load_enabled_intents()
        categories: dict[str, int] = {}
        for spec in enabled:
            categories[spec.category] = categories.get(spec.category, 0) + 1
        total_count = len(self._store.list_all()) if self._store.is_available() else 0
        return CatalogDiagnostics(
            enabled_count=len(enabled),
            total_count=total_count,
            categories=categories,
            last_refresh_at=self._last_refresh_at.isoformat() if self._last_refresh_at else None,
            db_path=self._store._db_path,
            available=self._store.is_available(),
        )


_CATALOG_SERVICE: IntentCatalogService | None = None


def get_catalog_service() -> IntentCatalogService:
    global _CATALOG_SERVICE
    expected_db_path = str(resolve_nervous_system_db_path())
    if _CATALOG_SERVICE is not None and _CATALOG_SERVICE.store._db_path != expected_db_path:
        _CATALOG_SERVICE = None
    if _CATALOG_SERVICE is None:
        ttl = int(os.getenv("ALPHONSE_INTENT_CATALOG_TTL", "60"))
        _CATALOG_SERVICE = IntentCatalogService(
            store=IntentCatalogStore(db_path=expected_db_path),
            ttl_seconds=ttl,
        )
    return _CATALOG_SERVICE


def reset_catalog_service() -> None:
    global _CATALOG_SERVICE
    _CATALOG_SERVICE = None


def match_intent_by_examples(text: str, intents: list[IntentSpec]) -> IntentSpec | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    for spec in intents:
        for example in spec.examples:
            if not example:
                continue
            if _example_in_text(normalized, _normalize_text(example)):
                return spec
    return None


def seed_default_intents(db_path: str | None = None) -> None:
    store = IntentCatalogStore(db_path=db_path)
    existing_enabled: dict[str, bool] = {}
    try:
        for spec in store.list_all():
            existing_enabled[spec.intent_name] = spec.enabled
    except Exception:
        existing_enabled = {}
    deprecated = {
        "identity_question",
        "user_identity_question",
        "identity.query_user_name",
        "schedule_reminder",
    }
    for name in deprecated:
        try:
            spec = store.get(name)
        except Exception:
            spec = None
        if spec:
            store.upsert(
                IntentSpec(
                    intent_name=spec.intent_name,
                    category=spec.category,
                    description=spec.description,
                    examples=spec.examples,
                    required_slots=spec.required_slots,
                    optional_slots=spec.optional_slots,
                    default_mode=spec.default_mode,
                    risk_level=spec.risk_level,
                    handler=spec.handler,
                    enabled=False,
                    intent_version=spec.intent_version,
                    origin=spec.origin,
                    parent_intent=spec.parent_intent,
                    created_at=spec.created_at,
                    updated_at=spec.updated_at,
                )
            )
    intents = [
        IntentSpec(
            intent_name="greeting",
            category="core_conversational",
            description="Greet the user.",
            examples=[
                "Hi",
                "Hello",
                "Hey",
                "Good morning",
                "Good afternoon",
                "Good evening",
                "Hola",
                "Buenos días",
                "Buenas tardes",
                "Buenas noches",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="greeting",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="help",
            category="core_conversational",
            description="Explain what Alphonse can do.",
            examples=["help", "ayuda", "what can you do"],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="help",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="get_status",
            category="debug_meta",
            description="Report system status.",
            examples=["status", "estado", "are you online"],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="get_status",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="update_preferences",
            category="control_plane",
            description="Update user preferences like locale, tone, or address style.",
            examples=[
                "Please speak to me in English",
                "Ahora hablemos en español",
                "Be more formal",
                "Trátame de usted",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="update_preferences",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="core.identity.query_agent_name",
            category="core_conversational",
            description="Tell the user who the assistant is.",
            examples=[
                "who are you",
                "what is your name",
                "quién eres",
                "como te llamas",
                "cómo te llamas",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="core.identity.query_agent_name",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="core.identity.query_user_name",
            category="core_conversational",
            description="Tell the user their name if known.",
            examples=[
                "do you know my name",
                "what is my name",
                "do you know who i am",
                "cómo me llamo",
                "como me llamo",
                "quién soy",
                "quien soy",
                "sabes como me llamo",
                "sabes como me llamo yo",
                "sabes mi nombre",
                "ya sabes mi nombre",
                "cuál es mi nombre",
                "cual es mi nombre",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="core.identity.query_user_name",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="core.onboarding.start",
            category="core_conversational",
            description="Begin the primary onboarding flow.",
            examples=[
                "start onboarding",
                "begin onboarding",
                "start the onboarding process",
                "inicia el onboarding",
                "inicia el proceso de onboarding",
                "comienza el onboarding",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="core.onboarding.start",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="core.onboarding.add_user",
            category="core_conversational",
            description="Introduce a new user to Alphonse.",
            examples=[
                "add a new user",
                "add a user",
                "introduce my wife",
                "introduce my son",
                "add my daughter",
                "introduce a family member",
                "agrega un usuario nuevo",
                "agrega a mi esposa",
                "agrega a mi hijo",
                "presenta a mi familia",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="core.onboarding.add_user",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="core.onboarding.authorize_channel",
            category="core_conversational",
            description="Authorize a user to communicate over a channel.",
            examples=[
                "authorize telegram for my wife",
                "authorize gaby on telegram",
                "allow my son on telegram",
                "authorize a user channel",
                "autoriza a mi esposa en telegram",
                "autoriza a gaby en telegram",
                "autoriza un canal para un usuario",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="core.onboarding.authorize_channel",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="core.onboarding.telegram_authorize_invite",
            category="core_conversational",
            description="Authorize a Telegram chat invite for a user.",
            examples=[
                "authorize this telegram chat",
                "allow this telegram chat",
                "approve this telegram chat",
                "autoriza este chat de telegram",
                "aprueba este chat de telegram",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="core.onboarding.telegram_authorize_invite",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="core.onboarding.introduce_authorize",
            category="core_conversational",
            description="Introduce a new user and authorize their channel.",
            examples=[
                "introduce and authorize gaby on telegram",
                "add and authorize my wife on telegram",
                "please meet gaby on telegram",
                "introduce gaby and allow telegram",
                "presenta a mi esposa y autoriza telegram",
                "presenta a gaby en telegram",
            ],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="core.onboarding.introduce_authorize",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
        IntentSpec(
            intent_name="cancel",
            category="core_conversational",
            description="Cancel the current flow or request.",
            examples=["cancel", "stop", "olvida", "olvídalo", "cancelar"],
            required_slots=[],
            optional_slots=[],
            default_mode="aventurizacion",
            risk_level="low",
            handler="cancel",
            enabled=True,
            intent_version="1.0.0",
            origin="factory",
        ),
    ]
    for spec in intents:
        try:
            if spec.intent_name in existing_enabled:
                spec = IntentSpec(
                    intent_name=spec.intent_name,
                    category=spec.category,
                    description=spec.description,
                    examples=spec.examples,
                    required_slots=spec.required_slots,
                    optional_slots=spec.optional_slots,
                    default_mode=spec.default_mode,
                    risk_level=spec.risk_level,
                    handler=spec.handler,
                    enabled=existing_enabled[spec.intent_name],
                    intent_version=spec.intent_version,
                    origin=spec.origin,
                    parent_intent=spec.parent_intent,
                )
            store.upsert(spec)
        except Exception:
            logger.exception("intent catalog seed failed intent=%s", spec.intent_name)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _is_schema_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "no such table" in message or "no such column" in message


def _is_dev() -> bool:
    env = os.getenv("ALPHONSE_ENV", "").lower()
    if env in {"dev", "development"}:
        return True
    return os.getenv("ALPHONSE_DEV") in {"1", "true", "yes"}


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _example_in_text(text: str, example: str) -> bool:
    if not example:
        return False
    example = example.strip()
    if not example:
        return False
    if " " not in example:
        pattern = r"\b" + re.escape(example) + r"\b"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return example in text
