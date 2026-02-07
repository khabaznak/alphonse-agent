from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptContext:
    locale: str | None = None
    address_style: str | None = None
    tone: str | None = None
    channel: str | None = None
    variant: str | None = None
    policy_tier: str | None = None


@dataclass(frozen=True)
class PromptMatch:
    template: str
    metadata: dict[str, Any]


class PromptStore(Protocol):
    def get_template(self, key: str, context: PromptContext) -> PromptMatch | None: ...

    def upsert_template(
        self,
        *,
        key: str,
        locale: str,
        address_style: str,
        tone: str,
        channel: str,
        variant: str,
        policy_tier: str,
        template: str,
        enabled: bool,
        priority: int,
        changed_by: str,
        reason: str | None = None,
    ) -> str: ...

    def list_templates(
        self, key: str | None = None, enabled_only: bool = False
    ) -> list[dict[str, Any]]: ...

    def rollback_template(
        self,
        template_id: str,
        version: int,
        changed_by: str,
        reason: str | None = None,
    ) -> None: ...


class SqlitePromptStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(resolve_nervous_system_db_path())

    def get_template(self, key: str, context: PromptContext) -> PromptMatch | None:
        rows = self._list_enabled_by_key(key)
        if not rows:
            return None
        best: tuple[float, dict[str, Any]] | None = None
        for row in rows:
            score = _score_template(row, context)
            if score <= 0:
                continue
            if best is None or score > best[0] or (
                score == best[0]
                and _tie_breaker(row, best[1])
            ):
                best = (score, row)
        if not best:
            return None
        match = best[1]
        return PromptMatch(
            template=str(match["template"]),
            metadata={
                "template_id": match["id"],
                "key": match["key"],
                "locale": match["locale"],
                "address_style": match["address_style"],
                "tone": match["tone"],
                "channel": match["channel"],
                "variant": match["variant"],
                "policy_tier": match["policy_tier"],
                "priority": match["priority"],
                "updated_at": match["updated_at"],
            },
        )

    def upsert_template(
        self,
        *,
        key: str,
        locale: str,
        address_style: str,
        tone: str,
        channel: str,
        variant: str,
        policy_tier: str,
        template: str,
        enabled: bool,
        priority: int,
        changed_by: str,
        reason: str | None = None,
    ) -> str:
        now = _timestamp()
        existing = self._find_template(
            key, locale, address_style, tone, channel, variant, policy_tier
        )
        with sqlite3.connect(self._db_path) as conn:
            if existing:
                template_id = existing["id"]
                conn.execute(
                    """
                    UPDATE prompt_templates
                    SET template = ?, enabled = ?, priority = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (template, 1 if enabled else 0, priority, now, template_id),
                )
            else:
                template_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO prompt_templates (
                      id, key, locale, address_style, tone, channel, variant,
                      policy_tier, template, enabled, priority, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        template_id,
                        key,
                        locale,
                        address_style,
                        tone,
                        channel,
                        variant,
                        policy_tier,
                        template,
                        1 if enabled else 0,
                        priority,
                        now,
                        now,
                    ),
                )
            version = _next_version(conn, template_id)
            conn.execute(
                """
                INSERT INTO prompt_versions (
                  id, template_id, version, template, changed_by, change_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    template_id,
                    version,
                    template,
                    changed_by,
                    reason,
                    now,
                ),
            )
        return template_id

    def list_templates(
        self, key: str | None = None, enabled_only: bool = False
    ) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
        if key:
            where.append("key = ?")
            params.append(key)
        if enabled_only:
            where.append("enabled = 1")
        clause = " AND ".join(where)
        sql = "SELECT * FROM prompt_templates"
        if clause:
            sql = f"{sql} WHERE {clause}"
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def rollback_template(
        self,
        template_id: str,
        version: int,
        changed_by: str,
        reason: str | None = None,
    ) -> None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT template
                FROM prompt_versions
                WHERE template_id = ? AND version = ?
                """,
                (template_id, version),
            ).fetchone()
            if not row:
                raise ValueError("prompt version not found")
            template = str(row[0])
            now = _timestamp()
            conn.execute(
                """
                UPDATE prompt_templates
                SET template = ?, updated_at = ?
                WHERE id = ?
                """,
                (template, now, template_id),
            )
            next_version = _next_version(conn, template_id)
            conn.execute(
                """
                INSERT INTO prompt_versions (
                  id, template_id, version, template, changed_by, change_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    template_id,
                    next_version,
                    template,
                    changed_by,
                    reason,
                    now,
                ),
            )

    def _list_enabled_by_key(self, key: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM prompt_templates
                WHERE key = ? AND enabled = 1
                """,
                (key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _find_template(
        self,
        key: str,
        locale: str,
        address_style: str,
        tone: str,
        channel: str,
        variant: str,
        policy_tier: str,
    ) -> dict[str, Any] | None:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT *
                FROM prompt_templates
                WHERE key = ? AND locale = ? AND address_style = ? AND tone = ?
                  AND channel = ? AND variant = ? AND policy_tier = ?
                """,
                (key, locale, address_style, tone, channel, variant, policy_tier),
            ).fetchone()
        return dict(row) if row else None


class NullPromptStore:
    def get_template(self, key: str, context: PromptContext) -> PromptMatch | None:
        return None

    def upsert_template(self, **_: Any) -> str:
        return ""

    def list_templates(self, key: str | None = None, enabled_only: bool = False) -> list[dict[str, Any]]:
        return []

    def rollback_template(
        self, template_id: str, version: int, changed_by: str, reason: str | None = None
    ) -> None:
        return None


def seed_default_templates(db_path: str | None = None) -> None:
    store = SqlitePromptStore(db_path=db_path)
    seeds = [
        ("core.greeting", "en", "any", "any", "any", "default", "safe", "Hi! How can I help?"),
        ("core.greeting", "es", "any", "any", "any", "default", "safe", "¡Hola! ¿En qué te ayudo?"),
        (
            "core.identity.agent",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "I'm Alphonse, your assistant. I only know this authorized chat.",
        ),
        (
            "core.identity.agent",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Soy Alphonse, tu asistente. Solo conozco este chat autorizado.",
        ),
        (
            "core.identity.user.ask_name",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "I don't know your name yet. Tell me what you'd like me to call you.",
        ),
        (
            "core.identity.user.ask_name",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Aún no sé tu nombre. Dime cómo quieres que te llame.",
        ),
        (
            "core.identity.user.known",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Yes, your name is {user_name}.",
        ),
        (
            "core.identity.user.known",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Sí, te llamas {user_name}.",
        ),
        (
            "clarify.intent",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "I'm not sure what you mean yet. What would you like to do?",
        ),
        (
            "clarify.intent",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "No estoy seguro de a qué te refieres. ¿Qué te gustaría hacer?",
        ),
        (
            "clarify.slot_abort",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "We can cancel this or try again later. What would you prefer?",
        ),
        (
            "clarify.slot_abort",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Puedo cancelar esto o intentarlo más tarde. ¿Qué prefieres?",
        ),
        (
            "ack.cancelled",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Okay, I cancelled that.",
        ),
        (
            "ack.cancelled",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Listo, lo cancelé.",
        ),
        (
            "intent_detector.rules.v1",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Rules:\n"
            "- Only choose intents from the catalog or \"unknown\".\n"
            "- Prefer \"unknown\" when ambiguous or too short.\n"
            "- Do not choose timed_signals.list unless the user is asking about reminders.\n"
            "- Do not choose timed_signals.create unless the user requests a reminder.\n"
            "- Confidence: high only when clear, low when uncertain.",
        ),
        (
            "intent_detector.rules.v1",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Reglas:\n"
            "- Solo elige intents del catálogo o \"unknown\".\n"
            "- Prefiere \"unknown\" cuando sea ambiguo o muy corto.\n"
            "- No elijas timed_signals.list a menos que pregunte por recordatorios.\n"
            "- No elijas timed_signals.create a menos que pida un recordatorio.\n"
            "- Confianza: alta solo cuando sea claro, baja cuando sea incierto.",
        ),
        (
            "intent_detector.catalog.prompt.v1",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "{rules_block}\n\n"
            "Catalog:\n{catalog_json}\n\n"
            "User message: {user_message}\n"
            "Return strict JSON: intent_name, confidence, slot_guesses, needs_clarification.",
        ),
        (
            "intent_detector.catalog.prompt.v1",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "{rules_block}\n\n"
            "Catálogo:\n{catalog_json}\n\n"
            "Mensaje del usuario: {user_message}\n"
            "Devuelve JSON estricto: intent_name, confidence, slot_guesses, needs_clarification.",
        ),
        (
            "clarify.reminder_text",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            'What should I remind you about? Example: "drink water".',
        ),
        (
            "clarify.reminder_text",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            '¿Qué debo recordarte? Ejemplo: "tomar agua".',
        ),
        (
            "clarify.trigger_time",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            'When should I remind you? Example: "in 10 min" or "at 7pm".',
        ),
        (
            "clarify.trigger_time",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            '¿Cuándo debo recordártelo? Ejemplo: "en 10 min" o "a las 7".',
        ),
        (
            "clarify.trigger_geo.stub_setup",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "I can do location-based reminders once home is set up. Want to switch to a time-based reminder?",
        ),
        (
            "clarify.trigger_geo.stub_setup",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Puedo hacer recordatorios por ubicación cuando casa esté configurada. ¿Quieres usar un recordatorio por hora?",
        ),
        (
            "ack.reminder_scheduled",
            "en",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Got it. Reminder scheduled.",
        ),
        (
            "ack.reminder_scheduled",
            "es",
            "any",
            "any",
            "any",
            "default",
            "safe",
            "Listo, programé el recordatorio.",
        ),
    ]
    for key, locale, address_style, tone, channel, variant, policy_tier, template in seeds:
        try:
            store.upsert_template(
                key=key,
                locale=locale,
                address_style=address_style,
                tone=tone,
                channel=channel,
                variant=variant,
                policy_tier=policy_tier,
                template=template,
                enabled=True,
                priority=0,
                changed_by="seed",
                reason="initial_seed",
            )
        except Exception:
            logger.exception("prompt seed failed key=%s locale=%s", key, locale)


def _score_template(row: dict[str, Any], context: PromptContext) -> float:
    score = 0.0
    score += _score_locale(row.get("locale"), context.locale)
    score += _score_exact(row.get("address_style"), context.address_style)
    score += _score_exact(row.get("tone"), context.tone)
    score += _score_exact(row.get("channel"), context.channel)
    score += _score_variant(row.get("variant"), context.variant)
    score += _score_exact(row.get("policy_tier"), context.policy_tier)
    return score


def _score_locale(template_locale: str | None, requested: str | None) -> float:
    if not template_locale:
        return 0.0
    tpl = template_locale.lower()
    if not requested:
        return 1.0 if tpl == "any" else 1.0
    req = requested.lower()
    if tpl == req:
        return 3.0
    req_lang = req.split("-")[0]
    if tpl == req_lang:
        return 2.0
    if tpl == "any":
        return 1.0
    return 0.0


def _score_exact(template_value: str | None, requested: str | None) -> float:
    if not template_value:
        return 0.0
    tpl = template_value.lower()
    if not requested:
        return 1.0 if tpl == "any" else 1.0
    req = requested.lower()
    if tpl == req:
        return 2.0
    if tpl == "any":
        return 1.0
    return 0.0


def _score_variant(template_variant: str | None, requested: str | None) -> float:
    if not template_variant:
        return 0.0
    tpl = template_variant.lower()
    if not requested:
        return 1.0 if tpl in {"any", "default"} else 1.0
    req = requested.lower()
    if tpl == req:
        return 2.0
    if tpl in {"any", "default"}:
        return 1.0
    return 0.0


def _tie_breaker(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    if candidate.get("priority", 0) != current.get("priority", 0):
        return candidate.get("priority", 0) > current.get("priority", 0)
    return str(candidate.get("updated_at") or "") > str(current.get("updated_at") or "")


def _next_version(conn: sqlite3.Connection, template_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(version) FROM prompt_versions WHERE template_id = ?",
        (template_id,),
    ).fetchone()
    current = row[0] if row and row[0] is not None else 0
    return int(current) + 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
