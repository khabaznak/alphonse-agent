from __future__ import annotations

import logging
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
        best = _pick_best_match(rows, key, context, relaxed=False)
        if best is None and not _is_sensitive_prompt_key(key):
            best = _pick_best_match(rows, key, context, relaxed=True)
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

    def is_available(self) -> bool:
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_templates'"
                ).fetchone()
                if not row:
                    return False
                columns = {
                    col[1] for col in conn.execute("PRAGMA table_info(prompt_templates)")
                }
                versions = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_versions'"
                ).fetchone()
        except sqlite3.Error:
            return False
        required = {
            "id",
            "key",
            "locale",
            "address_style",
            "tone",
            "channel",
            "variant",
            "policy_tier",
            "template",
            "enabled",
            "priority",
            "created_at",
            "updated_at",
        }
        return bool(versions) and required.issubset(columns)

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
    for seed in _load_seed_templates():
        try:
            store.upsert_template(
                key=seed["key"],
                locale=seed["locale"],
                address_style=seed["address_style"],
                tone=seed["tone"],
                channel=seed["channel"],
                variant=seed["variant"],
                policy_tier=seed["policy_tier"],
                template=seed["template"],
                enabled=True,
                priority=0,
                changed_by="seed",
                reason="initial_seed",
            )
        except Exception:
            logger.exception(
                "prompt seed failed key=%s locale=%s",
                seed.get("key"),
                seed.get("locale"),
            )


def _load_seed_templates() -> list[dict[str, str]]:
    seed_path = (
        Path(__file__).resolve().parent.parent / "nervous_system" / "resources" / "prompt_templates.seed.json"
    )
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("prompt template seed payload must be a list")
    required = {"key", "locale", "address_style", "tone", "channel", "variant", "policy_tier", "template"}
    rows: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("prompt template seed row must be an object")
        if not required.issubset(item.keys()):
            missing = sorted(required - set(item.keys()))
            raise ValueError(f"prompt template seed row missing fields: {missing}")
        rows.append({name: str(item[name]) for name in required})
    return rows


def _score_template(row: dict[str, Any], context: PromptContext) -> float:
    score = 0.0
    score += _score_locale(row.get("locale"), context.locale)
    score += _score_exact(row.get("address_style"), context.address_style)
    score += _score_exact(row.get("tone"), context.tone)
    score += _score_exact(row.get("channel"), context.channel)
    score += _score_variant(row.get("variant"), context.variant)
    score += _score_exact(row.get("policy_tier"), context.policy_tier)
    return score


def _score_template_relaxed(
    row: dict[str, Any],
    key: str,
    context: PromptContext,
) -> float:
    score = 0.0
    score += _score_locale(row.get("locale"), context.locale)
    score += _score_variant(row.get("variant"), context.variant)
    if _is_sensitive_prompt_key(key):
        score += _score_exact(row.get("policy_tier"), context.policy_tier)
    else:
        score += _score_soft(row.get("address_style"), context.address_style)
        score += _score_soft(row.get("tone"), context.tone)
        score += _score_soft(row.get("channel"), context.channel)
        score += _score_soft(row.get("policy_tier"), context.policy_tier)
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


def _score_soft(template_value: str | None, requested: str | None) -> float:
    if not template_value:
        return 0.0
    tpl = template_value.lower()
    if not requested:
        return 0.6 if tpl == "any" else 0.4
    req = requested.lower()
    if tpl == req:
        return 1.0
    if tpl == "any":
        return 0.7
    # In relaxed mode we prefer exact/any selectors, but do not reject mismatches.
    return 0.2


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


def _pick_best_match(
    rows: list[dict[str, Any]],
    key: str,
    context: PromptContext,
    *,
    relaxed: bool,
) -> tuple[float, dict[str, Any]] | None:
    best: tuple[float, dict[str, Any]] | None = None
    strict_sensitive = (not relaxed) and _is_sensitive_prompt_key(key)
    for row in rows:
        if strict_sensitive and not _selectors_compatible(row, context):
            continue
        score = (
            _score_template_relaxed(row, key, context)
            if relaxed
            else _score_template(row, context)
        )
        if score <= 0:
            continue
        if best is None or score > best[0] or (
            score == best[0] and _tie_breaker(row, best[1])
        ):
            best = (score, row)
    return best


def _is_sensitive_prompt_key(key: str) -> bool:
    return key.startswith("policy.") or key.startswith("security.") or key.startswith(
        "error."
    )


def _selectors_compatible(row: dict[str, Any], context: PromptContext) -> bool:
    return (
        _selector_compatible(row.get("address_style"), context.address_style)
        and _selector_compatible(row.get("tone"), context.tone)
        and _selector_compatible(row.get("channel"), context.channel)
        and _selector_compatible(row.get("policy_tier"), context.policy_tier)
    )


def _selector_compatible(template_value: Any, requested: str | None) -> bool:
    if requested is None:
        return True
    if template_value is None:
        return True
    tpl = str(template_value).lower()
    req = str(requested).lower()
    return tpl == "any" or tpl == req


def _next_version(conn: sqlite3.Connection, template_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(version) FROM prompt_versions WHERE template_id = ?",
        (template_id,),
    ).fetchone()
    current = row[0] if row and row[0] is not None else 0
    return int(current) + 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
