from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

ALLOWED_FACT_TYPES: set[str] = {
    "system_asset",
    "procedure",
    "workflow_rule",
    "location",
    "integration_note",
    "user_operational_preference",
}
ALLOWED_SCOPES: set[str] = {"private", "global"}


def upsert_operational_fact(
    *,
    created_by: str,
    key: str,
    title: str,
    fact_type: str,
    summary: str | None = None,
    content_json: Any = None,
    tags: list[str] | None = None,
    source: str | None = None,
    stability: str | None = None,
    importance: str | None = None,
    status: str | None = None,
    scope: str = "private",
    last_verified_at: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    owner = str(created_by or "").strip()
    if not owner:
        raise ValueError("created_by_required")
    fact_key = str(key or "").strip()
    if not fact_key:
        raise ValueError("key_required")
    fact_title = str(title or "").strip()
    if not fact_title:
        raise ValueError("title_required")
    fact_kind = str(fact_type or "").strip()
    if fact_kind not in ALLOWED_FACT_TYPES:
        raise ValueError("invalid_fact_type")
    fact_scope = str(scope or "private").strip().lower()
    if fact_scope not in ALLOWED_SCOPES:
        raise ValueError("invalid_scope")
    normalized_confidence = _normalize_confidence(confidence)
    encoded_content = _encode_content_json(content_json)
    encoded_tags = _encode_tags(tags)
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT id, created_by
            FROM operational_facts
            WHERE key = ?
            """,
            (fact_key,),
        ).fetchone()
        if row:
            existing_id = str(row[0] or "").strip()
            existing_owner = str(row[1] or "").strip()
            if existing_owner != owner:
                raise ValueError("fact_key_owned_by_other_user")
            conn.execute(
                """
                UPDATE operational_facts
                SET title = ?,
                    fact_type = ?,
                    summary = ?,
                    content_json = ?,
                    tags = ?,
                    source = ?,
                    stability = ?,
                    importance = ?,
                    status = ?,
                    scope = ?,
                    updated_at = ?,
                    last_verified_at = ?,
                    confidence = ?
                WHERE id = ?
                """,
                (
                    fact_title,
                    fact_kind,
                    _null_if_empty(summary),
                    encoded_content,
                    encoded_tags,
                    _null_if_empty(source),
                    _null_if_empty(stability),
                    _null_if_empty(importance),
                    _null_if_empty(status),
                    fact_scope,
                    now,
                    _null_if_empty(last_verified_at),
                    normalized_confidence,
                    existing_id,
                ),
            )
            conn.commit()
            return get_operational_fact_by_id(existing_id) or {}
        fact_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO operational_facts (
              id, key, title, fact_type, summary, content_json, tags, source, stability,
              importance, status, scope, created_by, created_at, updated_at, last_verified_at, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact_id,
                fact_key,
                fact_title,
                fact_kind,
                _null_if_empty(summary),
                encoded_content,
                encoded_tags,
                _null_if_empty(source),
                _null_if_empty(stability),
                _null_if_empty(importance),
                _null_if_empty(status),
                fact_scope,
                owner,
                now,
                now,
                _null_if_empty(last_verified_at),
                normalized_confidence,
            ),
        )
        conn.commit()
    return get_operational_fact_by_id(fact_id) or {}


def search_operational_facts(
    *,
    created_by: str,
    query: str | None = None,
    fact_type: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
    stability: str | None = None,
    importance: str | None = None,
    scope: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    owner = str(created_by or "").strip()
    if not owner:
        raise ValueError("created_by_required")
    fact_scope = str(scope or "").strip().lower()
    if fact_scope and fact_scope not in ALLOWED_SCOPES:
        raise ValueError("invalid_scope")
    limit_value = max(1, min(int(limit), 200))
    offset_value = max(0, int(offset))
    where_parts: list[str] = [
        "((scope = 'global') OR (scope = 'private' AND created_by = ?))",
    ]
    values: list[Any] = [owner]
    if fact_scope:
        where_parts.append("scope = ?")
        values.append(fact_scope)
    fact_kind = str(fact_type or "").strip()
    if fact_kind:
        if fact_kind not in ALLOWED_FACT_TYPES:
            raise ValueError("invalid_fact_type")
        where_parts.append("fact_type = ?")
        values.append(fact_kind)
    status_value = str(status or "").strip()
    if status_value:
        where_parts.append("status = ?")
        values.append(status_value)
    stability_value = str(stability or "").strip()
    if stability_value:
        where_parts.append("stability = ?")
        values.append(stability_value)
    importance_value = str(importance or "").strip()
    if importance_value:
        where_parts.append("importance = ?")
        values.append(importance_value)
    rendered_query = str(query or "").strip().lower()
    if rendered_query:
        like = f"%{rendered_query}%"
        where_parts.append(
            """
            (
              lower(key) LIKE ? OR
              lower(title) LIKE ? OR
              lower(COALESCE(summary, '')) LIKE ? OR
              lower(COALESCE(content_json, '')) LIKE ? OR
              lower(COALESCE(tags, '')) LIKE ?
            )
            """
        )
        values.extend([like, like, like, like, like])
    requested_tags = [str(item or "").strip().lower() for item in (tags or []) if str(item or "").strip()]
    if requested_tags:
        tag_expr = " OR ".join(["lower(COALESCE(tags, '')) LIKE ?" for _ in requested_tags])
        where_parts.append(f"({tag_expr})")
        values.extend([f'%"{item}"%' for item in requested_tags])
    sql = (
        """
        SELECT id, key, title, fact_type, summary, content_json, tags, source, stability,
               importance, status, scope, created_by, created_at, updated_at, last_verified_at, confidence
        FROM operational_facts
        WHERE
        """
        + " AND ".join(where_parts)
        + " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    )
    values.extend([limit_value, offset_value])
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(sql, tuple(values)).fetchall()
    return [_row_to_fact(row) for row in rows]


def remove_operational_fact(
    *,
    created_by: str,
    fact_id: str | None = None,
    key: str | None = None,
) -> bool:
    owner = str(created_by or "").strip()
    if not owner:
        raise ValueError("created_by_required")
    rendered_id = str(fact_id or "").strip()
    rendered_key = str(key or "").strip()
    if not rendered_id and not rendered_key:
        raise ValueError("id_or_key_required")
    where_parts: list[str] = []
    values: list[Any] = []
    if rendered_id:
        where_parts.append("id = ?")
        values.append(rendered_id)
    if rendered_key:
        where_parts.append("key = ?")
        values.append(rendered_key)
    where = " OR ".join(where_parts)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            f"DELETE FROM operational_facts WHERE ({where}) AND created_by = ?",
            (*values, owner),
        )
        conn.commit()
    return cur.rowcount > 0


def get_operational_fact_by_id(fact_id: str) -> dict[str, Any] | None:
    rendered = str(fact_id or "").strip()
    if not rendered:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT id, key, title, fact_type, summary, content_json, tags, source, stability,
                   importance, status, scope, created_by, created_at, updated_at, last_verified_at, confidence
            FROM operational_facts
            WHERE id = ?
            """,
            (rendered,),
        ).fetchone()
    return _row_to_fact(row) if row else None


def _row_to_fact(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "id": row[0],
        "key": row[1],
        "title": row[2],
        "fact_type": row[3],
        "summary": row[4],
        "content_json": _parse_json(row[5]),
        "tags": _parse_tags(row[6]),
        "source": row[7],
        "stability": row[8],
        "importance": row[9],
        "status": row[10],
        "scope": row[11],
        "created_by": row[12],
        "created_at": row[13],
        "updated_at": row[14],
        "last_verified_at": row[15],
        "confidence": row[16],
    }


def _encode_content_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        except json.JSONDecodeError:
            return json.dumps(text, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _encode_tags(value: list[str] | None) -> str | None:
    rows = [str(item or "").strip().lower() for item in (value or []) if str(item or "").strip()]
    if not rows:
        return None
    return json.dumps(sorted(set(rows)), ensure_ascii=False)


def _parse_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item or "").strip() for item in parsed if str(item or "").strip()]
        return []
    if isinstance(raw, list):
        return [str(item or "").strip() for item in raw if str(item or "").strip()]
    return []


def _parse_json(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    return raw


def _normalize_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    rendered = float(value)
    if rendered < 0 or rendered > 1:
        raise ValueError("invalid_confidence")
    return rendered


def _null_if_empty(value: str | None) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
