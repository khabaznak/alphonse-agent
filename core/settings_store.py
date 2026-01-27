import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DB_DEFAULT_PATH = "data/atrium.db"


def init_db() -> None:
    conn = _get_connection()
    conn.execute(
        """
        create table if not exists settings (
            id integer primary key autoincrement,
            name text not null unique,
            description text,
            schema text,
            config text not null,
            created_at text not null,
            updated_at text not null
        )
        """
    )
    conn.commit()
    _seed_timezone(conn)
    conn.close()


def list_settings() -> list[dict[str, Any]]:
    conn = _get_connection()
    cursor = conn.execute(
        "select id, name, description, schema, config, created_at, updated_at from settings "
        "order by name"
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_setting(row) for row in rows]


def get_setting(setting_id: int) -> dict[str, Any] | None:
    conn = _get_connection()
    cursor = conn.execute(
        "select id, name, description, schema, config, created_at, updated_at from settings "
        "where id = ?",
        (setting_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return _row_to_setting(row) if row else None


def get_setting_by_name(name: str) -> dict[str, Any] | None:
    conn = _get_connection()
    cursor = conn.execute(
        "select id, name, description, schema, config, created_at, updated_at from settings "
        "where name = ?",
        (name,),
    )
    row = cursor.fetchone()
    conn.close()
    return _row_to_setting(row) if row else None


def get_timezone() -> str:
    setting = get_setting_by_name("timezone")
    if not setting:
        return "UTC"
    config = _parse_json(setting.get("config"))
    tz_name = config.get("tz") if isinstance(config, dict) else None
    if not isinstance(tz_name, str):
        return "UTC"
    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return "UTC"
    return tz_name


def create_setting(payload: dict[str, Any]) -> dict[str, Any]:
    now = _timestamp()
    conn = _get_connection()
    cursor = conn.execute(
        """
        insert into settings (name, description, schema, config, created_at, updated_at)
        values (?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("name"),
            payload.get("description"),
            payload.get("schema"),
            payload.get("config"),
            now,
            now,
        ),
    )
    conn.commit()
    setting_id = cursor.lastrowid
    conn.close()
    if not setting_id:
        return {}
    return get_setting(setting_id) or {}


def update_setting(setting_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    now = _timestamp()
    conn = _get_connection()
    conn.execute(
        """
        update settings
        set name = ?, description = ?, schema = ?, config = ?, updated_at = ?
        where id = ?
        """,
        (
            payload.get("name"),
            payload.get("description"),
            payload.get("schema"),
            payload.get("config"),
            now,
            setting_id,
        ),
    )
    conn.commit()
    conn.close()
    return get_setting(setting_id) or {}


def delete_setting(setting_id: int) -> None:
    conn = _get_connection()
    conn.execute("delete from settings where id = ?", (setting_id,))
    conn.commit()
    conn.close()


def _get_connection() -> sqlite3.Connection:
    path = Path(os.getenv("ATRIUM_DB_PATH", DB_DEFAULT_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def _row_to_setting(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "schema": row[3],
        "config": row[4],
        "created_at": row[5],
        "updated_at": row[6],
    }


def _seed_timezone(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("select id from settings where name = ?", ("timezone",))
    if cursor.fetchone():
        return
    now = _timestamp()
    config = json.dumps({"tz": "America/Chicago"})
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
        }
    )
    conn.execute(
        """
        insert into settings (name, description, schema, config, created_at, updated_at)
        values (?, ?, ?, ?, ?, ?)
        """,
        (
            "timezone",
            "Local timezone for Rex interpretations",
            schema,
            config,
            now,
            now,
        ),
    )
    conn.commit()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
