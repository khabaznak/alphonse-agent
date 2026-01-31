from __future__ import annotations

import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def get_person(person_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM persons WHERE person_id = ?", (person_id,))


def list_groups() -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM groups WHERE is_active = 1", ())


def list_person_groups(person_id: str) -> list[dict[str, Any]]:
    query = """
        SELECT g.* FROM groups g
        JOIN person_groups pg ON pg.group_id = g.group_id
        WHERE pg.person_id = ? AND g.is_active = 1
    """
    return _fetch_all(query, (person_id,))


def list_channels_for_person(person_id: str, channel_type: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM channels WHERE person_id = ? AND is_enabled = 1"
    params: list[Any] = [person_id]
    if channel_type:
        query += " AND channel_type = ?"
        params.append(channel_type)
    query += " ORDER BY priority ASC"
    return _fetch_all(query, tuple(params))


def list_channels_for_group(group_id: str, channel_type: str | None = None) -> list[dict[str, Any]]:
    query = """
        SELECT c.* FROM channels c
        JOIN person_groups pg ON pg.person_id = c.person_id
        WHERE pg.group_id = ? AND c.is_enabled = 1
    """
    params: list[Any] = [group_id]
    if channel_type:
        query += " AND c.channel_type = ?"
        params.append(channel_type)
    query += " ORDER BY c.priority ASC"
    return _fetch_all(query, tuple(params))


def get_channel(channel_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM channels WHERE channel_id = ?", (channel_id,))


def resolve_person_by_channel(channel_type: str, address: str) -> dict[str, Any] | None:
    query = """
        SELECT p.* FROM persons p
        JOIN channels c ON c.person_id = p.person_id
        WHERE c.channel_type = ? AND c.address = ? AND p.is_active = 1
        ORDER BY c.priority ASC
        LIMIT 1
    """
    return _fetch_one(query, (channel_type, address))


def list_prefs_for_person(person_id: str) -> dict[str, Any] | None:
    query = """
        SELECT * FROM communication_prefs
        WHERE scope_type = 'person' AND scope_id = ?
        LIMIT 1
    """
    return _fetch_one(query, (person_id,))


def list_prefs_for_group(group_id: str) -> dict[str, Any] | None:
    query = """
        SELECT * FROM communication_prefs
        WHERE scope_type = 'group' AND scope_id = ?
        LIMIT 1
    """
    return _fetch_one(query, (group_id,))


def get_presence(person_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM presence_state WHERE person_id = ?", (person_id,))


def _fetch_all(query: str, params: tuple) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def _fetch_one(query: str, params: tuple) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(query, params).fetchone()
    return dict(row) if row else None


def _connect() -> sqlite3.Connection:
    path = resolve_nervous_system_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
