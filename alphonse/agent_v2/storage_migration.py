"""One-time migration from per-store v2 databases into the unified database."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from alphonse.agent_v2.database import (
    connect_database,
    default_database_path,
    migration_applied,
    record_migration,
)

LEGACY_IMPORT_MIGRATION_ID = "2026-07-30-unified-v2-database"
UNIFIED_SCHEMA_MIGRATION_ID = "2026-07-30-unified-schema-v1"

_LEGACY_DATABASES: tuple[tuple[str, str], ...] = (
    ("ALPHONSE_V2_USERS_DB_PATH", "v2-users.sqlite3"),
    ("ALPHONSE_V2_MESSAGES_DB_PATH", "v2-messages.sqlite3"),
    ("ALPHONSE_V2_QUESTION_DB_PATH", "v2-questions.sqlite3"),
    ("ALPHONSE_V2_PROJECT_DB_PATH", "v2-projects.sqlite3"),
    ("ALPHONSE_V2_PROJECT_SESSION_DB_PATH", "v2-project-sessions.sqlite3"),
    ("ALPHONSE_V2_SCHEDULE_DB_PATH", "v2-scheduled-tasks.sqlite3"),
    ("ALPHONSE_V2_OUTBOX_DB_PATH", "v2-outbox.sqlite3"),
    ("ALPHONSE_V2_CONVERSATIONS_DB_PATH", "v2-conversations.sqlite3"),
    ("ALPHONSE_V2_INTEGRATIONS_DB_PATH", "v2-integrations.sqlite3"),
    ("ALPHONSE_V2_INFERENCE_DB_PATH", "v2-inference.sqlite3"),
    ("ALPHONSE_V2_WEB_TOOLS_DB_PATH", "v2-web-tools.sqlite3"),
    ("ALPHONSE_V2_MEDIA_TOOLS_DB_PATH", "v2-media-tools.sqlite3"),
    ("ALPHONSE_V2_MEMORY_SETTINGS_DB_PATH", "v2-memory-settings.sqlite3"),
    ("ALPHONSE_V2_ASSETS_DB_PATH", "v2-assets.sqlite3"),
    ("ALPHONSE_V2_ARTIFACTS_DB_PATH", "v2-artifacts.sqlite3"),
    ("ALPHONSE_V2_AUTOMATIONS_DB_PATH", "v2-automations.sqlite3"),
    ("ALPHONSE_V2_COMMUNICATION_THREADS_DB_PATH", "v2-communication-threads.sqlite3"),
)


def legacy_database_paths() -> tuple[Path, ...]:
    base = Path.home() / ".alphonse"
    paths: list[Path] = []
    seen: set[Path] = set()
    for environment_name, filename in _LEGACY_DATABASES:
        configured = str(os.getenv(environment_name) or "").strip()
        path = (Path(configured).expanduser() if configured else base / filename).resolve()
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return tuple(paths)


def initialize_unified_schema(target: str | Path) -> None:
    """Create every relational schema in one file without changing store APIs."""
    from alphonse.agent_v2.artifacts import SQLiteArtifactStore
    from alphonse.agent_v2.assets import SQLiteAssetStore
    from alphonse.agent_v2.automations import EventAutomationStore
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.io.communication import SQLiteCommunicationThreadStore
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    from alphonse.agent_v2.core.projects import ProjectStore
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
    from alphonse.agent_v2.inference_settings import SQLiteInferenceSettingsStore
    from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
    from alphonse.agent_v2.media_tools_settings import SQLiteMediaToolsSettingsStore
    from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore
    from alphonse.agent_v2.intelligence_engine_settings import SQLiteIntelligenceEngineSettingsStore
    from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore
    from alphonse.agent_v2.users import V2UserStore
    from alphonse.agent_v2.web_tools_settings import SQLiteWebToolsSettingsStore
    from alphonse.agent_v2.system_one import SQLiteSystemOneSettingsStore

    path = str(Path(target).expanduser())
    SQLiteMessageQueue(path)
    SQLiteQuestionStore(path)
    ProjectStore(path)
    ScheduledTaskStore(path)
    SQLiteOutboundStore(path)
    SQLiteConversationStore(path)
    SQLiteIntegrationStore(path)
    SQLiteInferenceSettingsStore(path)
    SQLiteWebToolsSettingsStore(path)
    SQLiteMediaToolsSettingsStore(path)
    SQLiteMemorySettingsStore(path)
    SQLiteIntelligenceEngineSettingsStore(path)
    SQLiteSystemOneSettingsStore(path)
    SQLiteProjectSessionStore(path)
    V2UserStore(path)
    SQLiteAssetStore(path)
    SQLiteArtifactStore(path)
    EventAutomationStore(path)
    SQLiteCommunicationThreadStore(path)
    with connect_database(path) as connection:
        record_migration(connection, UNIFIED_SCHEMA_MIGRATION_ID)


def migrate_legacy_databases(
    *,
    target: str | Path | None = None,
    sources: Iterable[str | Path] | None = None,
    backup_root: str | Path | None = None,
) -> dict[str, object]:
    target_path = Path(target or default_database_path()).expanduser().resolve()
    explicit_sources = tuple(sources) if sources is not None else None
    # A caller explicitly requesting no sources has not performed the global
    # legacy import. Do not create a completion marker that would suppress a
    # later normal discovery, or initialize a database for this no-op.
    if explicit_sources == ():
        return {'status': 'no_sources', 'target': str(target_path), 'sources': []}
    initialize_unified_schema(target_path)
    candidates = tuple(Path(item).expanduser().resolve() for item in
                       (legacy_database_paths() if explicit_sources is None else explicit_sources))
    legacy_sources = tuple(path for path in candidates if path.exists() and path != target_path)

    with connect_database(target_path) as connection:
        if migration_applied(connection, LEGACY_IMPORT_MIGRATION_ID):
            return {
                "status": "already_applied",
                "target": str(target_path),
                "sources": [str(path) for path in legacy_sources],
            }

    backup_directory = _backup_legacy_files(
        legacy_sources,
        backup_root=Path(backup_root).expanduser() if backup_root is not None else target_path.parent / "backups",
    )
    imported: dict[str, int] = {}
    connection = connect_database(target_path)
    connection.create_function('migration_source_digest', 1,
        lambda value: hashlib.sha256(value.encode()).hexdigest() if isinstance(value, str) else None,
        deterministic=True)
    try:
        for index, source in enumerate(legacy_sources):
            alias = f"legacy_{index}"
            connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(source),))
            try:
                connection.execute("BEGIN IMMEDIATE")
                derived_fts = _verified_conversation_fts_tables(connection, alias)
                if derived_fts:
                    _validate_conversation_index(connection, alias)
                for table in _source_tables(connection, alias):
                    if table in derived_fts:
                        continue
                    copied = _copy_table(connection, alias=alias, table=table)
                    imported[table] = imported.get(table, 0) + copied
                _validate_conversation_tombstones(connection, alias)
                _validate_conversation_tombstones(connection, 'main')
                if 'v2_conversation_events' in _source_tables(connection, alias):
                    _verified_conversation_fts_tables(connection, 'main', required=True)
                    _validate_conversation_index(connection, 'main')
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.execute(f"DETACH DATABASE {alias}")
        connection.execute("BEGIN IMMEDIATE")
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"unified_database_integrity_failed: {integrity}")
        record_migration(
            connection,
            LEGACY_IMPORT_MIGRATION_ID,
            details_json=json.dumps(
                {
                    "sources": [str(path) for path in legacy_sources],
                    "backup_directory": str(backup_directory) if backup_directory is not None else "",
                    "imported_rows": imported,
                },
                sort_keys=True,
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {
        "status": "migrated",
        "target": str(target_path),
        "sources": [str(path) for path in legacy_sources],
        "backup_directory": str(backup_directory) if backup_directory is not None else "",
        "imported_rows": imported,
    }


def _backup_legacy_files(sources: tuple[Path, ...], *, backup_root: Path) -> Path | None:
    if not sources:
        return None
    # Different configured directories can contain the same filename. Copying
    # them into one directory would silently overwrite the earlier backup.
    names: set[str] = set()
    for source in sources:
        paths = [source, *(Path(f"{source}{suffix}") for suffix in ("-wal", "-shm"))]
        for path in paths:
            if path.exists():
                if path.name in names:
                    raise RuntimeError("unified_migration_backup_name_conflict")
                names.add(path.name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = backup_root / f"pre-unified-{stamp}"
    suffix = 1
    while destination.exists():
        destination = backup_root / f"pre-unified-{stamp}-{suffix}"
        suffix += 1
    destination.mkdir(parents=True, exist_ok=False)
    for source in sources:
        shutil.copy2(source, destination / source.name)
        for suffix in ("-wal", "-shm"):
            companion = Path(f"{source}{suffix}")
            if companion.exists():
                shutil.copy2(companion, destination / companion.name)
    return destination


def _source_tables(connection: sqlite3.Connection, alias: str) -> list[str]:
    rows = connection.execute(
        f"""
        SELECT name FROM {alias}.sqlite_master
        WHERE type = 'table' AND name LIKE 'v2_%' AND name != 'v2_schema_migrations'
        ORDER BY name
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def _verified_conversation_fts_tables(connection: sqlite3.Connection, alias: str, *, required: bool = False) -> set[str]:
    """Identify only the known FTS5 index and SQLite-managed shadow tables.

    Names alone are not evidence that a table is disposable derived data.
    Unexpected or partially installed FTS schemas require reconciliation.
    """
    name = 'v2_conversation_fts'
    entries = {str(row[1]): str(row[2]) for row in connection.execute(f'PRAGMA {alias}.table_list')}
    shadows = {f'{name}_{suffix}' for suffix in ('data', 'idx', 'content', 'docsize', 'config')}
    if name not in entries:
        if required or any(item in entries for item in shadows):
            raise RuntimeError('unified_migration_conversation_fts_schema_invalid')
        return set()
    definition = connection.execute(
        f'SELECT sql FROM {alias}.sqlite_master WHERE type = ? AND name = ?', ('table', name)
    ).fetchone()
    sql = str(definition[0] or '') if definition else ''
    columns = [str(row[1]) for row in connection.execute(f'PRAGMA {alias}.table_info("{name}")')]
    if (entries[name] != 'virtual' or not re.search(r'\bUSING\s+fts5\s*\(', sql, re.IGNORECASE)
            or columns != ['event_id', 'owner_user_id', 'content']
            or any(entries.get(item) != 'shadow' for item in shadows)):
        raise RuntimeError('unified_migration_conversation_fts_schema_invalid')
    return {name, *shadows}


def _validate_conversation_index(connection: sqlite3.Connection, alias: str) -> None:
    """Require the searchable projection to equal authoritative event rows."""
    event_table = connection.execute(
        f'SELECT 1 FROM {alias}.sqlite_master WHERE type = ? AND name = ?',
        ('table', 'v2_conversation_events'),
    ).fetchone()
    if not event_table:
        raise RuntimeError('unified_migration_conversation_events_missing')
    # Compare grouped multiplicities, not merely membership or total counts.
    mismatch = connection.execute(f'''WITH
        events AS (SELECT event_id COLLATE BINARY, owner_user_id COLLATE BINARY, content COLLATE BINARY, COUNT(*) AS n
                   FROM {alias}.v2_conversation_events GROUP BY 1, 2, 3),
        idx AS (SELECT event_id COLLATE BINARY, owner_user_id COLLATE BINARY, content COLLATE BINARY, COUNT(*) AS n
                FROM {alias}.v2_conversation_fts GROUP BY 1, 2, 3)
        SELECT 1 FROM (
            SELECT * FROM events EXCEPT SELECT * FROM idx
            UNION ALL SELECT * FROM (SELECT * FROM idx EXCEPT SELECT * FROM events)
        ) LIMIT 1''').fetchone()
    if mismatch:
        raise RuntimeError('unified_migration_conversation_fts_mismatch')


def _validate_conversation_tombstones(connection: sqlite3.Connection, alias: str) -> None:
    tables = {str(row[0]) for row in connection.execute(
        f"SELECT name FROM {alias}.sqlite_master WHERE type='table' AND name IN "
        "('v2_conversation_events','v2_conversation_tombstones')")}
    if len(tables) != 2:
        return
    if connection.execute(f'''SELECT 1 FROM {alias}.v2_conversation_events e
        JOIN {alias}.v2_conversation_tombstones t ON t.event_id=e.event_id LIMIT 1''').fetchone():
        raise RuntimeError('unified_migration_conversation_tombstone_conflict')
    columns = {str(row[1]) for row in connection.execute(f'PRAGMA {alias}.table_info("v2_conversation_events")')}
    if 'source_message_id' in columns:
        if connection.execute(f'''SELECT 1 FROM {alias}.v2_conversation_events e
            JOIN {alias}.v2_conversation_tombstones t
            ON t.source_digest=migration_source_digest(e.source_message_id)
            WHERE e.source_message_id <> '' LIMIT 1''').fetchone():
            raise RuntimeError('unified_migration_conversation_tombstone_conflict')


def _copy_table(connection: sqlite3.Connection, *, alias: str, table: str) -> int:
    destination_exists = connection.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if destination_exists is None:
        raise RuntimeError(f"unified_schema_missing_table: {table}")
    destination_info = connection.execute(f'PRAGMA main.table_info("{table}")').fetchall()
    destination_columns = [str(row[1]) for row in destination_info]
    source_columns = {str(row[1]) for row in connection.execute(f'PRAGMA {alias}.table_info("{table}")')}
    columns = [column for column in destination_columns if column in source_columns]
    source_count = int(connection.execute(f'SELECT COUNT(*) FROM {alias}."{table}"').fetchone()[0])
    primary_key = [
        str(item[1])
        for item in sorted(destination_info, key=lambda row: int(row[5]))
        if int(item[5]) > 0
    ]
    if source_count and set(primary_key) - source_columns:
        raise RuntimeError(f"unified_migration_source_key_missing: {table}")
    # Destination-only columns may be newer optional/defaulted fields. In the
    # other direction, silently dropping old source fields would lose data.
    if source_count and source_columns - set(destination_columns):
        raise RuntimeError(f"unified_migration_source_columns_unmapped: {table}")
    if not columns:
        return 0
    quoted = ", ".join(f'"{column}"' for column in columns)
    before = int(connection.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()[0])
    # Earlier sources are committed individually. Without a key or a durable
    # per-source receipt, existing target rows could be a previous partial
    # import or independent records. Neither skipping nor inserting is safe.
    if source_count and not primary_key and before:
        raise RuntimeError(f"unified_migration_unkeyed_target_not_empty: {table}")
    connection.execute(
        f'INSERT OR IGNORE INTO main."{table}" ({quoted}) SELECT {quoted} FROM {alias}."{table}"'
    )
    after = int(connection.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()[0])
    if after < source_count:
        raise RuntimeError(f"unified_migration_row_validation_failed: {table}")
    if primary_key:
        predicate = " AND ".join(f'd."{column}" IS s."{column}"' for column in primary_key)
        missing = int(
            connection.execute(
                f"""
                SELECT COUNT(*) FROM {alias}."{table}" AS s
                WHERE NOT EXISTS (
                  SELECT 1 FROM main."{table}" AS d WHERE {predicate}
                )
                """
            ).fetchone()[0]
        )
        if missing:
            raise RuntimeError(f"unified_migration_identifier_validation_failed: {table}")
        # An existing key alone is not proof that INSERT OR IGNORE preserved
        # the source row: a conflicting destination row may have won. Compare
        # every shared column, including NULL values, before marking migration
        # complete. Leave divergent copies for explicit reconciliation.
        matching_values = " AND ".join(
            f'(typeof(d."{column}") = typeof(s."{column}") AND d."{column}" COLLATE BINARY IS s."{column}")'
            for column in columns)
        conflicting = int(connection.execute(
            f'''SELECT COUNT(*) FROM {alias}."{table}" AS s
                JOIN main."{table}" AS d ON {predicate}
                WHERE NOT ({matching_values})'''
        ).fetchone()[0])
        if conflicting:
            raise RuntimeError(f"unified_migration_row_conflict: {table}")
    elif source_count:
        # Without a primary key, validate the multiplicity of each complete
        # shared row. A destination can have enough total rows while a UNIQUE
        # conflict caused a different source row to be ignored.
        aliases = [f'c{index}' for index in range(len(columns))]
        projected = ', '.join(f'"{column}" COLLATE BINARY AS "{key}", typeof("{column}") AS "{key}_type"' for column, key in zip(columns, aliases))
        group_by = ', '.join(f'"{column}" COLLATE BINARY, typeof("{column}")' for column in columns)
        match = ' AND '.join(f'(d."{key}" COLLATE BINARY IS s."{key}" AND d."{key}_type" = s."{key}_type")' for key in aliases)
        missing_rows = connection.execute(
            f'''WITH source_counts AS (
                    SELECT {projected}, COUNT(*) AS n FROM {alias}."{table}" GROUP BY {group_by}
                ), target_counts AS (
                    SELECT {projected}, COUNT(*) AS n FROM main."{table}" GROUP BY {group_by}
                )
                SELECT 1 FROM source_counts s LEFT JOIN target_counts d ON {match}
                WHERE d.n IS NULL OR d.n < s.n LIMIT 1'''
        ).fetchone()
        if missing_rows:
            raise RuntimeError(f"unified_migration_row_validation_failed: {table}")
    return max(0, after - before)
