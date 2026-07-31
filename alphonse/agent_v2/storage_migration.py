"""One-time migration from per-store v2 databases into the unified database."""

from __future__ import annotations

import json
import os
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
    from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore
    from alphonse.agent_v2.users import V2UserStore
    from alphonse.agent_v2.web_tools_settings import SQLiteWebToolsSettingsStore

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
    initialize_unified_schema(target_path)
    candidates = tuple(Path(item).expanduser().resolve() for item in (sources or legacy_database_paths()))
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
    try:
        for index, source in enumerate(legacy_sources):
            alias = f"legacy_{index}"
            connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(source),))
            try:
                connection.execute("BEGIN IMMEDIATE")
                for table in _source_tables(connection, alias):
                    copied = _copy_table(connection, alias=alias, table=table)
                    imported[table] = imported.get(table, 0) + copied
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
    if not columns:
        return 0
    quoted = ", ".join(f'"{column}"' for column in columns)
    before = int(connection.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()[0])
    connection.execute(
        f'INSERT OR IGNORE INTO main."{table}" ({quoted}) SELECT {quoted} FROM {alias}."{table}"'
    )
    after = int(connection.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()[0])
    source_count = int(connection.execute(f'SELECT COUNT(*) FROM {alias}."{table}"').fetchone()[0])
    if after < source_count:
        raise RuntimeError(f"unified_migration_row_validation_failed: {table}")
    primary_key = [
        str(item[1])
        for item in sorted(destination_info, key=lambda row: int(row[5]))
        if int(item[5]) > 0 and str(item[1]) in source_columns
    ]
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
    return max(0, after - before)
