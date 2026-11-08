"""Offline fencing for managed account authority in a staged restored database.

This is a low-level recovery primitive, not a general backup/restore workflow.
Call it only on an offline disposable restore copy before exposing the copy to a
runtime. It never contacts providers or alters the credential backend.
"""
from datetime import datetime, timezone
import json


def fence_restored_accounts(connection, *, now=None):
    """Fence every restored managed connection and revoke its restored grants.

    A database backup cannot prove which grants were later revoked. Treat all
    restored grants as revoked, preserve their IDs and scopes for reconciliation,
    and require explicit re-consent. This does not establish that external
    backups, queues, model sessions or copied private content have been deleted.
    """
    if connection.in_transaction:
        raise ValueError('account_recovery_requires_idle_transaction')
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('account_recovery_timezone_required')
    timestamp = stamp.astimezone(timezone.utc).isoformat()
    counts = {'connections': 0, 'grants': 0}
    try:
        connection.execute('BEGIN IMMEDIATE')
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'v2_integrations' not in tables:
            raise ValueError('account_recovery_missing_integrations')
        if 'v2_recovery_credential_references' in tables:
            raise ValueError('account_recovery_already_fenced')
        connection.execute('''CREATE TABLE v2_recovery_credential_references (
            integration_id TEXT PRIMARY KEY, credential_ref TEXT NOT NULL,
            credential_generation INTEGER NOT NULL, credential_deletion_pending INTEGER NOT NULL
        ) STRICT''')
        rows = connection.execute('''SELECT integration_id, connection_json, credential_ref,
            credential_generation, credential_deletion_pending FROM v2_integrations''').fetchall()
        managed = set()
        for integration_id, raw, credential_ref, generation, deletion_pending in rows:
            value = json.loads(raw or '{}')
            if not isinstance(value, dict):
                raise ValueError('account_recovery_invalid_connection')
            if not value:
                continue
            if (type(value.get('revision')) is not int or value['revision'] < 1
                    or not isinstance(value.get('owner_user_id'), str) or not value['owner_user_id']):
                raise ValueError('account_recovery_invalid_connection')
            managed.add(integration_id)
            # A restored reference can point at the live host's current Keychain
            # entry. Preserve it for reconciliation, but detach it from normal
            # credential replacement/deletion paths in this isolated copy.
            if (not isinstance(credential_ref, str) or type(generation) is not int
                    or type(deletion_pending) is not int or deletion_pending not in (0, 1)):
                raise ValueError('account_recovery_invalid_credential_reference')
            connection.execute('INSERT INTO v2_recovery_credential_references VALUES (?, ?, ?, ?)',
                               (integration_id, credential_ref, generation, deletion_pending))
            value.update(status='reauth_required', health_code='restore_reconciliation_required',
                         revision=value['revision'] + 1, updated_at=timestamp)
            connection.execute('''UPDATE v2_integrations SET enabled=0, connection_json=?, updated_at=?,
                               credential_ref='', credential_deletion_pending=0 WHERE integration_id=?''',
                               (json.dumps(value, sort_keys=True), timestamp, integration_id))
            counts['connections'] += 1
        if managed and 'v2_account_grants' not in tables:
            raise ValueError('account_recovery_missing_grants')
        if 'v2_account_grants' in tables:
            rows = connection.execute('SELECT grant_id, connection_id, revision, definition FROM v2_account_grants').fetchall()
            for grant_id, connection_id, revision, raw in rows:
                grant = json.loads(raw)
                if (not isinstance(grant, dict) or grant.get('grant_id') != grant_id
                        or grant.get('connection_id') != connection_id
                        or type(grant.get('revision')) is not int or grant.get('revision') != revision
                        or connection_id not in managed):
                    raise ValueError('account_recovery_invalid_grant')
                # Revocation stays irreversible for this grant ID, even after reconnect.
                grant.update(revoked_at=grant.get('revoked_at') or timestamp,
                             updated_at=timestamp, revision=revision + 1)
                connection.execute('UPDATE v2_account_grants SET revision=?, definition=? WHERE grant_id=?',
                                   (revision + 1, json.dumps(grant), grant_id))
                counts['grants'] += 1
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return counts


def prepare_restored_database(source, destination):
    """Create an offline, account-fenced SQLite copy; never activate it.

    Destination must not already exist. The copy is made in a private staging
    file in the destination directory, verified and fenced before it is made
    visible at the requested path. This is not a complete household restore:
    queues, backup retention, credential reconciliation and consent remain
    separate requirements. Both paths are trusted operator inputs.
    """
    import os
    from pathlib import Path
    import sqlite3
    import stat
    import tempfile

    source_path = Path(source).expanduser().absolute()
    destination_path = Path(destination).expanduser().absolute()
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError('account_recovery_source_invalid')
    if not destination_path.parent.is_dir() or os.path.lexists(destination_path):
        raise ValueError('account_recovery_destination_unavailable')
    parent_info = destination_path.parent.stat()
    if (parent_info.st_uid != os.getuid() or stat.S_IMODE(parent_info.st_mode) & 0o077):
        raise ValueError('account_recovery_destination_directory_unprotected')
    descriptor, stage_name = tempfile.mkstemp(prefix='.alphonse-recovery-', suffix='.sqlite', dir=destination_path.parent)
    os.close(descriptor)
    stage = Path(stage_name)
    try:
        source_db = sqlite3.connect(source_path.as_uri() + '?mode=ro', uri=True)
        try:
            stage_db = sqlite3.connect(stage)
            try:
                source_db.backup(stage_db)
                if stage_db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('account_recovery_database_invalid')
                result = fence_restored_accounts(stage_db)
                stage_db.execute("CREATE TABLE IF NOT EXISTS v2_recovery_staging (state TEXT NOT NULL CHECK(state='isolated'))")
                stage_db.execute("INSERT INTO v2_recovery_staging VALUES ('isolated')")
                stage_db.commit()
                if stage_db.execute('PRAGMA journal_mode=DELETE').fetchone()[0] != 'delete':
                    raise ValueError('account_recovery_journal_unavailable')
            finally:
                stage_db.close()
        finally:
            source_db.close()
        fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        # Hard-link publication is atomic and fails rather than overwriting an
        # existing destination, including a symlink created during staging.
        os.link(stage, destination_path)
        parent_fd = os.open(destination_path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return result
    finally:
        stage.unlink(missing_ok=True)
        for suffix in ('-journal', '-wal', '-shm'):
            Path(str(stage) + suffix).unlink(missing_ok=True)
        # Once linked, the completed destination is deliberately retained even
        # if directory sync fails: it must not be mistaken for an activated DB.


def reject_staged_database(path):
    """Fail closed before startup if an isolated recovery copy is selected."""
    from pathlib import Path
    import sqlite3

    candidate = Path(path).expanduser().absolute()
    if not candidate.exists():
        return
    with sqlite3.connect(candidate.as_uri() + '?mode=ro', uri=True) as conn:
        marker = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_recovery_staging'").fetchone()
        if marker:
            raise RuntimeError('account_recovery_staged_database_not_activated')
