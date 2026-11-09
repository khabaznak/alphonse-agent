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
        if 'v2_oauth_client_secrets' in tables:
            if 'v2_recovery_client_secrets' in tables:
                raise ValueError('account_recovery_already_fenced')
            connection.execute('''CREATE TABLE v2_recovery_client_secrets (
                provider TEXT PRIMARY KEY, client_id TEXT NOT NULL,
                credential_ref TEXT NOT NULL, generation INTEGER NOT NULL) STRICT''')
            connection.execute('''INSERT INTO v2_recovery_client_secrets
                SELECT provider,client_id,credential_ref,generation FROM v2_oauth_client_secrets''')
            connection.execute('DELETE FROM v2_oauth_client_secrets')
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


def inspect_staged_database(path):
    """Count recovery-sensitive rows without exposing their content or activating the copy.

    Missing tables are reported as unknown, never as zero. These counts are an
    inventory for reconciliation, not proof that a restore is safe to activate.
    """
    from pathlib import Path
    import sqlite3

    candidate = Path(path).expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError('account_recovery_source_invalid')
    # These table names are fixed by the application; never interpolate inputs.
    tables_to_count = {
        'managed_connections': 'v2_recovery_credential_references',
        'grants': 'v2_account_grants',
        'inbound_messages': 'v2_inbound_messages',
        'outbound_messages': 'v2_outbox',
        'scheduled_tasks': 'v2_scheduled_tasks',
        'scheduled_executions': 'v2_scheduled_task_executions',
        'automations': 'v2_automations',
        'automation_executions': 'v2_automation_executions',
        'project_check_policies': 'v2_project_check_policies',
        'project_check_runs': 'v2_project_check_runs',
        'event_history': 'v2_event_history',
        'event_receipts': 'v2_event_receipts',
    }
    with sqlite3.connect(candidate.as_uri() + '?mode=ro', uri=True) as conn:
        conn.execute('BEGIN')
        existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'v2_recovery_staging' not in existing or conn.execute(
                "SELECT COUNT(*) FROM v2_recovery_staging WHERE state='isolated'").fetchone()[0] != 1:
            raise ValueError('account_recovery_not_staged')
        return {label: (conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                        if table in existing else None)
                for label, table in tables_to_count.items()}


def apply_current_source_deletions(staged, current):
    """Apply trusted current source privacy state to an isolated copy.

    Deletions apply by source identity; revoked and expired states also apply to
    matching evidence IDs. Both paths must be offline, protected operator inputs.
    This copies no source content from current. It does not establish that the
    snapshot is current or complete, cover separate databases/backups, or authorize
    activation.
    """
    import os
    from pathlib import Path
    import sqlite3
    import stat

    stage_path = Path(staged).expanduser().absolute()
    current_path = Path(current).expanduser().absolute()
    if any(path.is_symlink() or not path.is_file() for path in (stage_path, current_path)):
        raise ValueError('account_recovery_source_invalid')
    info = stage_path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError('account_recovery_stage_unprotected')
    if os.path.samefile(stage_path, current_path):
        raise ValueError('account_recovery_current_matches_stage')
    source = sqlite3.connect(current_path.as_uri() + '?mode=ro', uri=True)
    target = sqlite3.connect(stage_path.as_uri() + '?mode=rw', uri=True)
    try:
        source.execute('BEGIN')
        source_tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'v2_source_evidence' not in source_tables:
            raise ValueError('account_recovery_current_tombstones_missing')
        tombstones = set()
        unavailable = {}
        current_identities = {}
        for evidence_id, raw in source.execute('SELECT evidence_id,definition FROM v2_source_evidence'):
            evidence = json.loads(raw)
            if (not isinstance(evidence, dict) or type(evidence.get('deleted')) is not bool
                    or evidence.get('state') not in {'available', 'expired', 'deleted', 'revoked'}):
                raise ValueError('account_recovery_invalid_evidence')
            if evidence.get('deleted') is True or evidence.get('state') in {'deleted', 'revoked', 'expired'}:
                key = tuple(evidence.get(k) for k in ('connection_id', 'resource', 'source_id'))
                if any(not isinstance(v, str) or not v for v in key):
                    raise ValueError('account_recovery_invalid_evidence')
                if not isinstance(evidence_id, str) or not evidence_id:
                    raise ValueError('account_recovery_invalid_evidence')
                current_identities[evidence_id] = key
            if evidence.get('deleted') is True or evidence.get('state') == 'deleted':
                tombstones.add(key)
            elif evidence['state'] in {'revoked', 'expired'}:
                unavailable[evidence_id] = (key, evidence['state'])
        target.execute('PRAGMA secure_delete=ON')
        target.execute('BEGIN IMMEDIATE')
        tables = {r[0] for r in target.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if ('v2_recovery_staging' not in tables or target.execute(
                "SELECT COUNT(*) FROM v2_recovery_staging WHERE state='isolated'").fetchone()[0] != 1):
            raise ValueError('account_recovery_not_staged')
        if 'v2_source_evidence' not in tables:
            raise ValueError('account_recovery_staged_evidence_missing')
        if ('v2_findings' in tables) != ('v2_finding_versions' in tables):
            raise ValueError('account_recovery_findings_incomplete')
        target.execute('''CREATE TABLE IF NOT EXISTS v2_recovery_source_tombstones (
            connection_id TEXT NOT NULL, resource TEXT NOT NULL, source_id TEXT NOT NULL,
            PRIMARY KEY(connection_id,resource,source_id)) STRICT''')
        target.executemany('INSERT OR IGNORE INTO v2_recovery_source_tombstones VALUES (?,?,?)', tombstones)
        # Reapply all prior tombstones as well, so retries are monotonic.
        tombstones = set(target.execute('SELECT connection_id,resource,source_id FROM v2_recovery_source_tombstones'))
        evidence_ids = set()
        staged_evidence_ids = set()
        staged_evidence = []
        for evidence_id, raw in target.execute('SELECT evidence_id,definition FROM v2_source_evidence').fetchall():
            if not isinstance(evidence_id, str) or not evidence_id:
                raise ValueError('account_recovery_invalid_evidence')
            staged_evidence_ids.add(evidence_id)
            evidence = json.loads(raw)
            if (not isinstance(evidence, dict) or type(evidence.get('deleted')) is not bool
                    or evidence.get('state') not in {'available', 'expired', 'deleted', 'revoked'}):
                raise ValueError('account_recovery_invalid_evidence')
            key = tuple(evidence.get(k) for k in ('connection_id', 'resource', 'source_id'))
            if any(not isinstance(v, str) or not v for v in key):
                raise ValueError('account_recovery_invalid_evidence')
            staged_evidence.append((evidence_id, evidence, key))
            if evidence['deleted'] or evidence['state'] == 'deleted':
                # An older staged copy may itself have a deletion that the
                # supplied snapshot lacks. Never weaken that privacy state.
                tombstones.add(key)
                target.execute('INSERT OR IGNORE INTO v2_recovery_source_tombstones VALUES (?,?,?)', key)
        for evidence_id, evidence, key in staged_evidence:
            current_state = unavailable.get(evidence_id)
            if evidence_id in current_identities and current_identities[evidence_id] != key:
                raise ValueError('account_recovery_evidence_identity_mismatch')
            if current_state and evidence['state'] == 'available':
                evidence['state'] = current_state[1]
            elif current_state and current_state[1] == 'revoked' and evidence['state'] == 'expired':
                evidence['state'] = 'revoked'
            if key in tombstones:
                evidence.update(content={}, state='deleted', deleted=True)
                target.execute('UPDATE v2_source_evidence SET definition=? WHERE evidence_id=?',
                               (json.dumps(evidence), evidence_id))
                evidence_ids.add(evidence_id)
            elif evidence['state'] == 'revoked':
                evidence['content'] = {}
                target.execute('UPDATE v2_source_evidence SET definition=? WHERE evidence_id=?',
                               (json.dumps(evidence), evidence_id))
                evidence_ids.add(evidence_id)
            elif evidence['state'] == 'expired':
                # Expired cache bytes are not retained, but expiry alone does
                # not invalidate independently retained finding history.
                evidence['content'] = {}
                target.execute('UPDATE v2_source_evidence SET definition=? WHERE evidence_id=?',
                               (json.dumps(evidence), evidence_id))
        redacted = 0
        if 'v2_findings' in tables:
            findings = {}
            affected = set()
            def references_deleted(value):
                if not isinstance(value, dict) or not isinstance(value.get('source_refs'), list):
                    raise ValueError('account_recovery_invalid_finding')
                refs = value['source_refs']
                if any(not isinstance(ref, dict) or not isinstance(ref.get('evidence_id'), str)
                       or not ref['evidence_id'] for ref in refs):
                    raise ValueError('account_recovery_invalid_finding')
                # A finding with an unresolved source cannot be reconciled
                # against current deletion markers. Keep the copy isolated.
                if any(ref['evidence_id'] not in staged_evidence_ids for ref in refs):
                    raise ValueError('account_recovery_finding_evidence_missing')
                return any(ref['evidence_id'] in evidence_ids for ref in refs)

            for finding_id, raw in target.execute('SELECT finding_id,definition FROM v2_findings').fetchall():
                finding = json.loads(raw)
                if references_deleted(finding):
                    affected.add(finding_id)
                findings[finding_id] = finding
            for finding_id, raw in target.execute('SELECT finding_id,definition FROM v2_finding_versions'):
                if finding_id not in findings:
                    raise ValueError('account_recovery_invalid_finding')
                if references_deleted(json.loads(raw)):
                    affected.add(finding_id)
            for finding_id in affected:
                finding = findings[finding_id]
                keys = ('finding_id', 'version', 'policy_id', 'policy_revision', 'project_id',
                        'executing_user_id', 'audience_user_ids')
                if any(key not in finding for key in keys) or finding['finding_id'] != finding_id:
                    raise ValueError('account_recovery_invalid_finding')
                tombstone = {key: finding[key] for key in keys}
                tombstone.update(state='redacted', source_refs=[])
                target.execute('UPDATE v2_findings SET definition=? WHERE finding_id=?',
                               (json.dumps(tombstone), finding_id))
                target.execute('DELETE FROM v2_finding_versions WHERE finding_id=?', (finding_id,))
                if 'v2_outbox' in tables:
                    target.execute("""UPDATE v2_outbox SET message='[Finding unavailable]',
                        status=CASE WHEN status='delivered' THEN status ELSE 'failed' END,
                        metadata_json=json_set(metadata_json, '$.finding_delivery_state', 'revoked'),
                        last_error='', lease_owner='', lease_expires_at=''
                        WHERE kind='finding' AND json_extract(metadata_json, '$.finding_id')=?""", (finding_id,))
                redacted += 1
        target.commit()
        return {'source_tombstones': len(tombstones), 'evidence_redacted': len(evidence_ids),
                'findings_redacted': redacted, 'outbox_inspected': 'v2_outbox' in tables}
    except Exception:
        target.rollback()
        raise
    finally:
        target.close()
        source.close()
