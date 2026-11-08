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
        rows = connection.execute('SELECT integration_id, connection_json FROM v2_integrations').fetchall()
        managed = set()
        for integration_id, raw in rows:
            value = json.loads(raw or '{}')
            if not isinstance(value, dict):
                raise ValueError('account_recovery_invalid_connection')
            if not value:
                continue
            if (type(value.get('revision')) is not int or value['revision'] < 1
                    or not isinstance(value.get('owner_user_id'), str) or not value['owner_user_id']):
                raise ValueError('account_recovery_invalid_connection')
            managed.add(integration_id)
            value.update(status='reauth_required', health_code='restore_reconciliation_required',
                         revision=value['revision'] + 1, updated_at=timestamp)
            connection.execute('UPDATE v2_integrations SET enabled=0, connection_json=?, updated_at=? WHERE integration_id=?',
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
