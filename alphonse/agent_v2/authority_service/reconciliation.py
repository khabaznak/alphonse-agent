"""Read-only comparison of an authority request and owner queue observation.

This does not resume, enqueue, cancel, claim, dispatch, or verify an effect.
The service must authenticate the Desktop and protect its ledger; local queue
rows are same-user-writable observations, not a source of authority.
"""
from __future__ import annotations

from pathlib import Path
import re
import sqlite3

from alphonse.agent_v2.core.messages.sqlite_queue import _owner_ingress_matches


def _local_observation(path, *, actor, message_id, request_id, goal):
    """Read one consistent local snapshot without initializing or updating it."""
    try:
        if not isinstance(path, (str, Path)) or str(path) == ':memory:':
            return 'unavailable'
        location = Path(path).expanduser().resolve(strict=True)
        with sqlite3.connect(location.as_uri() + '?mode=ro', uri=True, timeout=3) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA query_only=ON')
            conn.execute('BEGIN')
            required = {'v2_inbound_messages', 'v2_trusted_owner_ingress', 'v2_authority_ingress_bindings'}
            found = {row['name'] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not required.issubset(found):
                return 'unavailable'
            row = conn.execute('''SELECT q.message_id,q.timestamp,q.prompt,q.user_id,q.project_id,q.tag,
                q.correlation_id,q.memory_session_id,q.metadata_json,q.status,
                o.owner_id,o.payload_sha256,o.history_version,o.history_event_id,b.request_id
                FROM v2_inbound_messages q
                LEFT JOIN v2_trusted_owner_ingress o ON o.message_id=q.message_id
                LEFT JOIN v2_authority_ingress_bindings b ON b.message_id=q.message_id
                WHERE q.message_id=?''', (message_id,)).fetchone()
            if row is None:
                # The queue may already have removed/finalized the row. Absence
                # alone cannot prove that the request was never enqueued.
                return 'absent'
            payload = [row[key] for key in ('message_id', 'timestamp', 'prompt', 'user_id', 'project_id',
                       'tag', 'correlation_id', 'memory_session_id', 'metadata_json')]
            if (row['user_id'] != actor or row['owner_id'] != actor or row['prompt'] != goal
                    or row['request_id'] != request_id or not _owner_ingress_matches(conn, row, payload)):
                return 'mismatch'
            status = row['status']
            return status if status in {'pending', 'processing', 'retry_wait', 'failed'} else 'unavailable'
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
        return 'unavailable'


def inspect_owner_request(*, desktop_client, queue_db, actor, message_id, request_id, goal):
    """Describe matching protected and local observations; never mutate either.

    The supplied Desktop client must authenticate to the protected service;
    this function cannot establish that its transport or actor is trustworthy.
    Every state requires an external, authorized recovery decision. A matching
    local row is not a claim or permission to execute.
    """
    if (not isinstance(actor, str) or not actor.strip() or not isinstance(goal, str) or not goal.strip()
            or not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None
            or (request_id is not None and (not isinstance(request_id, str)
                                             or re.fullmatch(r'[0-9a-f]{32}', request_id) is None))
            or not callable(getattr(desktop_client, 'lookup_request' if request_id is None else 'inspect_request', None))):
        raise ValueError('owner_authority_reconciliation_invalid')
    resolving = request_id is None
    try:
        if resolving:
            protected = desktop_client.lookup_request(message_id=message_id, goal=goal)
            if (isinstance(protected, dict) and set(protected) == {'actor', 'found'}
                    and protected['actor'] == actor and protected['found'] is False):
                return {'authority': 'not_found', 'queue': 'not_checked', 'started_actions': None,
                        'recovery': 'manual_review', 'may_auto_replay': False}
            if (not isinstance(protected, dict)
                    or set(protected) != {'request_id', 'actor', 'found', 'status', 'started_actions'}
                    or protected['found'] is not True
                    or not isinstance(protected['request_id'], str)
                    or re.fullmatch(r'[0-9a-f]{32}', protected['request_id']) is None):
                raise ValueError()
            request_id = protected['request_id']
            protected = {key: protected[key] for key in ('actor', 'status', 'started_actions')}
        else:
            protected = desktop_client.inspect_request(request_id=request_id, message_id=message_id, goal=goal)
        if (not isinstance(protected, dict) or set(protected) != {'actor', 'status', 'started_actions'}
                or protected['actor'] != actor or protected['status'] not in {'pending', 'claimed', 'cancelled', 'finished'}
                or type(protected['started_actions']) is not int or protected['started_actions'] < 0):
            raise ValueError()
    except Exception:
        return {'authority': 'unavailable_or_mismatch', 'queue': 'not_checked', 'started_actions': None,
                'recovery': 'manual_review', 'may_auto_replay': False}
    local = _local_observation(queue_db, actor=actor, message_id=message_id, request_id=request_id, goal=goal)
    remote = protected['status']
    # A claimed request or any started action may already have effects. A lost
    # reply, expired lease, queue row or process return cannot disprove them.
    recovery = ('execution_uncertain' if remote in {'claimed', 'finished'} or protected['started_actions'] > 0
                else 'cancelled' if remote == 'cancelled'
                else 'local_observation_mismatch' if local == 'mismatch'
                else 'local_state_unavailable' if local == 'unavailable'
                else 'local_row_absent' if local == 'absent'
                else 'local_state_requires_review' if local != 'pending'
                else 'awaiting_authorized_claim')
    result = {'authority': remote, 'queue': local, 'started_actions': protected['started_actions'],
              'recovery': recovery, 'may_auto_replay': False}
    if resolving:
        result['resolved_request_id'] = request_id
    return result
