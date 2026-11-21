"""Claim-bound cancellation of a previously observed owner schedule."""
from __future__ import annotations

from .owner_schedule_origin import verified_schedule_origin_locked


def cancel_schedule_locked(conn, task, arguments, snapshot):
    """Cancel the exact owner schedule from a complete, current observation.

    The caller must hold the owner's fenced transaction and cancellation lock.
    The model interprets the request; the host checks identity, provenance,
    exact target and whether execution might already have started.
    """
    if (not conn.in_transaction or not isinstance(arguments, dict)
            or set(arguments) != {'operation', 'scheduled_task_id', 'request_quote'}
            or arguments['operation'] != 'cancel' or arguments['request_quote'] != task.goal
            or not isinstance(arguments['scheduled_task_id'], str)
            or not isinstance(snapshot, dict) or snapshot.get('status') != 'complete'):
        raise ValueError('owner_schedule_cancel_unavailable')
    schedule_id = arguments['scheduled_task_id']
    matches = [item for item in snapshot.get('schedules', [])
               if isinstance(item, dict) and item.get('id') == schedule_id]
    row = verified_schedule_origin_locked(conn, schedule_id)
    if (len(matches) != 1 or row is None or row['owner_user_id'] != task.user
            or row['status'] not in {'active', 'paused'}
            or matches[0].get('execution') != 'owner_worker'
            or matches[0].get('status') != row['status']
            or matches[0].get('prompt') != row['prompt']
            or matches[0].get('name') != row['name']
            or matches[0].get('next_run_at') != row['next_run_at']):
        raise ValueError('owner_schedule_cancel_unavailable')
    occurrence = conn.execute('''SELECT x.message_id,q.status AS queue_status
        FROM v2_owner_schedule_occurrences x LEFT JOIN v2_inbound_messages q ON q.message_id=x.message_id
        WHERE x.schedule_id=? AND x.owner_id=?''', (schedule_id, task.user)).fetchone()
    running = occurrence is not None and occurrence['queue_status'] != 'pending'
    changed = conn.execute("""UPDATE v2_scheduled_tasks SET status='cancelled', next_run_at=NULL
        WHERE scheduled_task_id=? AND owner_user_id=? AND status IN ('active','paused')""", (schedule_id, task.user))
    if changed.rowcount != 1:
        raise ValueError('owner_schedule_cancel_unavailable')
    if occurrence is not None:
        conn.execute("UPDATE v2_owner_schedule_occurrences SET state='cancelled' WHERE schedule_id=? AND state='queued'", (schedule_id,))
        if not running:
            conn.execute("DELETE FROM v2_inbound_messages WHERE message_id=? AND status='pending'", (occurrence['message_id'],))
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_schedule_cancellations (
        request_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, schedule_id TEXT NOT NULL,
        execution_may_have_started INTEGER NOT NULL CHECK(execution_may_have_started IN (0,1)),
        observed_schedule_sha256 TEXT NOT NULL, source_observation_sha256 TEXT NOT NULL) STRICT''')
    from .owner_atomic_checkpoint import observation_digest
    conn.execute('''INSERT INTO v2_owner_schedule_cancellations
        (request_id,owner_id,schedule_id,execution_may_have_started,observed_schedule_sha256,source_observation_sha256)
        VALUES (?,?,?,?,?,?)''', (task.message_id, task.user, schedule_id, int(running), observation_digest(matches[0]), observation_digest(snapshot)))
    return {'id': schedule_id, 'status': 'cancelled', 'execution_may_have_started': running,
            'observed_schedule': matches[0], 'source_observation_sha256': observation_digest(snapshot)}


def valid_cancel_locked(conn, actor, arguments, observation, *, task):
    try:
        from .owner_atomic_checkpoint import observation_digest
        if (set(arguments) != {'operation', 'scheduled_task_id', 'request_quote'}
                or arguments['operation'] != 'cancel' or arguments['request_quote'] != task.goal
                or set(observation) != {'id', 'status', 'execution_may_have_started', 'observed_schedule', 'source_observation_sha256'}
                or observation['id'] != arguments['scheduled_task_id'] or observation['status'] != 'cancelled'
                or type(observation['execution_may_have_started']) is not bool):
            return False
        receipt = conn.execute('''SELECT * FROM v2_owner_schedule_cancellations
            WHERE request_id=? AND owner_id=? AND schedule_id=?''',
            (task.message_id, actor, observation['id'])).fetchone()
        row = verified_schedule_origin_locked(conn, observation['id'])
        return bool(receipt and row and row['owner_user_id'] == actor and row['status'] == 'cancelled'
                    and receipt['execution_may_have_started'] == observation['execution_may_have_started']
                    and receipt['observed_schedule_sha256'] == observation_digest(observation['observed_schedule'])
                    and receipt['source_observation_sha256'] == observation['source_observation_sha256'])
    except Exception:
        return False


def valid_cancel_observation(actor, arguments, observation, *, context):
    try:
        if context.task.user != actor or context.cancellation_checker() is not False:
            return False
        with context.messages._connect() as conn:
            conn.execute('BEGIN')
            return valid_cancel_locked(conn, actor, arguments, observation, task=context.task)
    except Exception:
        return False


def committed_cancel_for_recovery(queue, message_id, owner):
    """Expose a verified committed effect without replaying it or its response."""
    import json
    import os
    from types import SimpleNamespace
    from .owner_atomic_checkpoint import observation_digest
    try:
        with queue._connect() as conn:
            conn.execute('BEGIN')
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_owner_schedule_cancellations'").fetchone()
            if not exists:
                return None
            row = conn.execute('''SELECT q.prompt,c.task_state_json FROM v2_trusted_owner_ingress o
                JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=o.owner_id
                JOIN v2_users u ON u.user_id=o.owner_id AND u.role IN ('admin','owner') AND u.is_active=1
                JOIN v2_task_checkpoints c ON c.task_id=o.message_id AND c.owner_id=o.owner_id
                LEFT JOIN v2_inbound_messages q ON q.message_id=o.message_id AND q.user_id=o.owner_id
                WHERE o.message_id=? AND o.owner_id=? AND o.peer_uid=?''', (message_id, owner, os.getuid())).fetchone()
            if row is None:
                return None
            raw = json.loads(row['task_state_json'])
            if raw.get('task_id') != message_id or raw.get('user') != owner:
                return None
            if row['prompt'] is None:
                from alphonse.agent_v2.core.messages.sqlite_queue import finalized_owner_history_matches
                if not finalized_owner_history_matches(conn, message_id, owner, raw):
                    return None
            goal = row['prompt'] if row['prompt'] is not None else raw.get('goal')
            bindings = raw['metadata']['owner_read_observations']
            for binding in bindings.values():
                args = binding.get('arguments', {})
                effect = binding.get('schedule_observation')
                if (binding.get('actor') == owner and binding.get('tool_id') == 'native.scheduled_task'
                        and args.get('operation') == 'cancel' and isinstance(effect, dict)
                        and binding.get('observation_sha256') == observation_digest(effect)
                        and valid_cancel_locked(conn, owner, args, effect,
                                                task=SimpleNamespace(goal=goal, message_id=message_id))):
                    return {'kind': 'scheduled_task_cancelled', 'schedule_id': effect['id'],
                            'name': effect['observed_schedule']['name'],
                            'execution_may_have_started': effect['execution_may_have_started']}
    except Exception:
        return None
    return None
