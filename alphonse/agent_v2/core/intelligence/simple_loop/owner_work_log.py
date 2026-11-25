"""Read-only, bounded and sanitized owner activity from claim-bound checkpoints."""
from __future__ import annotations

import json
import os
import re

from alphonse.agent_v2.core.messages.sqlite_queue import finalized_owner_history_matches
from .owner_failure_diagnostics import public_failure_code

_LABELS = {
    'native.search_memory': 'Searched conversation history',
    'native.read_memory_source': 'Read a conversation source',
    'native.web_search': 'Searched the web',
    'native.web_fetch': 'Read a web source',
    'native.bash': 'Local command',
    'native.owner_clock': 'Checked the clock',
    'native.owner_schedules': 'Checked schedules',
    'native.owner_gmail_unread': 'Checked unread mail',
    'native.skill_catalog': 'Checked available skills',
    'native.owner_named_skill': 'Checked a named skill',
}


def _events(actor: str, refs: list, observations: dict, pending: object) -> list[dict]:
    events = []
    for ref in refs:
        item = observations[ref]
        if not isinstance(item, dict) or item.get('actor') != actor:
            raise ValueError('owner_work_log_unavailable')
        tool = item.get('tool_id')
        if tool not in _LABELS:
            events.append({'label': 'Used an available tool', 'outcome': 'Observation recorded'})
            continue
        outcome = 'Observation recorded'
        if tool == 'native.bash':
            shell = item.get('host_bash_observation')
            if isinstance(shell, dict) and shell.get('timed_out') is True:
                outcome = 'Command timed out; effects are unconfirmed'
            elif isinstance(shell, dict) and shell.get('cancelled') is True:
                outcome = 'Command was interrupted; effects are unconfirmed'
            elif isinstance(shell, dict) and type(shell.get('exit_code')) is int and -255 <= shell['exit_code'] <= 255:
                outcome = f"Command returned exit code {shell['exit_code']}; effects are unconfirmed"
            else:
                outcome = 'Command returned; effects are unconfirmed'
        elif tool in {'native.web_search', 'native.web_fetch'}:
            web = item.get('web_observation')
            if isinstance(web, dict) and web.get('status') == 'failed':
                outcome = 'The source could not be read'
            elif tool == 'native.web_search' and isinstance(web, dict) and web.get('result_count') == 0:
                outcome = 'No results returned'
        events.append({'label': _LABELS[tool], 'outcome': outcome})
    if isinstance(pending, dict) and pending.get('kind') == 'tool':
        events.append({'label': 'Tool action', 'outcome': 'Reserved; execution or effects are unconfirmed'})
    return events


def owner_work_log(queue, message_id: str) -> dict:
    """Return safe facts only after matching owner ingress and durable history.

    This is a projection of retained checkpoint data, not a complete audit log.
    Unknown, in-flight and shell effects are never represented as verified.
    """
    if not isinstance(message_id, str) or not re.fullmatch(r'[0-9a-f]{32}', message_id):
        raise ValueError('owner_work_log_unavailable')
    with queue._connect() as conn:
        row = conn.execute('''SELECT c.task_state_json, c.owner_id FROM v2_task_checkpoints c
            JOIN v2_trusted_owner_ingress o ON o.message_id=c.task_id AND o.owner_id=c.owner_id
            JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=c.owner_id
            JOIN v2_users u ON u.user_id=c.owner_id AND u.role IN ('admin','owner') AND u.is_active=1
            WHERE c.task_id=? AND o.peer_uid=? AND o.history_version=1''',
            (message_id, os.getuid())).fetchone()
        if row is None:
            raise ValueError('owner_work_log_unavailable')
        try:
            raw = json.loads(row['task_state_json'])
            actor = row['owner_id']
            if (raw.get('task_id') != message_id or raw.get('message_id') != message_id
                    or raw.get('user') != actor or not finalized_owner_history_matches(conn, message_id, actor, raw)):
                raise ValueError('owner_work_log_unavailable')
            metadata = raw['metadata']
            state = metadata['single_loop_checkpoint']['state']
            observations = metadata['owner_read_observations']
            refs = state['evidence_sources']
            if (state.get('task_id') != message_id or state.get('actor') != actor
                    or not isinstance(refs, list) or not isinstance(observations, dict)
                    or len(refs) > 100 or len(set(refs)) != len(refs)
                    or set(refs) != set(observations)):
                raise ValueError('owner_work_log_unavailable')
            events = _events(actor, refs, observations, metadata.get('owner_read_pending_action'))
            status = state.get('status')
            if status not in {'running', 'blocked', 'completed', 'waiting_user', 'cancelled'}:
                raise ValueError('owner_work_log_unavailable')
            failure = public_failure_code(state.get('blocker')) if status == 'blocked' else None
            return {'status': status, 'failure_code': failure, 'events': events,
                    'notice': 'This log shows recorded tool observations; it may not include every activity.'}
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ValueError('owner_work_log_unavailable') from None
