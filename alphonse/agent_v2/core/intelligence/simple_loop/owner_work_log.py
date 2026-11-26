"""Read-only, bounded and sanitized owner activity from claim-bound checkpoints."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone

from alphonse.agent_v2.accounts import exact_gmail_label_id, valid_gmail_label_name
from alphonse.agent_v2.core.inference.models import RESPONSE_PARSE_STAGES
from alphonse.agent_v2.core.messages.sqlite_queue import finalized_owner_history_matches
from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
from .owner_atomic_checkpoint import observation_digest, _decision_digest
from .owner_failure_diagnostics import public_failure_code
from .owner_gmail import OWNER_GMAIL_TOOL_ID, owner_gmail_failure_message
from .owner_outlook import OWNER_OUTLOOK_TOOL_ID, owner_outlook_failure_message
from .owner_gmail_search import OWNER_GMAIL_SEARCH_TOOL_ID
from .owner_web import is_web_failure_observation
from .candidate_decision_policy import ACTIVITY_INTENTS

_LABELS = {
    'native.search_memory': 'Searched conversation history',
    'native.read_memory_source': 'Read a conversation source',
    'native.web_search': 'Searched the web',
    'native.web_fetch': 'Read a web source',
    'native.bash': 'Local command',
    'native.owner_clock': 'Checked the clock',
    'native.owner_schedules': 'Checked schedules',
    'native.owner_gmail_unread': 'Checked unread mail',
    OWNER_OUTLOOK_TOOL_ID: 'Checked Outlook Inbox',
    OWNER_GMAIL_SEARCH_TOOL_ID: 'Searched Gmail',
    'native.owner_calendar_agenda': 'Checked Google Calendar',
    'native.skill_catalog': 'Checked available skills',
    'native.owner_named_skill': 'Checked a named skill',
}

# These describe the recorded tool, not the assistant's purpose or its effect.
_TOOL_KINDS = {
    'native.search_memory': 'conversation_search',
    'native.read_memory_source': 'conversation_source',
    'native.web_search': 'web_search',
    'native.web_fetch': 'web_source',
    'native.bash': 'local_command',
    'native.owner_clock': 'clock',
    'native.owner_schedules': 'schedules',
    'native.owner_gmail_unread': 'mail',
    OWNER_OUTLOOK_TOOL_ID: 'mail',
    OWNER_GMAIL_SEARCH_TOOL_ID: 'mail_search',
    'native.owner_calendar_agenda': 'calendar',
    'native.skill_catalog': 'skills',
    'native.owner_named_skill': 'named_skill',
}

_WEB_FAILURE_DETAILS = {
    'searxng_timeout': 'timeout',
    'searxng_http_error': 'http_error',
    'searxng_invalid_json': 'invalid_response',
    'searxng_upstream_unavailable': 'upstream_search_failed',
    'web_fetch_timeout': 'timeout',
    'web_fetch_http_error': 'http_error',
    'web_fetch_unresolvable_host': 'unresolvable_host',
    'web_fetch_output_too_large': 'output_too_large',
    'web_fetch_output_invalid': 'invalid_response',
}


def _pending_intent(actor: str, task_id: str, state: dict, pending: object, marker: object) -> str | None:
    """Return only a bounded, action-bound model intention, never raw text."""
    keys = {'schema_version', 'actor', 'task_id', 'reservation_id', 'tool_id',
            'claim_digest', 'decision_sha256', 'projection_sha256', 'intent', 'usage', 'selection'}
    if (not isinstance(marker, dict) or set(marker) != keys
            or type(marker['schema_version']) is not int or marker['schema_version'] != 1
            or not isinstance(pending, dict) or pending.get('kind') != 'tool'
            or state.get('status') != 'running' or not isinstance(state.get('inflight'), str)
            or not state['inflight'] or state['inflight'] != pending.get('reservation_id')
            or marker['reservation_id'] != pending.get('reservation_id')
            or marker['tool_id'] != pending.get('tool_id')
            or not isinstance(pending.get('arguments'), dict)
            or marker['claim_digest'] != pending.get('claim_digest')
            or marker['actor'] != actor or marker['task_id'] != task_id
            or not isinstance(marker['intent'], str) or marker['intent'] not in ACTIVITY_INTENTS
            or any(not isinstance(marker[key], str) or re.fullmatch(r'[0-9a-f]{64}', marker[key]) is None
                   for key in ('claim_digest', 'decision_sha256', 'projection_sha256'))):
        return None
    selection = marker['selection']
    if (not isinstance(selection, dict)
            or set(selection) != {'reservation_id', 'claim_digest', 'decision_sha256', 'projection_sha256'}
            or not isinstance(selection['reservation_id'], str) or not selection['reservation_id']
            or any(selection[key] != marker[key]
                   for key in ('claim_digest', 'decision_sha256', 'projection_sha256'))):
        return None
    usage = marker['usage']
    if (usage is not None and (not isinstance(usage, dict)
            or set(usage) != {'input_tokens', 'output_tokens'}
            or any(value is not None and (type(value) is not int or value < 0) for value in usage.values()))):
        return None
    decision = {'kind': 'tool', 'tool_id': pending['tool_id'], 'arguments': pending['arguments'],
                'text': None, 'activity_intent': marker['intent']}
    if usage is not None:
        decision['usage'] = usage
    try:
        if _decision_digest(decision) != marker['decision_sha256']:
            return None
    except (TypeError, ValueError):
        return None
    return marker['intent']


def _live_claim(conn, task_id: str, actor: str, marker: object) -> bool:
    """An old checkpoint marked running does not prove a live owner claim."""
    try:
        row = conn.execute('''SELECT user_id,status,lease_owner,claim_generation,lease_expires_at
                              FROM v2_inbound_messages WHERE message_id=?''', (task_id,)).fetchone()
        if row is None or row['user_id'] != actor or row['status'] != 'processing':
            return False
        expiry = datetime.fromisoformat(row['lease_expires_at'])
        if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
            return False
        claim = ClaimReceipt(task_id, actor, row['lease_owner'], row['claim_generation'])
        return isinstance(marker, dict) and marker.get('claim_digest') == _claim_digest(claim)
    except (sqlite3.Error, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


def _gmail_scope(item: dict) -> dict[str, str]:
    """Use only the recorded provider scope for an observed search."""
    mail = item.get('gmail_search_observation')
    if (not isinstance(mail, dict) or mail.get('status') not in ('complete', 'partial')
            or item.get('observation_sha256') != observation_digest(mail)):
        return {}
    authority = mail.get('authority')
    if not isinstance(authority, dict) or not authority.get('grant_id'):
        return {}
    resource = authority.get('resource', 'INBOX')
    name = authority.get('label_name')
    if resource == 'INBOX' and name is None:
        return {'gmail_label': 'Inbox'}
    if resource != 'INBOX' and exact_gmail_label_id(resource) and valid_gmail_label_name(name):
        return {'gmail_label': name}
    return {}


def _activity(actor: str, refs: list, observations: dict, pending: object, *,
              task_id: str = '', state: dict | None = None, intent_marker: object = None) -> dict:
    """Summarize retained observations without interpreting arguments or output.

    A reservation can survive interruption and is not evidence of execution.
    A process exit code is not evidence that any intended effect occurred.
    """
    latest = None
    unconfirmed_effect_count = 0
    outcome_counts = {}
    for ref in refs:
        item = observations[ref]
        if not isinstance(item, dict) or item.get('actor') != actor:
            raise ValueError('owner_work_log_unavailable')
        tool = item.get('tool_id')
        kind = _TOOL_KINDS.get(tool, 'other') if isinstance(tool, str) else 'other'
        outcome, effects = 'recorded', 'not_reported'
        if tool == 'native.bash':
            unconfirmed_effect_count += 1
            effects, outcome = 'unconfirmed', 'unknown'
            shell = item.get('host_bash_observation')
            if isinstance(shell, dict):
                if shell.get('timed_out') is True:
                    outcome = 'timed_out'
                elif shell.get('cancelled') is True:
                    outcome = 'interrupted'
                elif type(shell.get('exit_code')) is int and -255 <= shell['exit_code'] <= 255:
                    outcome = 'exit_zero' if shell['exit_code'] == 0 else 'exit_nonzero'
        elif tool in ('native.web_search', 'native.web_fetch'):
            web = item.get('web_observation')
            if isinstance(web, dict) and web.get('status') == 'failed':
                outcome = 'source_failed'
            elif tool == 'native.web_search' and isinstance(web, dict) and web.get('partial') is True:
                outcome = 'partial'
            elif (tool == 'native.web_search' and isinstance(web, dict)
                  and type(web.get('result_count')) is int and web['result_count'] == 0 and not web.get('answers')):
                outcome = 'no_results'
        elif tool == OWNER_GMAIL_TOOL_ID and owner_gmail_failure_message(item.get('gmail_observation')):
            outcome = 'source_failed'
        elif tool == OWNER_OUTLOOK_TOOL_ID and owner_outlook_failure_message(item.get('outlook_observation')):
            outcome = 'source_failed'
        elif tool == OWNER_GMAIL_SEARCH_TOOL_ID:
            mail = item.get('gmail_search_observation')
            if isinstance(mail, dict) and mail.get('status') == 'unavailable':
                outcome = 'source_failed'
        latest = {'tool_kind': kind, 'outcome': outcome, 'effects': effects}
        if tool == OWNER_GMAIL_SEARCH_TOOL_ID:
            latest.update(_gmail_scope(item))
        key = (kind, outcome)
        outcome_counts[key] = outcome_counts.get(key, 0) + 1
        # Only expose a bounded reason explicitly reported by a native web
        # tool and bound to the recorded bytes. Never include remote text.
        if tool in ('native.web_search', 'native.web_fetch'):
            web = item.get('web_observation')
            if (is_web_failure_observation(tool, web)
                    and item.get('observation_sha256') == observation_digest(web)):
                latest['detail'] = _WEB_FAILURE_DETAILS[web['error_code']]
    return {'recorded_action_count': len(refs),
            'unconfirmed_effect_count': unconfirmed_effect_count,
            'pending_tool': isinstance(pending, dict) and pending.get('kind') == 'tool',
            'pending_intent': _pending_intent(actor, task_id, state or {}, pending, intent_marker),
            'outcome_counts': [{'tool_kind': kind, 'outcome': outcome, 'count': count}
                               for (kind, outcome), count in outcome_counts.items()],
            'latest': latest}


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
                outcome = ('Upstream search engines failed'
                           if is_web_failure_observation(tool, web) and web['error_code'] == 'searxng_upstream_unavailable'
                           else 'The source could not be read')
            elif tool == 'native.web_search' and isinstance(web, dict) and web.get('partial') is True:
                outcome = 'Partial results returned because some search engines failed'
            elif (tool == 'native.web_search' and isinstance(web, dict)
                  and web.get('result_count') == 0 and not web.get('answers')):
                outcome = 'No results returned'
        elif tool == OWNER_GMAIL_TOOL_ID and owner_gmail_failure_message(item.get('gmail_observation')):
            events.append({'label': 'Attempted an unread mail check', 'outcome': 'The Inbox count could not be verified'})
            continue
        elif tool == OWNER_OUTLOOK_TOOL_ID and owner_outlook_failure_message(item.get('outlook_observation')):
            events.append({'label': 'Attempted an Outlook Inbox check', 'outcome': 'The unread count could not be verified'})
            continue
        event = {'label': _LABELS[tool], 'outcome': outcome}
        if tool == OWNER_GMAIL_SEARCH_TOOL_ID:
            event.update(_gmail_scope(item))
        events.append(event)
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
            checkpoint = metadata['single_loop_checkpoint']
            state = checkpoint['state']
            observations = metadata['owner_read_observations']
            refs = state['evidence_sources']
            if (state.get('task_id') != message_id or state.get('actor') != actor
                    or not isinstance(refs, list) or not isinstance(observations, dict)
                    or len(refs) > 100 or len(set(refs)) != len(refs)
                    or set(refs) != set(observations)):
                raise ValueError('owner_work_log_unavailable')
            events = _events(actor, refs, observations, metadata.get('owner_read_pending_action'))
            activity = _activity(actor, refs, observations, metadata.get('owner_read_pending_action'),
                                 task_id=message_id, state=state,
                                 intent_marker=(metadata.get('owner_activity_intent')
                                                if _live_claim(conn, message_id, actor, metadata.get('owner_activity_intent'))
                                                else None))
            status = state.get('status')
            if status not in {'running', 'blocked', 'completed', 'waiting_user', 'cancelled'}:
                raise ValueError('owner_work_log_unavailable')
            failure = public_failure_code(state.get('blocker')) if status == 'blocked' else None
            reservations = checkpoint.get('budget', {}).get('reservations', {})
            parse_stages = sorted({stage for row in reservations.values()
                                   if isinstance(row, dict) and row.get('kind') == 'model'
                                   and row.get('status') == 'settled' and row.get('outcome') == 'failure'
                                   and type(stage := row.get('parse_stage')) is str and stage in RESPONSE_PARSE_STAGES})
            return {'status': status, 'failure_code': failure, 'events': events, 'activity': activity,
                    **({'model_parse_stages': parse_stages} if parse_stages else {}),
                    'notice': 'This log shows recorded tool observations; it may not include every activity.'}
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ValueError('owner_work_log_unavailable') from None
