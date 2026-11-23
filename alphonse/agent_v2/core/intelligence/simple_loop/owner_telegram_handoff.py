"""Validate a selected owner-loop draft for Telegram one-shot delivery.

Requires a checkpoint whose transaction and recovery paths support genuine
Telegram claims. This module does not start a listener or authorize tools.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4

from alphonse.agent_v2.core.messages.claim_fence import _read_claim, _time
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
from .owner_atomic_checkpoint import OwnerAtomicCheckpoint, _decision_digest
from .owner_authority_bridge import context_authority_active
from .owner_clock import display_response
from .owner_read_recovery import restore_owner_read_observations
from .owner_telegram_delivery import TelegramReplyUnavailable, dispatch_once, stage_locked
from .owner_telegram_ingress import verified_telegram_origin_locked, _allowed, _digest


def _reply_locked(conn, checkpoint, context):
    """Recheck origin, current claim, exact selected draft and sensitive sources."""
    if (type(checkpoint) is not OwnerAtomicCheckpoint or context.messages is not checkpoint.queue
            or context.claim_receipt is not checkpoint.receipt or not context_authority_active(context)
            or not callable(context.cancellation_checker) or context.cancellation_checker() is not False):
        raise TelegramReplyUnavailable('telegram_handoff_unavailable')
    receipt, task = checkpoint.receipt, checkpoint.task
    _read_claim(conn, receipt, _time(checkpoint.clock()))
    origin = verified_telegram_origin_locked(conn, receipt.message_id)
    queued = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (receipt.message_id,)).fetchone()
    if (origin is None or queued is None or origin['owner_id'] != receipt.user_id
            or task.task_id != receipt.message_id or task.message_id != receipt.message_id
            or task.user != receipt.user_id or task.goal != queued['prompt']
            or task.intelligence_engine != 'single_loop_experimental'
            or type(task.intelligence_schema_version) is not int or task.intelligence_schema_version != 1
            or any(getattr(task, key, None) != queued[key] for key in ('project_id', 'tag', 'correlation_id', 'memory_session_id'))
            or json.dumps(task.metadata, sort_keys=True) != queued['metadata_json']):
        raise TelegramReplyUnavailable('telegram_handoff_unavailable')
    _, raw, state, budget = checkpoint._load(conn, checkpoint.version)
    md, observations = raw['metadata'], raw['metadata'].get('owner_read_observations')
    selected = md.get('owner_read_selected_draft')
    ready = md.get('owner_read_ready_handoff')
    if (state.status not in {'completed', 'waiting_user'} or not state.response.strip() or state.inflight
            or budget.usage()['in_flight'] or budget.usage()['unknown_usage']
            or 'owner_read_pending_decision' in md or 'owner_read_pending_action' in md
            or 'owner_read_desktop_pickup_started' in md or not isinstance(selected, dict)
            or selected.get('schema_version') != 1 or selected.get('claim_digest') != _claim_digest(receipt)
            or selected.get('kind') != ('answer' if state.status == 'completed' else 'question')
            or selected.get('answer_sha256') != hashlib.sha256(state.response.encode('utf-8')).hexdigest()
            or selected.get('evidence_refs') != state.evidence_sources or not isinstance(observations, dict)
            or set(observations) != set(state.evidence_sources)
            or selected.get('evidence_bindings_sha256') != _decision_digest(observations)
            or ready != {'schema_version': 1, 'claim_digest': _claim_digest(receipt),
                         'selected_sha256': _decision_digest(selected)}):
        raise TelegramReplyUnavailable('telegram_handoff_unavailable')
    from .owner_gmail import OWNER_GMAIL_TOOL_ID, valid_owner_gmail_locked
    from .owner_reminder_create import OWNER_REMINDER_CREATE_TOOL_ID, valid_reminder_locked
    from .owner_schedule_create import SCHEDULED_TASK_TOOL_ID, valid_schedule_locked
    from .owner_schedule_cancel import valid_cancel_locked
    from .owner_knowledge_snapshot import OWNER_KNOWLEDGE_TOOL_ID, matches_owner_knowledge_locked
    for binding in observations.values():
        if not isinstance(binding, dict):
            raise TelegramReplyUnavailable('telegram_handoff_unavailable')
        tool, args = binding.get('tool_id'), binding.get('arguments')
        if (tool == OWNER_KNOWLEDGE_TOOL_ID and (args != {} or not matches_owner_knowledge_locked(
                conn, task, binding.get('knowledge_snapshot_digest'), cancellation_checker=context.cancellation_checker, clock=checkpoint.clock))
                or tool == OWNER_GMAIL_TOOL_ID and (state.status != 'completed' or not valid_owner_gmail_locked(
                    conn, task.user, binding.get('gmail_observation'), context=context))
                or tool == OWNER_REMINDER_CREATE_TOOL_ID and not valid_reminder_locked(
                    conn, task.user, args, binding.get('reminder_observation'), task=task)
                or tool == SCHEDULED_TASK_TOOL_ID and not (valid_cancel_locked if isinstance(args, dict)
                    and args.get('operation') == 'cancel' else valid_schedule_locked)(
                    conn, task.user, args, binding.get('schedule_observation'), task=task)):
            raise TelegramReplyUnavailable('telegram_handoff_source_unavailable')
    return {key: origin[key] for key in ('message_id', 'owner_id', 'integration_id', 'provider_user_id', 'chat_id', 'provider_message_id')} | {
        'text': display_response(state.response, observations)}


def stage_telegram_reply(checkpoint, context):
    """Stage a selected draft under the checkpoint's fenced claim transaction."""
    restore_owner_read_observations(checkpoint, context, expected_version=checkpoint.version, allow_completed=True)
    with checkpoint._transaction() as conn:
        reply = _reply_locked(conn, checkpoint, context)
        stage_locked(conn, reply, verify=lambda c, candidate: _reply_locked(c, checkpoint, context) == candidate)
    return reply['message_id']


def dispatch_telegram_reply(checkpoint, context, *, send):
    """Reauthorize sources and destination; dispatch at most once."""
    restore_owner_read_observations(checkpoint, context, expected_version=checkpoint.version, allow_completed=True)
    return dispatch_once(checkpoint.queue.db_path, checkpoint.receipt.message_id,
        verify=lambda conn, candidate: _reply_locked(conn, checkpoint, context) == candidate, send=send)


def finalize_telegram_reply(checkpoint, context):
    """Record provider acceptance and history, then consume the fenced claim.

    Never finalize uncertain sends or send again. An accepted send whose claim
    expired, whose source was revoked, or whose finalization crashed needs
    explicit reconciliation; this function does not infer successful delivery.
    """
    if verify_finalized_telegram_reply(checkpoint.queue, checkpoint.task.message_id):
        return
    restore_owner_read_observations(checkpoint, context, expected_version=checkpoint.version, allow_completed=True)
    committed = None
    def finish_claim(conn):
        if committed is None:
            raise TelegramReplyUnavailable('telegram_finalize_unavailable')
        row = conn.execute('SELECT status FROM v2_owner_telegram_replies WHERE message_id=?',
                           (checkpoint.receipt.message_id,)).fetchone()
        if row is None or row[0] != 'finalized':
            raise TelegramReplyUnavailable('telegram_finalize_unavailable')
        deleted = conn.execute('''DELETE FROM v2_inbound_messages WHERE message_id=? AND user_id=?
            AND status='processing' AND lease_owner=? AND claim_generation=?''',
            (checkpoint.receipt.message_id, checkpoint.receipt.user_id,
             checkpoint.receipt.lease_owner, checkpoint.receipt.claim_generation))
        if deleted.rowcount != 1:
            raise TelegramReplyUnavailable('telegram_finalize_unavailable')
    with checkpoint._transaction(_finalize=finish_claim) as conn:
        reply = _reply_locked(conn, checkpoint, context)
        row = conn.execute('SELECT * FROM v2_owner_telegram_replies WHERE message_id=?', (reply['message_id'],)).fetchone()
        if (row is None or row['status'] != 'provider_accepted' or not row['reply_provider_message_id']
                or any(row[key] != reply[key] for key in reply)):
            raise TelegramReplyUnavailable('telegram_finalize_unavailable')
        from alphonse.agent_v2.conversations import SQLiteConversationStore
        event = SQLiteConversationStore._record(conn, event_id=str(uuid4()),
            owner=reply['owner_id'], project='', session='', role='assistant', message=reply['text'],
            source='telegram', source_id=f"outbound:owner:{reply['message_id']}", timestamp=datetime.now(timezone.utc).isoformat())
        if event is None:
            raise TelegramReplyUnavailable('telegram_finalize_unavailable')
        checkpoint_row, raw, state, budget = checkpoint._load(conn, checkpoint.version)
        if 'owner_telegram_finalized' in raw['metadata']:
            raise TelegramReplyUnavailable('telegram_finalize_unavailable')
        raw['metadata']['owner_telegram_finalized'] = {
            'schema_version': 1, 'reply_sha256': hashlib.sha256(reply['text'].encode()).hexdigest(),
            'outgoing_event_id': event.event_id,
            'provider_message_id': row['reply_provider_message_id'],
            'claim_digest': _claim_digest(checkpoint.receipt),
        }
        committed = checkpoint.store._write(conn, checkpoint_row, raw, state, budget)
        changed = conn.execute("UPDATE v2_owner_telegram_replies SET status='finalized' WHERE message_id=? AND status='provider_accepted'", (reply['message_id'],))
        if changed.rowcount != 1:
            raise TelegramReplyUnavailable('telegram_finalize_unavailable')
    checkpoint.version = committed.version


def verify_finalized_telegram_reply_locked(conn, message_id):
    """Verify a finalized Telegram origin on the caller's transaction."""
    try:
        origin = conn.execute('SELECT * FROM v2_owner_telegram_ingress WHERE message_id=?', (message_id,)).fetchone()
        reply = conn.execute('SELECT * FROM v2_owner_telegram_replies WHERE message_id=?', (message_id,)).fetchone()
        if (origin is None or reply is None or reply['status'] != 'finalized' or not reply['reply_provider_message_id']
                or conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone() is not None):
            return False
        owner = _allowed(conn, origin['integration_id'], origin['provider_user_id'], origin['chat_id'])
        if (owner is None or owner['user_id'] != origin['owner_id'] or owner['address_id'] != origin['address_id']
                or owner['credential_generation'] != origin['credential_generation']
                or owner['address_updated_at'] != origin['address_updated_at']
                or owner['integration_updated_at'] != origin['integration_updated_at']
                or any(reply[key] != origin[key] for key in ('message_id', 'owner_id', 'integration_id', 'provider_user_id', 'chat_id', 'provider_message_id'))):
            return False
        incoming = conn.execute('SELECT * FROM v2_conversation_events WHERE event_id=?', (origin['history_event_id'],)).fetchone()
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=? AND owner_id=?',
                           (message_id, origin['owner_id'])).fetchone()
        if incoming is None or row is None:
            return False
        raw = json.loads(row[0]); md = raw['metadata']
        from .single_loop_candidate import CandidateState
        from .single_loop_checkpoint import STATE_KEY
        from .task_budget import TaskBudget
        state = CandidateState.restore(md[STATE_KEY]['state'])
        budget = TaskBudget.from_dict(md[STATE_KEY]['budget'])
        selected, observations, marker = md['owner_read_selected_draft'], md['owner_read_observations'], md['owner_telegram_finalized']
        metadata = {key: md[key] for key in ('intelligence_engine', 'intelligence_schema_version', 'routing_disposition')}
        payload = [message_id, incoming['created_at'], raw['goal'], raw['user'], raw['project_id'], raw['tag'],
                   raw['correlation_id'], raw['memory_session_id'], json.dumps(metadata, sort_keys=True)]
        expected = _digest([payload, origin['integration_id'], origin['provider_user_id'], origin['chat_id'],
                            origin['provider_message_id'], origin['update_id'], origin['address_id'],
                            origin['address_updated_at'], origin['integration_updated_at'],
                            origin['credential_generation'], origin['history_event_id']])
        expected_update = _digest([origin['integration_id'], origin['update_id'], origin['provider_user_id'],
                                   origin['chat_id'], origin['provider_message_id'], raw['goal']])
        outgoing = conn.execute('SELECT * FROM v2_conversation_events WHERE event_id=?', (marker['outgoing_event_id'],)).fetchone()
        return bool(origin['payload_sha256'] == expected and origin['update_sha256'] == expected_update
            and raw['task_id'] == raw['message_id'] == message_id and raw['intelligence_engine'] == 'single_loop_experimental'
            and type(raw['intelligence_schema_version']) is int and raw['intelligence_schema_version'] == 1
            and raw['user'] == origin['owner_id'] and incoming['owner_user_id'] == origin['owner_id']
            and incoming['source'] == 'telegram' and incoming['role'] == 'user'
            and incoming['source_message_id'] == f'inbound:{message_id}' and incoming['content'] == raw['goal']
            and incoming['project_id'] == raw['project_id'] and incoming['visibility'] == 'private'
            and state.status in {'completed', 'waiting_user'} and state.task_id == message_id and state.actor == origin['owner_id']
            and not state.inflight and not budget.usage()['in_flight'] and not budget.usage()['unknown_usage']
            and 'owner_read_pending_decision' not in md and 'owner_read_pending_action' not in md
            and 'owner_read_desktop_pickup_started' not in md
            and selected['kind'] == ('answer' if state.status == 'completed' else 'question')
            and selected['claim_digest'] == marker['claim_digest']
            and md['owner_read_ready_handoff'] == {'schema_version': 1, 'claim_digest': marker['claim_digest'],
                                                   'selected_sha256': _decision_digest(selected)}
            and selected['answer_sha256'] == hashlib.sha256(state.response.encode()).hexdigest()
            and selected['evidence_refs'] == state.evidence_sources and set(observations) == set(state.evidence_sources)
            and selected['evidence_bindings_sha256'] == _decision_digest(observations)
            and marker['schema_version'] == 1 and marker['reply_sha256'] == hashlib.sha256(reply['text'].encode()).hexdigest()
            and marker['provider_message_id'] == reply['reply_provider_message_id']
            and reply['text'] == display_response(state.response, observations)
            and outgoing is not None and outgoing['source'] == 'telegram' and outgoing['role'] == 'assistant'
            and outgoing['owner_user_id'] == origin['owner_id'] and outgoing['content'] == reply['text']
            and outgoing['visibility'] == 'private' and outgoing['project_id'] == raw['project_id']
            and outgoing['source_message_id'] == f'outbound:owner:{message_id}')
    except Exception:
        return False


def verify_finalized_telegram_reply(queue, message_id):
    """Read-only proof of a prior local finalization, never proof of a read."""
    try:
        with queue._connect() as conn:
            conn.execute('BEGIN')
            return verify_finalized_telegram_reply_locked(conn, message_id)
    except Exception:
        return False
