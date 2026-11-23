"""One-shot Telegram delivery for an owner-created scheduled result.

The scheduled occurrence retains its own origin and claim. This code never
manufactures a Telegram or Desktop ingress for the due task. The caller must
stage and dispatch before consuming that claim; a failed or ambiguous send is
never repeated automatically.
"""
from __future__ import annotations

import hashlib

from alphonse.agent_v2.core.messages.claim_fence import _read_claim, _time
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
from .owner_atomic_checkpoint import OwnerAtomicCheckpoint, _decision_digest
from .owner_authority_bridge import context_authority_active
from .owner_clock import display_response
from .owner_read_recovery import restore_owner_read_observations
from .owner_schedule_origin import (verified_occurrence_locked, verified_schedule_origin_locked,
                                    verified_telegram_schedule_destination_locked)
from .owner_telegram_delivery import TelegramReplyUnavailable, dispatch_once, stage_locked


def is_telegram_scheduled_origin(checkpoint):
    """Recognize a Telegram-origin schedule even if its grant was revoked.

    This is a routing hint only: stage/dispatch independently revalidate the
    current actor, destination, source and claim. It never grants permission.
    Return None if origin cannot be checked so callers cannot silently treat
    an unavailable Telegram binding as a confirmed non-Telegram schedule.
    """
    if type(checkpoint) is not OwnerAtomicCheckpoint or not checkpoint.scheduled_origin:
        return False
    try:
        with checkpoint.queue._connect() as conn:
            table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_schedule_telegram_origins'").fetchone()
            occurrence = conn.execute('SELECT schedule_id FROM v2_owner_schedule_occurrences WHERE message_id=?',
                                      (checkpoint.receipt.message_id,)).fetchone()
            if not occurrence:
                return None
            if table and conn.execute('''SELECT 1 FROM v2_owner_schedule_occurrences x
                JOIN v2_owner_schedule_telegram_origins o ON o.scheduled_task_id=x.schedule_id
                WHERE x.message_id=?''', (checkpoint.receipt.message_id,)).fetchone():
                return True
            # Pre-Telegram schedules have no Telegram-origin table. Only a
            # still-verified local Desktop origin proves the negative case;
            # missing Telegram provenance alone must never imply Desktop.
            local_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_trusted_owner_ingress'").fetchone()
            if not local_table:
                return None
            local = conn.execute('''SELECT o.request_message_id FROM v2_owner_schedule_origins o
                JOIN v2_trusted_owner_ingress i ON i.message_id=o.request_message_id
                WHERE o.scheduled_task_id=?''', (occurrence['schedule_id'],)).fetchone()
            if local is None:
                return None
            telegram_ingress = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_telegram_ingress'").fetchone()
            if telegram_ingress and conn.execute('SELECT 1 FROM v2_owner_telegram_ingress WHERE message_id=?',
                                                (local['request_message_id'],)).fetchone():
                return None
            return False if verified_schedule_origin_locked(conn, occurrence['schedule_id']) is not None else None
    except Exception:
        return None


def _reply_locked(conn, checkpoint, context):
    if (type(checkpoint) is not OwnerAtomicCheckpoint or not checkpoint.scheduled_origin
            or context.messages is not checkpoint.queue or context.claim_receipt is not checkpoint.receipt
            or not context_authority_active(context) or not callable(context.cancellation_checker)
            or context.cancellation_checker() is not False):
        raise TelegramReplyUnavailable('telegram_scheduled_reply_unavailable')
    receipt, task = checkpoint.receipt, checkpoint.task
    now = _time(checkpoint.clock())
    _read_claim(conn, receipt, now)
    occurrence = conn.execute('''SELECT schedule_id,owner_id,state FROM v2_owner_schedule_occurrences
        WHERE message_id=?''', (receipt.message_id,)).fetchone()
    if (occurrence is None or occurrence['owner_id'] != receipt.user_id or occurrence['state'] != 'queued'
            or not verified_occurrence_locked(conn, receipt.message_id, now=checkpoint.queue._owner_schedule_clock())):
        raise TelegramReplyUnavailable('telegram_scheduled_reply_unavailable')
    destination = verified_telegram_schedule_destination_locked(conn, occurrence['schedule_id'])
    if destination is None or destination['owner_id'] != task.user or task.user != receipt.user_id:
        raise TelegramReplyUnavailable('telegram_scheduled_reply_unavailable')
    _, raw, state, budget = checkpoint._load(conn, checkpoint.version)
    metadata = raw['metadata']
    selected = metadata.get('owner_read_selected_draft')
    observations = metadata.get('owner_read_observations')
    if (state.status not in {'completed', 'waiting_user'} or not state.response.strip() or state.inflight
            or budget.usage()['in_flight'] or budget.usage()['unknown_usage']
            or 'owner_read_pending_decision' in metadata or 'owner_read_pending_action' in metadata
            or not isinstance(selected, dict) or selected.get('schema_version') != 1
            or selected.get('kind') != ('answer' if state.status == 'completed' else 'question')
            or selected.get('claim_digest') != _claim_digest(receipt)
            or selected.get('answer_sha256') != hashlib.sha256(state.response.encode()).hexdigest()
            or selected.get('evidence_refs') != state.evidence_sources or not isinstance(observations, dict)
            or set(observations) != set(state.evidence_sources)
            or selected.get('evidence_bindings_sha256') != _decision_digest(observations)):
        raise TelegramReplyUnavailable('telegram_scheduled_reply_unavailable')
    # Source refresh and this transaction are separate; fence the DB-backed
    # Gmail grant again before staging or dispatching an external reply.
    from .owner_gmail import OWNER_GMAIL_TOOL_ID, valid_owner_gmail_locked
    for binding in observations.values():
        if (binding.get('tool_id') == OWNER_GMAIL_TOOL_ID
                and (state.status != 'completed' or not valid_owner_gmail_locked(
                    conn, task.user, binding.get('gmail_observation'), context=context))):
            raise TelegramReplyUnavailable('telegram_scheduled_reply_source_unavailable')
    return {key: destination[key] for key in ('owner_id', 'integration_id', 'provider_user_id', 'chat_id', 'provider_message_id')} | {
        'message_id': receipt.message_id, 'text': display_response(state.response, observations)}


def stage_scheduled_telegram_reply(checkpoint, context):
    """Stage the exact selected result under the still-live scheduled claim."""
    restore_owner_read_observations(checkpoint, context, expected_version=checkpoint.version, allow_completed=True)
    with checkpoint._transaction() as conn:
        reply = _reply_locked(conn, checkpoint, context)
        stage_locked(conn, reply, verify=lambda c, value: _reply_locked(c, checkpoint, context) == value)
    return reply['message_id']


def dispatch_scheduled_telegram_reply(checkpoint, context, *, send):
    """Recheck source and destination, then try the exact result at most once."""
    restore_owner_read_observations(checkpoint, context, expected_version=checkpoint.version, allow_completed=True)
    return dispatch_once(checkpoint.queue.db_path, checkpoint.receipt.message_id,
        verify=lambda conn, value: _reply_locked(conn, checkpoint, context) == value, send=send)
