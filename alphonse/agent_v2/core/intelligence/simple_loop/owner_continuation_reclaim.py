"""Acquire an explicitly approved pause without ordinary queue replay."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from alphonse.agent_v2.core.intelligence.task_state import TaskState

from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt, new_claim_generation
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
from alphonse.agent_v2.core.messages.sqlite_queue import _row_to_message
from .owner_atomic_checkpoint import _decision_digest
from .owner_continuation_detached import _identity, open_detached_paused_checkpoint
from .owner_continuation_grant import _valid_approval
from .owner_continuation_question import _paused
from .single_loop_checkpoint import CheckpointConflict


def reclaim_approved_pause(queue, store, message_id, *, lease_owner, lease_seconds=180, clock=None,
                           cancellation_checker):
    """Fence a fresh worker claim and preserve the original claim in lineage.

    The previous claim must be paused and quiescent, authenticated,
    approved and prepared. This does not run or activate work. A caller must
    revalidate sources and consume the grant before any action.
    """
    if (not callable(cancellation_checker) or not isinstance(lease_owner, str) or not lease_owner.strip() or type(lease_seconds) is not int
            or lease_seconds <= 0 or lease_seconds > 180):
        raise ValueError('owner_continuation_reclaim_unavailable')
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('owner_continuation_reclaim_unavailable')
    now = now.astimezone(timezone.utc)
    now_ms = int(now.timestamp()*1000)
    old = open_detached_paused_checkpoint(queue, store, message_id)
    with queue._connect() as conn:
        try:
            conn.execute('BEGIN IMMEDIATE')
            _identity(conn, old.receipt, old.task)
            row, raw, state, budget, pause = _paused(old, conn, old.version, cancellation_checker)
            q = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
            expiry = datetime.fromisoformat(q['lease_expires_at'])
            if expiry.tzinfo is None or expiry.utcoffset() is None:
                raise CheckpointConflict('owner_continuation_reclaim_unavailable')
            md = raw['metadata']
            question, choice = md.get('owner_continuation_question'), md.get('owner_continuation_choice')
            grant, handoff = md.get('owner_continuation_grant'), md.get('owner_continuation_handoff')
            if (not _valid_approval(old, pause, question, choice) or not isinstance(grant, dict)
                    or grant.get('status') != 'consumed' or grant.get('owner_id') != old.task.user
                    or grant.get('message_id') != message_id or grant.get('pause_sha256') != _decision_digest(pause)
                    or grant.get('choice_sha256') != _decision_digest(choice)
                    or grant.get('metric') != pause['metric'] or grant.get('checkpoint') != pause['counter']
                    or grant.get('through') != pause['counter'] + pause['interval']
                    or type(grant.get('issued_at_ms')) is not int or type(grant.get('expires_at_ms')) is not int
                    or not grant['issued_at_ms'] <= now_ms < grant['expires_at_ms']
                    or not isinstance(handoff, dict) or handoff.get('status') != 'held'
                    or handoff.get('grant_sha256') != _decision_digest(grant)
                    or handoff.get('original_claim_digest') != pause['claim_digest']):
                raise CheckpointConflict('owner_continuation_reclaim_unavailable')
            generation = new_claim_generation()
            new = ClaimReceipt(message_id, old.task.user, lease_owner, generation)
            md['owner_continuation_reclaim'] = {'schema_version': 1,
                'original_claim_digest': pause['claim_digest'], 'current_claim_digest': _claim_digest(new),
                'grant_sha256': _decision_digest(grant), 'pause_sha256': _decision_digest(pause)}
            saved = store._write(conn, row, raw, state, budget)
            cursor = conn.execute('''UPDATE v2_inbound_messages SET lease_owner=?,claim_generation=?,
                processing_at=?,lease_expires_at=?,attempt_count=attempt_count+1
                WHERE message_id=? AND status='processing' AND lease_owner=? AND claim_generation=? AND lease_expires_at=?''',
                (lease_owner, generation, now.isoformat(), (now+timedelta(seconds=lease_seconds)).isoformat(),
                 message_id, old.receipt.lease_owner, old.receipt.claim_generation, q['lease_expires_at']))
            if cursor.rowcount != 1:
                raise CheckpointConflict('owner_continuation_reclaim_unavailable')
            if cancellation_checker() is not False:
                raise CheckpointConflict('owner_continuation_reclaim_unavailable')
            updated = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
            conn.commit()
            return _row_to_message(updated, include_claim=True), saved.version
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise


def reclaim_quiescent_activated(queue, store, message_id, *, lease_owner, cancellation_checker,
                                lease_seconds=180, clock=None):
    """Recover an expired activated continuation only when no work is uncertain.

    A settled reservation is never replayed. In-flight calls, pending decisions,
    effect markers, selected output and unknown token usage require separate
    reconciliation, so this operation refuses them.
    """
    if (not callable(cancellation_checker) or not isinstance(lease_owner, str) or not lease_owner.strip()
            or type(lease_seconds) is not int or not 1 <= lease_seconds <= 180):
        raise ValueError('owner_continuation_recovery_unavailable')
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('owner_continuation_recovery_unavailable')
    now = now.astimezone(timezone.utc)
    with queue._connect() as conn:
        try:
            conn.execute('BEGIN IMMEDIATE')
            q = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
            if q is None or q['status'] != 'processing':
                raise CheckpointConflict('owner_continuation_recovery_unavailable')
            old = _row_to_message(q, include_claim=True)
            task = TaskState.from_message(old.message, message_id=message_id)
            task.task_id = message_id
            _identity(conn, old.claim_receipt, task)
            expiry = datetime.fromisoformat(q['lease_expires_at'])
            if expiry.tzinfo is None or expiry.utcoffset() is None or expiry.astimezone(timezone.utc) > now:
                raise CheckpointConflict('owner_continuation_recovery_unavailable')
            row = store._row(conn, message_id)
            raw = store._validate_identity(row, message_id, task.user, row['version'])
            state, budget = store._decode(raw, message_id, task.user)
            md = raw['metadata']
            history = md.get('owner_continuation_history')
            crash = md.get('owner_continuation_crash_lineage', [])
            forbidden = ('owner_continuation_pause', 'owner_continuation_reclaim', 'owner_read_pending_action',
                'owner_read_pending_decision', 'owner_read_selected_draft', 'owner_read_desktop_pickup_started',
                'owner_read_ready_handoff', 'owner_bash_dispatch_started', 'owner_bash_return_unreconciled')
            if (cancellation_checker() is not False or state.inflight or budget.usage()['in_flight']
                    or budget.usage()['unknown_usage'] or not budget.within_work_limits()
                    or any(k in md for k in forbidden) or not isinstance(history, list) or not history
                    or not isinstance(crash, list) or not isinstance(history[-1], dict)):
                raise CheckpointConflict('owner_continuation_recovery_unavailable')
            last = history[-1]
            prior = last.get('reclaim')
            grant = last.get('grant')
            handoff = last.get('handoff')
            windows = budget.to_dict().get('continuation_windows', [])
            if (not isinstance(last.get('pause'), dict) or not isinstance(grant, dict)
                    or not _valid_approval(SimpleNamespace(task=task), last['pause'], last.get('question'), last.get('choice'))
                    or grant.get('status') != 'consumed' or grant.get('owner_id') != task.user
                    or grant.get('message_id') != message_id or grant.get('pause_sha256') != _decision_digest(last['pause'])
                    or grant.get('choice_sha256') != _decision_digest(last['choice'])
                    or not isinstance(handoff, dict) or handoff.get('status') != 'held'
                    or handoff.get('grant_sha256') != _decision_digest(grant)
                    or handoff.get('original_claim_digest') != last['pause'].get('claim_digest')
                    or not isinstance(windows, list) or not any(w.get('grant_id') == _decision_digest(grant)
                                                              for w in windows if isinstance(w, dict))):
                raise CheckpointConflict('owner_continuation_recovery_unavailable')
            initial = prior.get('current_claim_digest') if isinstance(prior, dict) else last['pause'].get('claim_digest')
            original = prior.get('original_claim_digest') if isinstance(prior, dict) else initial
            if (not isinstance(initial, str) or original != last['pause'].get('claim_digest')
                    or (prior is not None and (prior.get('pause_sha256') != _decision_digest(last['pause'])
                        or prior.get('grant_sha256') != _decision_digest(last['grant'])))):
                raise CheckpointConflict('owner_continuation_recovery_unavailable')
            current = initial
            for link in crash:
                if (not isinstance(link, dict) or set(link) != {'from','to','grant_sha256'}
                        or link['from'] != current or link['grant_sha256'] != _decision_digest(last['grant'])
                        or not isinstance(link['to'], str)):
                    raise CheckpointConflict('owner_continuation_recovery_unavailable')
                current = link['to']
            if current != _claim_digest(old.claim_receipt):
                raise CheckpointConflict('owner_continuation_recovery_unavailable')
            generation = new_claim_generation()
            new = ClaimReceipt(message_id, task.user, lease_owner, generation)
            crash.append({'from':current, 'to':_claim_digest(new), 'grant_sha256':_decision_digest(last['grant'])})
            md['owner_continuation_crash_lineage'] = crash
            saved = store._write(conn, row, raw, state, budget)
            changed = conn.execute('''UPDATE v2_inbound_messages SET lease_owner=?,claim_generation=?,processing_at=?,
                lease_expires_at=?,attempt_count=attempt_count+1 WHERE message_id=? AND status='processing'
                AND lease_owner=? AND claim_generation=? AND lease_expires_at=?''',
                (lease_owner, generation, now.isoformat(), (now+timedelta(seconds=lease_seconds)).isoformat(),
                 message_id, old.claim_receipt.lease_owner, old.claim_receipt.claim_generation, q['lease_expires_at']))
            if changed.rowcount != 1 or cancellation_checker() is not False:
                raise CheckpointConflict('owner_continuation_recovery_unavailable')
            updated = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
            conn.commit()
            return _row_to_message(updated, include_claim=True), saved.version
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
