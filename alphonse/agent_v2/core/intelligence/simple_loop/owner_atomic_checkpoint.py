"""Owner-claim-bound candidate checkpoints in the queue's SQLite database.

This is an experimental persistence boundary, not a registered processor.
The caller must validate read authorization and observation freshness at the
I/O boundary and again before using restored observations. Neither external
calls nor delivery are fenced by a SQLite transaction.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
import re
import secrets
from typing import Any

from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
from .single_loop_checkpoint import Checkpoint, CheckpointConflict, ExperimentalCheckpointStore, STATE_KEY, SCHEMA_VERSION
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetExhausted, TaskBudget


def observation_digest(observation: dict) -> str:
    """Fingerprint canonical observed bytes; this is not source authority."""
    encoded = json.dumps(observation, ensure_ascii=False, sort_keys=True, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _decision_digest(decision: Any) -> str:
    """Bind the exact JSON decision without retaining its private bytes."""
    encoded = json.dumps(decision, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(encoded) > 262144:
        raise ValueError('owner_atomic_decision_too_large')
    return hashlib.sha256(encoded).hexdigest()


def _matches_decision(marker: Any, receipt: Any, decision: Any, fingerprint: str) -> bool:
    return bool(isinstance(marker, dict) and set(marker) == {
        'reservation_id', 'claim_digest', 'decision_sha256', 'projection_sha256'}
        and marker['claim_digest'] == _claim_digest(receipt)
        and isinstance(fingerprint, str) and re.fullmatch(r'[0-9a-f]{64}', fingerprint)
        and marker['projection_sha256'] == fingerprint
        and marker['decision_sha256'] == _decision_digest(decision))


def _steering_claim_in_lineage(metadata: dict, receipt: Any, claimed: str) -> bool:
    """Keep accepted steering bound to its original claim across continuation."""
    valid = {_claim_digest(receipt)}
    if claimed in valid:
        return True
    try:
        lineage = metadata.get('owner_continuation_reclaim')
        if lineage is not None:
            pause = metadata.get('owner_continuation_pause')
            if (not isinstance(lineage, dict) or not isinstance(pause, dict)
                    or lineage.get('current_claim_digest') not in valid
                    or lineage.get('original_claim_digest') != pause.get('claim_digest')
                    or lineage.get('pause_sha256') != _decision_digest(pause)
                    or lineage.get('grant_sha256') != _decision_digest(metadata.get('owner_continuation_grant'))):
                return False
            valid.add(lineage['original_claim_digest'])
        def add_crashes(crashes):
            if not isinstance(crashes, list):
                raise ValueError('invalid continuation lineage')
            for crash in reversed(crashes):
                if (not isinstance(crash, dict) or set(crash) != {'from', 'to', 'grant_sha256'}
                        or crash['to'] not in valid or not isinstance(crash['from'], str)):
                    raise ValueError('invalid continuation lineage')
                valid.add(crash['from'])
        add_crashes(metadata.get('owner_continuation_crash_lineage', []))
        history = metadata.get('owner_continuation_history', [])
        if not isinstance(history, list):
            return False
        for entry in reversed(history):
            if not isinstance(entry, dict):
                return False
            link = entry.get('reclaim')
            if link is not None:
                pause = entry.get('pause')
                if (not isinstance(link, dict) or not isinstance(pause, dict)
                        or link.get('current_claim_digest') not in valid
                        or link.get('original_claim_digest') != pause.get('claim_digest')
                        or link.get('pause_sha256') != _decision_digest(pause)
                        or link.get('grant_sha256') != _decision_digest(entry.get('grant'))):
                    return False
                valid.add(link['original_claim_digest'])
            add_crashes(entry.get('crash_before_pause', []))
        return claimed in valid
    except (KeyError, TypeError, ValueError):
        return False


def verify_finalized_owner_ack(queue: SQLiteMessageQueue, message_id: str, token: str) -> bool:
    """Verify a previously committed owner ack without recreating a claim.

    Caller must authenticate the local peer. A committed acknowledgment is a
    client action, not proof that a human read the response.
    """
    if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
            or not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None
            or not isinstance(token, str) or re.fullmatch(r'[0-9a-f]{32}', token) is None):
        return False
    try:
        with queue._connect() as conn:
            conn.execute('BEGIN')
            row = conn.execute("""SELECT o.owner_id,o.peer_uid,o.payload_sha256,o.created_at,
                    o.history_version,o.history_event_id,c.task_state_json,c.owner_id AS checkpoint_owner
                FROM v2_trusted_owner_ingress o
                JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=o.owner_id
                JOIN v2_users u ON u.user_id=o.owner_id AND u.role IN ('admin','owner') AND u.is_active=1
                JOIN v2_task_checkpoints c ON c.task_id=o.message_id AND c.owner_id=o.owner_id
                WHERE o.message_id=? AND NOT EXISTS
                    (SELECT 1 FROM v2_inbound_messages q WHERE q.message_id=o.message_id)""", (message_id,)).fetchone()
            if row is None or row['peer_uid'] != os.getuid():
                return False
            raw = json.loads(row['task_state_json'])
            metadata = raw['metadata']
            marker = metadata['owner_read_desktop_pickup_started']
            selected = metadata['owner_read_selected_draft']
            state = CandidateState.restore(metadata[STATE_KEY]['state'])
            supplied = hashlib.sha256(token.encode('ascii')).hexdigest()
            if (raw.get('task_id') != message_id or raw.get('message_id') != message_id
                    or raw.get('user') != row['owner_id'] or raw.get('intelligence_engine') != 'single_loop_experimental'
                    or state.task_id != message_id or state.actor != row['owner_id']
                    or state.status not in {'completed', 'waiting_user'}
                    or selected.get('kind') != ('answer' if state.status == 'completed' else 'question')
                    or state.inflight or marker.get('status') != 'client_acknowledged'
                    or not isinstance(marker.get('acknowledgement_sha256'), str)
                    or not hmac.compare_digest(marker['acknowledgement_sha256'], supplied)
                    or marker.get('answer_sha256') != selected.get('answer_sha256')
                    or marker.get('answer_sha256') != hashlib.sha256(state.response.encode('utf-8')).hexdigest()):
                return False
            rows = conn.execute("""SELECT * FROM v2_conversation_events
                WHERE source_message_id IN (?,?)""", (f'inbound:{message_id}', f'outbound:owner:{message_id}')).fetchall()
            entries = {item['source_message_id']: (item['role'], item['content'], item['owner_user_id']) for item in rows}
            observations = metadata['owner_read_observations']
            if not isinstance(observations, dict) or selected.get('evidence_bindings_sha256') != _decision_digest(observations):
                return False
            from alphonse.agent_v2.core.messages.sqlite_queue import finalized_owner_history_matches
            if not finalized_owner_history_matches(conn, message_id, row['owner_id'], raw):
                return False
            from .owner_clock import display_response
            return entries == {f'inbound:{message_id}': ('user', raw['goal'], row['owner_id']),
                               f'outbound:owner:{message_id}': ('assistant', display_response(state.response, observations), row['owner_id'])}
    except Exception:
        return False


class OwnerAtomicCheckpoint:
    """Persist candidate state, budget and captured read bindings atomically.

    All methods require the same live owner claim. A reservation is not proof
    that a tool is authorized, nor is a captured observation proof it remains
    visible or fresh. An unresolved reservation must not be replayed.
    """

    def __init__(self, queue: SQLiteMessageQueue, store: ExperimentalCheckpointStore, task: Any, receipt: Any, *, clock):
        if (type(queue) is not SQLiteMessageQueue or not isinstance(store, ExperimentalCheckpointStore)
                or not callable(clock) or not os.path.samefile(queue.db_path, store.store.db_path)):
            raise ValueError('owner_atomic_checkpoint_configuration_invalid')
        self.queue, self.store, self.task, self.receipt, self.clock = queue, store, task, receipt, clock
        with queue._connect() as conn:
            table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_schedule_occurrences'").fetchone()
            self.scheduled_origin = bool(table and conn.execute(
                'SELECT 1 FROM v2_owner_schedule_occurrences WHERE message_id=?', (task.message_id,)).fetchone())
            telegram_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_telegram_ingress'").fetchone()
            self.telegram_origin = bool(telegram_table and conn.execute(
                'SELECT 1 FROM v2_owner_telegram_ingress WHERE message_id=?', (task.message_id,)).fetchone())
            if self.scheduled_origin and self.telegram_origin:
                raise ValueError('owner_atomic_checkpoint_origin_conflict')
        self.version = None

    @contextmanager
    def _transaction(self, *, _finalize=None):
        if self.telegram_origin:
            from .owner_telegram_ingress import telegram_owner_claim_transaction
            transaction = telegram_owner_claim_transaction(self.queue, self.receipt, self.task, clock=self.clock, _finalize=_finalize)
        else:
            claim = (self.queue.trusted_scheduled_owner_claim_transaction if self.scheduled_origin
                     else self.queue.trusted_local_owner_claim_transaction)
            transaction = claim(self.receipt, self.task, clock=self.clock, _finalize=_finalize)
        def check(conn):
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_continuation_cancellations'").fetchone()
            if exists and conn.execute('SELECT 1 FROM v2_owner_continuation_cancellations WHERE message_id=?',
                                       (self.task.message_id,)).fetchone():
                raise CheckpointConflict('owner_continuation_cancelled')
        with transaction as conn:
            check(conn)
            yield conn
            check(conn)

    def check_claim(self, reservation_id: str, kind: str, *, tool_id: str = '', arguments: dict | None = None) -> bool:
        """Check the claim and exact active action immediately before dispatch.

        The check cannot fence I/O after this transaction releases its lock.
        """
        expected = {'reservation_id': reservation_id, 'kind': kind, 'tool_id': tool_id,
                    'arguments': arguments, 'claim_digest': _claim_digest(self.receipt)}
        try:
            with self._transaction() as conn:
                _, raw, state, budget = self._load(conn, self.version)
                pending = budget.to_dict()['reservations'].get(reservation_id)
                return (state.status == 'running' and state.inflight == reservation_id
                        and isinstance(pending, dict) and pending['status'] == 'reserved' and pending['kind'] == kind
                        and json.dumps(raw['metadata'].get('owner_read_pending_action'), sort_keys=True)
                        == json.dumps(expected, sort_keys=True))
        except Exception:
            return False

    def steering_context(self) -> list[str]:
        """Read ordered steer text only under the current verified owner claim."""
        with self._transaction() as conn:
            _, raw, _, _ = self._load(conn, self.version)
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_steering'").fetchone():
                return []
            rows = conn.execute('''SELECT * FROM v2_owner_steering WHERE message_id=? AND owner_id=?
                ORDER BY sequence''', (self.task.message_id, self.task.user)).fetchall()
            result = []
            for row in rows:
                digest = hashlib.sha256((row['id'] + '\0' + row['content']).encode()).hexdigest()
                if (row['peer_uid'] != os.getuid() or not _steering_claim_in_lineage(raw['metadata'], self.receipt, row['claim_digest'])
                        or not hmac.compare_digest(row['content_digest'], digest)):
                    raise CheckpointConflict('owner_steering_binding_unavailable')
                result.append(row['content'])
            return result

    def mark_steering_included(self, decision: dict, fingerprint: str, cancellation_checker) -> None:
        """Record validated model-input inclusion, never task/effect success."""
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, self.version)
            marker = raw['metadata'].get('owner_steering_pending_projection')
            if marker is None:
                return
            selected = raw['metadata'].get('owner_read_pending_decision')
            if (not callable(cancellation_checker) or cancellation_checker() is not False
                    or state.status != 'running' or state.inflight
                    or budget.usage()['unknown_usage'] or not budget.within_work_limits()
                    or not isinstance(marker, dict)
                    or marker.get('projection_sha256') != fingerprint
                    or not _matches_decision(selected, self.receipt, decision, fingerprint)):
                raise CheckpointConflict('owner_steering_projection_unavailable')
            rows = conn.execute('''SELECT * FROM v2_owner_steering WHERE message_id=? AND owner_id=?
                ORDER BY sequence''', (self.task.message_id, self.task.user)).fetchall()
            ids = [item['id'] for item in rows]
            if (ids != marker.get('steering_ids')
                    or _decision_digest([[item['id'], item['content_digest']] for item in rows]) != marker.get('steering_sha256')):
                raise CheckpointConflict('owner_steering_projection_unavailable')
            for item in rows:
                digest = hashlib.sha256((item['id'] + '\0' + item['content']).encode()).hexdigest()
                if (item['peer_uid'] != os.getuid() or not _steering_claim_in_lineage(raw['metadata'], self.receipt, item['claim_digest'])
                        or not hmac.compare_digest(item['content_digest'], digest)):
                    raise CheckpointConflict('owner_steering_binding_unavailable')
            conn.execute("UPDATE v2_owner_steering SET status='included_in_planning' WHERE message_id=? AND owner_id=? AND status='accepted'",
                         (self.task.message_id, self.task.user))
            raw['metadata']['owner_steering_included_projection'] = marker
            del raw['metadata']['owner_steering_pending_projection']
            if cancellation_checker() is not False:
                raise CheckpointConflict('owner_steering_cancelled')
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version

    def _load(self, conn, version, *, allow_paused=False):
        row = self.store._row(conn, self.task.task_id)
        raw = self.store._validate_identity(row, self.task.task_id, self.task.user, version)
        if raw.get('message_id') != self.receipt.message_id:
            raise CheckpointConflict('owner_atomic_checkpoint_identity_mismatch')
        if 'owner_continuation_pause' in raw['metadata'] and not allow_paused:
            raise CheckpointConflict('owner_continuation_paused')
        envelope = raw['metadata'].get(STATE_KEY)
        if (not isinstance(envelope, dict) or set(envelope) != {'schema_version', 'state', 'budget'}
                or type(envelope['schema_version']) is not int or envelope['schema_version'] != SCHEMA_VERSION):
            raise CheckpointConflict('owner_atomic_envelope_invalid')
        state = CandidateState.restore(envelope['state'])
        budget = TaskBudget.from_dict(envelope['budget'])
        pending = budget.usage()['in_flight']
        if (state.task_id != self.task.task_id or state.actor != self.task.user
                or (state.inflight and pending != [state.inflight]) or (not state.inflight and pending)):
            raise CheckpointConflict('owner_atomic_state_invalid')
        return row, raw, state, budget

    def initialize(self, state: CandidateState, budget: TaskBudget) -> Checkpoint:
        if (state.task_id != self.task.task_id or state.actor != self.task.user
                or state.status != 'running' or state.inflight or state.evidence_sources
                or state.response or state.blocker or budget.revision != 0):
            raise ValueError('owner_atomic_initial_state_invalid')
        CandidateState.restore(state.snapshot())
        TaskBudget.from_dict(budget.to_dict())
        with self._transaction() as conn:
            if self.store._row(conn, state.task_id) is not None:
                raise CheckpointConflict('owner_atomic_checkpoint_already_exists')
            version = self.store.store.save_task_checkpoint(self.task, status='running', connection=conn)
            row = self.store._row(conn, state.task_id)
            raw = self.store._validate_identity(row, state.task_id, state.actor, version)
            raw['metadata']['owner_read_observations'] = {}
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return result

    def load(self, *, expected_version: int) -> tuple[Checkpoint, dict]:
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version, allow_paused=True)
            if 'owner_read_pending_decision' in raw['metadata']:
                raise CheckpointConflict('owner_atomic_pending_decision_unresolved')
            observations = raw['metadata'].get('owner_read_observations')
            if not isinstance(observations, dict) or set(observations) != set(state.evidence_sources):
                raise CheckpointConflict('owner_atomic_observation_bindings_invalid')
            return Checkpoint(row['version'], state, budget), deepcopy(observations)

    def mark_ready_handoff(self) -> None:
        """Record a completed cancellation handoff before exposing a draft.

        The caller must hold the daemon's kill-switch handoff lock and its
        single-instance lock. This only records a selected, unpicked draft; it
        does not authorize pickup, settle an effect, or extend the lease.
        """
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, self.version)
            md = raw['metadata']
            selected = md.get('owner_read_selected_draft')
            expected_ready = {
                'schema_version': 1, 'claim_digest': _claim_digest(self.receipt),
                'selected_sha256': _decision_digest(selected),
            } if isinstance(selected, dict) else None
            if (expected_ready is not None and md.get('owner_read_ready_handoff') == expected_ready
                    and 'owner_read_desktop_pickup_started' not in md
                    and state.status in {'completed', 'waiting_user'} and not state.inflight
                    and not budget.usage()['in_flight'] and not budget.usage()['unknown_usage']
                    and 'owner_read_pending_action' not in md and 'owner_read_pending_decision' not in md
                    and selected.get('claim_digest') == _claim_digest(self.receipt)
                    and selected.get('answer_sha256') == hashlib.sha256(state.response.encode('utf-8')).hexdigest()):
                return
            if (state.status not in {'completed', 'waiting_user'} or state.inflight
                    or budget.usage()['in_flight'] or budget.usage()['unknown_usage']
                    or 'owner_read_pending_action' in md or 'owner_read_pending_decision' in md
                    or 'owner_read_desktop_pickup_started' in md or 'owner_read_ready_handoff' in md
                    or not isinstance(selected, dict)
                    or selected.get('claim_digest') != _claim_digest(self.receipt)
                    or selected.get('answer_sha256') != hashlib.sha256(state.response.encode('utf-8')).hexdigest()):
                raise CheckpointConflict('owner_ready_handoff_unavailable')
            md['owner_read_ready_handoff'] = {
                'schema_version': 1, 'claim_digest': _claim_digest(self.receipt),
                'selected_sha256': _decision_digest(selected),
            }
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version

    def take_desktop_draft(self, context: Any) -> dict:
        """Start one owner-only pickup after revalidating its sources.

        A started pickup is never replayed automatically: a lost IPC response
        leaves delivery uncertain. This does not confirm user receipt, fence
        external I/O, authorize other channels, or enable the normal daemon.
        """
        if self.telegram_origin:
            raise CheckpointConflict('owner_desktop_pickup_unavailable')
        from .owner_authority_bridge import context_authority_active
        from .owner_read_recovery import restore_owner_read_observations
        if not context_authority_active(context):
            raise CheckpointConflict('owner_desktop_pickup_unavailable')
        expected_version = self.version
        restore_owner_read_observations(self, context, expected_version=expected_version, allow_completed=True)
        acknowledgement_token = secrets.token_hex(16)
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            metadata = raw['metadata']
            if 'owner_clarification' in self.task.metadata:
                from .owner_clarification import load_bound_owner_question
                load_bound_owner_question(conn, self.task.user, self.task.metadata['owner_clarification'])
            if 'owner_followup' in self.task.metadata:
                from .owner_clarification import load_bound_owner_followup
                load_bound_owner_followup(conn, self.task.user, self.task.metadata['owner_followup'])
            selected = metadata.get('owner_read_selected_draft')
            observations = metadata.get('owner_read_observations')
            steer_marker = metadata.get('owner_steering_included_projection')
            if (steer_marker is None and conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_steering'").fetchone()
                    and conn.execute('SELECT 1 FROM v2_owner_steering WHERE message_id=? AND owner_id=? LIMIT 1',
                                     (self.task.message_id, self.task.user)).fetchone()):
                raise CheckpointConflict('owner_steering_projection_unavailable')
            if steer_marker is not None:
                if not isinstance(steer_marker, dict) or not isinstance(selected, dict):
                    raise CheckpointConflict('owner_steering_projection_unavailable')
                steer_rows = conn.execute('''SELECT * FROM v2_owner_steering WHERE message_id=? AND owner_id=?
                    ORDER BY sequence''', (self.task.message_id, self.task.user)).fetchall()
                if (steer_marker.get('reservation_id') != selected.get('model_reservation_id')
                        or steer_marker.get('projection_sha256') != selected.get('projection_sha256')
                        or steer_marker.get('steering_ids') != [item['id'] for item in steer_rows]
                        or steer_marker.get('steering_sha256') != _decision_digest([[item['id'], item['content_digest']] for item in steer_rows])):
                    raise CheckpointConflict('owner_steering_projection_unavailable')
                for item in steer_rows:
                    digest = hashlib.sha256((item['id'] + '\0' + item['content']).encode()).hexdigest()
                    if (item['status'] != 'included_in_planning' or item['peer_uid'] != os.getuid()
                            or not _steering_claim_in_lineage(metadata, self.receipt, item['claim_digest'])
                            or not hmac.compare_digest(item['content_digest'], digest)):
                        raise CheckpointConflict('owner_steering_binding_unavailable')
            if (state.status not in {'completed', 'waiting_user'} or state.inflight or not state.response.strip()
                    or context.cancellation_checker() is not False
                    or 'owner_read_pending_decision' in metadata or 'owner_read_pending_action' in metadata
                    or 'owner_read_desktop_pickup_started' in metadata
                    or not isinstance(selected, dict) or selected.get('schema_version') != 1
                    or selected.get('kind') != ('answer' if state.status == 'completed' else 'question')
                    or selected.get('claim_digest') != _claim_digest(self.receipt)
                    or selected.get('answer_sha256') != hashlib.sha256(state.response.encode('utf-8')).hexdigest()
                    or selected.get('evidence_refs') != state.evidence_sources
                    or not isinstance(observations, dict) or set(observations) != set(state.evidence_sources)
                    or selected.get('evidence_bindings_sha256') != _decision_digest(observations)):
                raise CheckpointConflict('owner_desktop_pickup_unavailable')
            from .owner_gmail import OWNER_GMAIL_TOOL_ID, valid_owner_gmail_locked
            from .owner_calendar import OWNER_CALENDAR_TOOL_ID, valid_owner_calendar_locked
            from .owner_gmail_search import (
                OWNER_GMAIL_SEARCH_TOOL_ID, final_gmail_search_result, gmail_source_title, gmail_summary_text,
                valid_owner_gmail_search_locked,
            )
            from alphonse.agent_v2.interfaces.private_provider_a2ui import calendar_agenda_surface, gmail_summary_surface
            from .owner_reminder_create import OWNER_REMINDER_CREATE_TOOL_ID, valid_reminder_locked
            from .owner_schedule_create import SCHEDULED_TASK_TOOL_ID, valid_schedule_locked
            from .owner_schedule_cancel import valid_cancel_locked
            from .owner_knowledge_snapshot import OWNER_KNOWLEDGE_TOOL_ID, matches_owner_knowledge_locked
            for binding in observations.values():
                if (binding.get('tool_id') == OWNER_KNOWLEDGE_TOOL_ID
                        and (binding.get('arguments') != {} or not matches_owner_knowledge_locked(
                            conn, self.task, binding.get('knowledge_snapshot_digest'),
                            cancellation_checker=context.cancellation_checker, clock=self.clock))):
                    raise CheckpointConflict('owner_knowledge_binding_unavailable')
                if (binding.get('tool_id') == OWNER_GMAIL_TOOL_ID
                        and (state.status != 'completed' or not valid_owner_gmail_locked(conn, self.task.user,
                                binding.get('gmail_observation'), context=context,
                                desktop_pickup=not self.scheduled_origin))):
                    raise CheckpointConflict('owner_gmail_authority_unavailable')
                if (binding.get('tool_id') == OWNER_CALENDAR_TOOL_ID
                        and (state.status != 'completed' or not valid_owner_calendar_locked(
                            conn, self.task.user, binding.get('arguments'), binding.get('calendar_observation'),
                            context=context))):
                    raise CheckpointConflict('owner_calendar_authority_unavailable')
                if (binding.get('tool_id') == OWNER_GMAIL_SEARCH_TOOL_ID
                        and (state.status != 'completed' or not valid_owner_gmail_search_locked(
                            conn, self.task.user, binding.get('arguments'), binding.get('gmail_search_observation'),
                            context=context))):
                    raise CheckpointConflict('owner_gmail_search_authority_unavailable')
                if (binding.get('tool_id') == OWNER_REMINDER_CREATE_TOOL_ID
                        and not valid_reminder_locked(conn, self.task.user, binding.get('arguments'),
                                                      binding.get('reminder_observation'), task=self.task)):
                    raise CheckpointConflict('owner_reminder_unavailable')
                if (binding.get('tool_id') == SCHEDULED_TASK_TOOL_ID
                        and not (valid_cancel_locked if binding.get('arguments', {}).get('operation') == 'cancel'
                                 else valid_schedule_locked)(conn, self.task.user, binding.get('arguments'),
                                                              binding.get('schedule_observation'), task=self.task)):
                    raise CheckpointConflict('owner_schedule_unavailable')
            metadata['owner_read_desktop_pickup_started'] = {
                'schema_version': 1, 'claim_digest': _claim_digest(self.receipt),
                'answer_sha256': selected['answer_sha256'], 'channel': 'desktop',
                'acknowledgement_sha256': hashlib.sha256(acknowledgement_token.encode('ascii')).hexdigest(),
                'status': 'started_unconfirmed',
            }
            metadata.pop('owner_read_ready_handoff', None)
            result = self.store._write(conn, row, raw, state, budget)
            from .owner_clock import display_response
            response = {'message_id': self.task.message_id,
                        'message': display_response(state.response, observations, private_desktop=True),
                        'kind': selected['kind'],
                        'delivery_status': 'started_unconfirmed', 'acknowledgement_token': acknowledgement_token}
            searches = [binding['gmail_search_observation'] for binding in observations.values()
                        if binding.get('tool_id') == OWNER_GMAIL_SEARCH_TOOL_ID]
            mail = final_gmail_search_result(searches)
            if mail is not None and mail['status'] in ('complete', 'partial'):
                response['private_gmail'] = True
                response['private_a2ui'] = gmail_summary_surface(
                    self.task.message_id, mail, source=gmail_source_title(mail),
                    summary=gmail_summary_text(state.response), searches=searches)
            for binding in observations.values():
                calendar = binding.get('calendar_observation')
                if binding.get('tool_id') == OWNER_CALENDAR_TOOL_ID and calendar['status'] in ('complete', 'partial'):
                    response['private_calendar'] = True
                    local_timezone = context.user_timezone_provider
                    response['private_a2ui'] = calendar_agenda_surface(
                        self.task.message_id, calendar,
                        timezone_name=local_timezone(self.task.user) if callable(local_timezone) else None)
                if (binding.get('tool_id') == SCHEDULED_TASK_TOOL_ID
                        and binding.get('arguments', {}).get('operation') == 'cancel'):
                    effect = binding['schedule_observation']
                    response['committed_effect'] = {'kind': 'scheduled_task_cancelled', 'schedule_id': effect['id'],
                        'name': effect['observed_schedule']['name'],
                        'execution_may_have_started': effect['execution_may_have_started']}
        self.version = result.version
        # The protected claim and local checkpoint live in separate stores.
        # If revocation races pickup, retain the started marker as uncertain
        # and withhold the response; never automatically replay it.
        if not context_authority_active(context):
            raise CheckpointConflict('owner_desktop_pickup_unavailable')
        return response

    def acknowledge_desktop_draft(self, context: Any, token: str) -> None:
        """Record a client acknowledgement and conversation in one transaction.

        The acknowledgment does not prove a person read the response. The
        conversation store must share the claim database so the record cannot
        commit independently of the live-claim and checkpoint checks.
        """
        if self.telegram_origin:
            raise CheckpointConflict('owner_desktop_ack_unavailable')
        from alphonse.agent_v2.conversations import SQLiteConversationStore
        if not isinstance(token, str) or re.fullmatch(r'[0-9a-f]{32}', token) is None:
            raise ValueError('owner_desktop_ack_unavailable')
        conversation = getattr(context, 'conversation_store', None)
        try:
            if (type(conversation) is not SQLiteConversationStore
                    or not os.path.samefile(self.queue.db_path, conversation.db_path)):
                raise ValueError('owner_desktop_history_unavailable')
        except (AttributeError, OSError):
            raise ValueError('owner_desktop_history_unavailable') from None
        if verify_finalized_owner_ack(self.queue, self.task.message_id, token):
            return
        result = None
        def finish_claim(conn):
            # This finalizer runs under the same write lock, after the claim,
            # owner, exact ingress and immutable queue row were rechecked.
            if result is None or context.cancellation_checker() is not False:
                raise CheckpointConflict('owner_desktop_ack_unavailable')
            _, raw, state, budget = self._load(conn, result.version)
            marker = raw['metadata'].get('owner_read_desktop_pickup_started')
            has_outbox = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()
            if (state.status not in {'completed', 'waiting_user'} or state.inflight or budget.usage()['in_flight']
                    or budget.usage()['unknown_usage'] or not isinstance(marker, dict)
                    or marker.get('status') != 'client_acknowledged'
                    or 'owner_read_pending_decision' in raw['metadata']
                    or 'owner_read_pending_action' in raw['metadata']
                    or not has_outbox or conn.execute('SELECT 1 FROM v2_outbox WHERE task_id=? LIMIT 1',
                                                     (self.receipt.message_id,)).fetchone() is not None
                    or conn.execute('SELECT 1 FROM v2_scheduled_handoffs WHERE message_id=?',
                                    (self.receipt.message_id,)).fetchone() is not None):
                raise CheckpointConflict('owner_desktop_ack_unavailable')
            cursor = conn.execute("""DELETE FROM v2_inbound_messages WHERE message_id=?
                AND user_id=? AND status='processing' AND lease_owner=? AND claim_generation=?""",
                (self.receipt.message_id, self.receipt.user_id, self.receipt.lease_owner,
                 self.receipt.claim_generation))
            if cursor.rowcount != 1:
                raise CheckpointConflict('owner_desktop_ack_unavailable')
        with self._transaction(_finalize=finish_claim) as conn:
            row, raw, state, budget = self._load(conn, self.version)
            marker = raw['metadata'].get('owner_read_desktop_pickup_started')
            selected = raw['metadata'].get('owner_read_selected_draft')
            token_digest = hashlib.sha256(token.encode('ascii')).hexdigest()
            if (state.status not in {'completed', 'waiting_user'} or context.cancellation_checker() is not False
                    or not isinstance(marker, dict) or marker.get('status') not in {'started_unconfirmed', 'client_acknowledged'}
                    or marker.get('claim_digest') != _claim_digest(self.receipt)
                    or not isinstance(marker.get('acknowledgement_sha256'), str)
                    or not hmac.compare_digest(marker['acknowledgement_sha256'], token_digest)
                    or not isinstance(selected, dict) or marker.get('answer_sha256') != selected.get('answer_sha256')
                    or selected.get('kind') != ('answer' if state.status == 'completed' else 'question')
                    or marker.get('answer_sha256') != hashlib.sha256(state.response.encode('utf-8')).hexdigest()
                    or not isinstance(raw['metadata'].get('owner_read_observations'), dict)
                    or selected.get('evidence_bindings_sha256') != _decision_digest(raw['metadata']['owner_read_observations'])):
                raise CheckpointConflict('owner_desktop_ack_unavailable')
            if marker['status'] == 'client_acknowledged':
                return
            origin = conn.execute('''SELECT history_version,history_event_id FROM v2_trusted_owner_ingress
                WHERE message_id=? AND owner_id=?''', (self.task.message_id, self.task.user)).fetchone()
            if origin is None:
                raise CheckpointConflict('owner_desktop_history_unavailable')
            if origin['history_version'] == 1:
                # The claim transaction checks the version-bound event and its
                # full payload before and after this block. Never recreate it.
                incoming = conn.execute('SELECT 1 FROM v2_conversation_events WHERE event_id=?',
                                        (origin['history_event_id'],)).fetchone()
            elif origin['history_version'] == 0 and origin['history_event_id'] == '':
                # Requests created before history-at-ingress retain their old
                # acknowledgement behavior. Their origin uses the legacy digest.
                incoming = conversation.record(owner_user_id=self.task.user, role='user',
                    content=self.task.goal, source='desktop', source_message_id=f'inbound:{self.task.message_id}',
                    connection=conn)
            else:
                raise CheckpointConflict('owner_desktop_history_unavailable')
            from .owner_clock import display_response
            outgoing = conversation.record(owner_user_id=self.task.user, role='assistant',
                content=display_response(state.response, raw['metadata']['owner_read_observations']), source='desktop', source_message_id=f'outbound:owner:{self.task.message_id}',
                connection=conn)
            if incoming is None or outgoing is None:
                raise CheckpointConflict('owner_desktop_history_unavailable')
            marker['status'] = 'client_acknowledged'
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version

    def finalize_scheduled_result(self, context: Any) -> None:
        """Record a scheduled response; never claim goal success or delivery.

        The caller holds the cancellation handoff lock. The distinct scheduled
        claim fences provenance and due time; no owner IPC acknowledgement is
        fabricated. The event and occurrence transition share one transaction.
        """
        from datetime import datetime, timezone
        from alphonse.agent_v2.conversations import SQLiteConversationStore
        from .owner_read_recovery import restore_owner_read_observations
        from .owner_clock import display_response
        conversation = getattr(context, 'conversation_store', None)
        if (not self.scheduled_origin or type(conversation) is not SQLiteConversationStore
                or not os.path.samefile(self.queue.db_path, conversation.db_path)
                or context.cancellation_checker() is not False):
            raise CheckpointConflict('owner_schedule_result_unavailable')
        expected_version = self.version
        restore_owner_read_observations(self, context, expected_version=expected_version, allow_completed=True)
        result = None
        def finish_claim(conn):
            if result is None or context.cancellation_checker() is not False:
                raise CheckpointConflict('owner_schedule_result_unavailable')
            _, raw, state, budget = self._load(conn, result.version)
            if (state.status not in {'completed', 'waiting_user'} or state.inflight or budget.usage()['in_flight']
                    or budget.usage()['unknown_usage']
                    or raw['metadata'].get('owner_scheduled_result_recorded') != self.receipt.message_id):
                raise CheckpointConflict('owner_schedule_result_unavailable')
            now = datetime.now(timezone.utc).isoformat()
            occurrence_state = 'completed' if state.status == 'completed' else 'failed'
            schedule_state = 'completed' if state.status == 'completed' else 'failed'
            occurrence = conn.execute('''UPDATE v2_owner_schedule_occurrences SET state=?
                WHERE message_id=? AND owner_id=? AND state='queued' RETURNING schedule_id''',
                (occurrence_state, self.receipt.message_id, self.task.user)).fetchone()
            if occurrence is None:
                raise CheckpointConflict('owner_schedule_result_unavailable')
            updated = conn.execute('''UPDATE v2_scheduled_tasks SET status=?, next_run_at=NULL,
                last_run_at=?,updated_at=? WHERE scheduled_task_id=? AND owner_user_id=? AND status='active' ''',
                (schedule_state, now, now, occurrence['schedule_id'], self.task.user))
            deleted = conn.execute('''DELETE FROM v2_inbound_messages WHERE message_id=? AND user_id=?
                AND status='processing' AND lease_owner=? AND claim_generation=?''',
                (self.receipt.message_id, self.receipt.user_id, self.receipt.lease_owner,
                 self.receipt.claim_generation))
            if updated.rowcount != 1 or deleted.rowcount != 1:
                raise CheckpointConflict('owner_schedule_result_unavailable')
        with self._transaction(_finalize=finish_claim) as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            selected = raw['metadata'].get('owner_read_selected_draft')
            bindings = raw['metadata'].get('owner_read_observations')
            if (state.status not in {'completed', 'waiting_user'} or state.inflight or budget.usage()['in_flight']
                    or budget.usage()['unknown_usage'] or context.cancellation_checker() is not False
                    or not isinstance(selected, dict)
                    or selected.get('kind') != ('answer' if state.status == 'completed' else 'question')
                    or selected.get('claim_digest') != _claim_digest(self.receipt)
                    or selected.get('answer_sha256') != hashlib.sha256(state.response.encode('utf-8')).hexdigest()
                    or not isinstance(bindings, dict) or selected.get('evidence_refs') != state.evidence_sources
                    or selected.get('evidence_bindings_sha256') != _decision_digest(bindings)
                    or 'owner_read_pending_decision' in raw['metadata']
                    or 'owner_read_pending_action' in raw['metadata']):
                raise CheckpointConflict('owner_schedule_result_unavailable')
            # Recheck Gmail's DB-backed account/grant at the exact publication
            # transaction, after the separate source-refresh check above.
            from .owner_gmail import OWNER_GMAIL_TOOL_ID, valid_owner_gmail_locked
            for binding in bindings.values():
                if (binding.get('tool_id') == OWNER_GMAIL_TOOL_ID
                        and (state.status != 'completed' or not valid_owner_gmail_locked(
                            conn, self.task.user, binding.get('gmail_observation'), context=context))):
                    raise CheckpointConflict('owner_gmail_authority_unavailable')
            event = conversation.record(owner_user_id=self.task.user, role='assistant', source='scheduled',
                content=display_response(state.response, bindings),
                source_message_id='outbound:scheduled:' + self.task.message_id, connection=conn)
            if event is None:
                raise CheckpointConflict('owner_schedule_result_unavailable')
            from alphonse.agent_v2.owner_scheduled_delivery import enqueue_locked
            enqueue_locked(conn, owner_id=self.task.user, message_id=self.task.message_id,
                           event_id=event.event_id, kind=selected['kind'])
            raw['metadata']['owner_scheduled_result_recorded'] = self.receipt.message_id
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version

    def save_state(self, state: CandidateState, *, consume_decision: bool = False,
                   selected_decision: dict | None = None, projection_fingerprint: str = '') -> Checkpoint:
        """Save a non-I/O state change without changing evidence or accounting."""
        CandidateState.restore(state.snapshot())
        with self._transaction() as conn:
            row, raw, previous, budget = self._load(conn, self.version)
            if (previous.status != 'running' or state.task_id != previous.task_id or state.actor != previous.actor
                    or state.goal_source != previous.goal_source or state.evidence_sources != previous.evidence_sources
                    or state.inflight != previous.inflight or (state.status != 'running' and state.inflight)):
                raise CheckpointConflict('owner_atomic_state_transition_invalid')
            marker = raw['metadata'].get('owner_read_pending_decision')
            if state.status in {'completed', 'waiting_user'}:
                if budget.usage()['unknown_usage'] or not budget.within_work_limits():
                    raise CheckpointConflict('owner_atomic_model_budget_unavailable')
                expected_kind = 'answer' if state.status == 'completed' else 'question'
                if (not consume_decision or not isinstance(selected_decision, dict)
                        or selected_decision.get('kind') != expected_kind
                        or not isinstance(selected_decision.get('text'), str)
                        or selected_decision['text'] != state.response
                        or not _matches_decision(marker, self.receipt, selected_decision, projection_fingerprint)):
                    raise CheckpointConflict('owner_atomic_selected_decision_mismatch')
                observations = raw['metadata'].get('owner_read_observations')
                if (not isinstance(observations, dict) or set(observations) != set(state.evidence_sources)
                        or 'owner_read_selected_draft' in raw['metadata']):
                    raise CheckpointConflict('owner_atomic_observation_bindings_invalid')
                # This records provenance of an undelivered selection only.
                # It does not establish semantic support or current access to
                # external sources, and is never a publication capability.
                raw['metadata']['owner_read_selected_draft'] = {
                    'schema_version': 1, 'kind': expected_kind,
                    'decision_sha256': marker['decision_sha256'],
                    'answer_sha256': hashlib.sha256(state.response.encode('utf-8')).hexdigest(),
                    'projection_sha256': marker['projection_sha256'],
                    'model_reservation_id': marker['reservation_id'],
                    'claim_digest': marker['claim_digest'],
                    'evidence_refs': list(state.evidence_sources),
                    'evidence_bindings_sha256': _decision_digest(observations),
                }
            if consume_decision:
                raw['metadata'].pop('owner_steering_pending_projection', None)
                if marker is not None:
                    if not isinstance(marker, dict) or marker.get('claim_digest') != _claim_digest(self.receipt):
                        raise CheckpointConflict('owner_atomic_pending_decision_unavailable')
                    del raw['metadata']['owner_read_pending_decision']
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return result

    def pause_for_continuation(self, *, expected_version: int, metric: str, interval: int,
                               last_checkpoint: int, cancellation_checker,
                               progress_assessment: str = 'unknown') -> Checkpoint:
        """Fence a quiescent checkpoint. This does not release or resume a claim.

        The recorded counter comes from the durable ledger, not the caller.
        Delivery, authenticated continuation and a fresh-claim handoff are not
        implemented here; an active marker prevents further reservations.
        """
        if (metric not in {'actions', 'model_calls', 'input_tokens', 'output_tokens'}
                or type(interval) is not int or interval <= 0
                or type(last_checkpoint) is not int or last_checkpoint < 0
                or not callable(cancellation_checker)
                or progress_assessment not in {'progress', 'no_progress', 'unknown'}):
            raise ValueError('owner_pause_invalid')
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            md = raw['metadata']
            usage = budget.usage()
            current = usage['charged'][metric]
            if (cancellation_checker() is not False or state.status != 'running'
                    or state.inflight or usage['in_flight'] or usage['unknown_usage']
                    or current < last_checkpoint or current - last_checkpoint < interval
                    or any(key in md for key in ('owner_read_pending_action', 'owner_read_pending_decision',
                        'owner_read_selected_draft', 'owner_read_desktop_pickup_started',
                        'owner_read_ready_handoff', 'owner_bash_dispatch_started',
                        'owner_bash_return_unreconciled', 'owner_continuation_pause'))):
                raise CheckpointConflict('owner_pause_unavailable')
            observations = md.get('owner_read_observations')
            if not isinstance(observations, dict) or set(observations) != set(state.evidence_sources):
                raise CheckpointConflict('owner_pause_unavailable')
            paused_at = self.clock()
            if paused_at.tzinfo is None or paused_at.utcoffset() is None:
                raise CheckpointConflict('owner_pause_unavailable')
            md['owner_continuation_pause'] = {
                'schema_version': 2, 'claim_digest': _claim_digest(self.receipt),
                'metric': metric, 'interval': interval, 'counter': current,
                'previous_counter': last_checkpoint,
                'paused_at_ms': int(paused_at.timestamp() * 1000),
                'observations_sha256': _decision_digest(observations),
                'state_sha256': _decision_digest(state.snapshot()),
                'progress_assessment': progress_assessment,
                'evidence_refs': list(state.evidence_sources),
            }
            result = self.store._write(conn, row, raw, state, budget)
            if cancellation_checker() is not False:
                raise CheckpointConflict('owner_pause_unavailable')
        self.version = result.version
        return result

    def reserve(self, *, expected_version: int, reservation_id: str, kind: str,
                amounts: dict, now_ms: int, tool_id: str = '', arguments: dict | None = None,
                selected_decision: dict | None = None, projection_fingerprint: str = '') -> Checkpoint:
        if (not isinstance(reservation_id, str) or not reservation_id.strip()
                or kind not in {'model', 'tool'}
                or (kind == 'model' and (tool_id or arguments is not None))
                or (kind == 'tool' and (not isinstance(tool_id, str) or not tool_id.strip()
                                        or not isinstance(arguments, dict)))):
            raise ValueError('owner_atomic_reservation_invalid')
        rejection = None
        result = None
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            if state.status != 'running' or state.inflight or 'owner_read_pending_action' in raw['metadata']:
                raise CheckpointConflict('owner_atomic_inflight_requires_reconciliation')
            pending_decision = raw['metadata'].get('owner_read_pending_decision')
            if budget.usage()['unknown_usage']:
                raise CheckpointConflict('owner_atomic_unknown_usage')
            if kind == 'tool' and pending_decision is None:
                raise CheckpointConflict('owner_atomic_pending_decision_unresolved')
            if pending_decision is not None:
                if (kind != 'tool' or not isinstance(selected_decision, dict)
                        or selected_decision.get('kind') != 'tool'
                        or selected_decision.get('tool_id') != tool_id
                        or json.dumps(selected_decision.get('arguments'), sort_keys=True, allow_nan=False)
                           != json.dumps(arguments, sort_keys=True, allow_nan=False)
                        or not _matches_decision(pending_decision, self.receipt, selected_decision, projection_fingerprint)):
                    raise CheckpointConflict('owner_atomic_pending_decision_unresolved')
            old_revision = budget.revision
            try:
                budget.reserve(reservation_id, kind=kind, amounts=amounts, now_ms=now_ms)
            except BudgetExhausted as exc:
                rejection = exc
                if budget.revision != old_revision:
                    result = self.store._write(conn, row, raw, state, budget)
            else:
                state.inflight = reservation_id
                if pending_decision is not None:
                    del raw['metadata']['owner_read_pending_decision']
                if kind == 'model' and conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_steering'").fetchone():
                    steer_rows = conn.execute('''SELECT id,content,content_digest FROM v2_owner_steering
                        WHERE message_id=? AND owner_id=? ORDER BY sequence''', (self.task.message_id, self.task.user)).fetchall()
                    if steer_rows:
                        if not re.fullmatch(r'[0-9a-f]{64}', projection_fingerprint):
                            raise CheckpointConflict('owner_steering_projection_unavailable')
                        raw['metadata']['owner_steering_model_projection'] = {
                            'reservation_id': reservation_id, 'projection_sha256': projection_fingerprint,
                            'steering_sha256': _decision_digest([[item['id'], item['content_digest']] for item in steer_rows]),
                            'steering_ids': [item['id'] for item in steer_rows],
                        }
                raw['metadata']['owner_read_pending_action'] = {
                    'reservation_id': reservation_id, 'kind': kind,
                    'tool_id': tool_id, 'arguments': deepcopy(arguments), 'claim_digest': _claim_digest(self.receipt),
                }
                result = self.store._write(conn, row, raw, state, budget)
        if result is not None:
            self.version = result.version
        if rejection is not None:
            raise rejection
        assert result is not None
        return result

    def settle_model(self, *, expected_version: int, reservation_id: str, outcome: str,
                     input_tokens: int | None, output_tokens: int | None, decision: Any = None,
                     projection_fingerprint: str = '') -> Checkpoint:
        if outcome not in {'success', 'failure', 'timeout'}:
            raise ValueError('owner_atomic_outcome_invalid')
        decision_digest = _decision_digest(decision)
        if projection_fingerprint and (not isinstance(projection_fingerprint, str)
                                       or re.fullmatch(r'[0-9a-f]{64}', projection_fingerprint) is None):
            raise ValueError('owner_atomic_projection_invalid')
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            expected = {'reservation_id': reservation_id, 'kind': 'model', 'tool_id': '', 'arguments': None,
                        'claim_digest': _claim_digest(self.receipt)}
            if state.inflight != reservation_id or raw['metadata'].get('owner_read_pending_action') != expected:
                raise CheckpointConflict('owner_atomic_reservation_mismatch')
            budget.settle(reservation_id, outcome=outcome, input_tokens=input_tokens, output_tokens=output_tokens)
            state.inflight = ''
            del raw['metadata']['owner_read_pending_action']
            # A steer accepted while this model call was outstanding makes
            # its decision stale. Settlement still accounts for usage, but no
            # selected decision survives to authorize a tool or final answer.
            has_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_steering'").fetchone()
            steering = conn.execute('''SELECT * FROM v2_owner_steering WHERE message_id=? AND owner_id=?
                ORDER BY sequence''', (self.task.message_id, self.task.user)).fetchall() if has_table else []
            for item in steering:
                digest = hashlib.sha256((item['id'] + '\0' + item['content']).encode()).hexdigest()
                if (item['peer_uid'] != os.getuid() or not _steering_claim_in_lineage(raw['metadata'], self.receipt, item['claim_digest'])
                        or not hmac.compare_digest(item['content_digest'], digest)):
                    raise CheckpointConflict('owner_steering_binding_unavailable')
            stale = any(item['reservation_id'] == reservation_id and item['status'] == 'accepted' for item in steering)
            projection_marker = raw['metadata'].pop('owner_steering_model_projection', None)
            if projection_marker is None and any(item['reservation_id'] != reservation_id for item in steering):
                raise CheckpointConflict('owner_steering_projection_unavailable')
            if projection_marker is not None:
                if not isinstance(projection_marker, dict) or not isinstance(projection_marker.get('steering_ids'), list):
                    raise CheckpointConflict('owner_steering_projection_unavailable')
                included = [item for item in steering if item['id'] in projection_marker.get('steering_ids', [])]
                if (set(projection_marker) != {'reservation_id', 'projection_sha256', 'steering_sha256', 'steering_ids'}
                        or projection_marker['reservation_id'] != reservation_id
                        or projection_marker['projection_sha256'] != projection_fingerprint
                        or projection_marker['steering_ids'] != [item['id'] for item in included]
                        or projection_marker['steering_ids'] != [item['id'] for item in steering if item['reservation_id'] != reservation_id]
                        or projection_marker['steering_sha256'] != _decision_digest([[item['id'], item['content_digest']] for item in included])):
                    raise CheckpointConflict('owner_steering_projection_unavailable')
            if not stale:
                raw['metadata']['owner_read_pending_decision'] = {
                    'reservation_id': reservation_id, 'claim_digest': _claim_digest(self.receipt),
                    'decision_sha256': decision_digest, 'projection_sha256': projection_fingerprint,
                }
                if projection_marker is not None:
                    raw['metadata']['owner_steering_pending_projection'] = projection_marker
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        self.steering_decision_stale = stale
        return result

    def settle_read(self, *, expected_version: int, reservation_id: str, tool_id: str,
                    arguments: dict, evidence_ref: str, observation: dict,
                    knowledge_snapshot_digest: str | None = None, cancellation_checker=None) -> Checkpoint:
        """Store an already-validated read result, action binding, and budget together.

        An explicit bounded native web read failure is settled as a failure;
        other failed or ambiguous actions remain unsettled. Observation validity
        and actor visibility are established and revalidated by the host.
        """
        if tool_id == 'native.bash':
            # Generic shell may have changed state even when it exits cleanly.
            # It must never be settled using the read-only observation path.
            raise ValueError('owner_atomic_shell_not_read')
        if (not isinstance(evidence_ref, str) or not evidence_ref.strip()
                or not isinstance(observation, dict) or not isinstance(arguments, dict)):
            raise ValueError('owner_atomic_observation_invalid')
        # Reject unserializable or oversized content before opening the lock.
        serialized = json.dumps(observation, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
        if len(serialized.encode('utf-8')) > 262144:
            raise ValueError('owner_atomic_observation_too_large')
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            expected = {'reservation_id': reservation_id, 'kind': 'tool', 'tool_id': tool_id, 'arguments': arguments,
                        'claim_digest': _claim_digest(self.receipt)}
            observations = raw['metadata'].get('owner_read_observations')
            if (state.inflight != reservation_id or json.dumps(raw['metadata'].get('owner_read_pending_action'), sort_keys=True) != json.dumps(expected, sort_keys=True)
                    or not isinstance(observations, dict) or evidence_ref == state.goal_source
                    or evidence_ref in observations or evidence_ref in state.evidence_sources):
                raise CheckpointConflict('owner_atomic_observation_binding_mismatch')
            from .owner_knowledge_snapshot import (OWNER_KNOWLEDGE_TOOL_ID, owner_knowledge_projection,
                                                   snapshot_owner_knowledge_locked)
            if tool_id == OWNER_KNOWLEDGE_TOOL_ID:
                if arguments != {} or not callable(cancellation_checker):
                    raise CheckpointConflict('owner_knowledge_binding_unavailable')
                snapshot = snapshot_owner_knowledge_locked(conn, self.task, cancellation_checker=cancellation_checker,
                                                          clock=self.clock)
                if (snapshot.status != 'available' or not isinstance(knowledge_snapshot_digest, str)
                        or not hmac.compare_digest(snapshot.digest, knowledge_snapshot_digest)
                        or owner_knowledge_projection(snapshot) != observation):
                    raise CheckpointConflict('owner_knowledge_binding_unavailable')
            elif knowledge_snapshot_digest is not None:
                raise CheckpointConflict('owner_knowledge_binding_unavailable')
            from .owner_web import WEB_TOOL_IDS, is_web_failure_observation
            failed_web_read = is_web_failure_observation(tool_id, observation)
            if observation.get('status') == 'failed' and not failed_web_read:
                raise CheckpointConflict('owner_atomic_failed_observation_invalid')
            budget.settle(reservation_id, outcome='failure' if failed_web_read else 'success')
            observations[evidence_ref] = {'actor': state.actor, 'tool_id': tool_id,
                                          'arguments': deepcopy(arguments),
                                          'observation_sha256': observation_digest(observation)}
            if tool_id == OWNER_KNOWLEDGE_TOOL_ID:
                observations[evidence_ref]['knowledge_snapshot_digest'] = knowledge_snapshot_digest
            from .owner_clock import OWNER_CLOCK_TOOL_ID
            if tool_id == OWNER_CLOCK_TOOL_ID:
                observations[evidence_ref]['clock_observation'] = deepcopy(observation)
            from .owner_gmail import OWNER_GMAIL_TOOL_ID
            if tool_id == OWNER_GMAIL_TOOL_ID:
                observations[evidence_ref]['gmail_observation'] = deepcopy(observation)
            from .owner_calendar import OWNER_CALENDAR_TOOL_ID
            if tool_id == OWNER_CALENDAR_TOOL_ID:
                observations[evidence_ref]['calendar_observation'] = deepcopy(observation)
            from .owner_gmail_search import OWNER_GMAIL_SEARCH_TOOL_ID
            if tool_id == OWNER_GMAIL_SEARCH_TOOL_ID:
                observations[evidence_ref]['gmail_search_observation'] = deepcopy(observation)
            if tool_id in WEB_TOOL_IDS:
                observations[evidence_ref]['web_observation'] = deepcopy(observation)
            state.evidence_sources.append(evidence_ref)
            state.inflight = ''
            del raw['metadata']['owner_read_pending_action']
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return result

    def settle_dojo_bash(self, *, expected_version: int, reservation_id: str,
                         arguments: dict, evidence_ref: str, observation: dict,
                         session_id: str, context: Any) -> Checkpoint:
        """Persist a disposable fixture observation, never a host-effect receipt.

        Only the in-process Dojo session may consume this record. An interrupted
        call remains in flight; recovery cannot recreate or replay its container.
        """
        from .dojo_bash_guard import valid_dojo_observation, valid_dojo_session
        if (not valid_dojo_session(context, session_id) or not valid_dojo_observation(observation)
                or not isinstance(arguments, dict) or not isinstance(observation, dict)
                or not isinstance(evidence_ref, str) or not evidence_ref.strip()
                or len(json.dumps(observation, ensure_ascii=False).encode()) > 100000):
            raise ValueError('owner_atomic_dojo_bash_invalid')
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            expected = {'reservation_id': reservation_id, 'kind': 'tool', 'tool_id': 'native.bash',
                        'arguments': arguments, 'claim_digest': _claim_digest(self.receipt)}
            observations = raw['metadata'].get('owner_read_observations')
            if (state.inflight != reservation_id or raw['metadata'].get('owner_read_pending_action') != expected
                    or not isinstance(observations, dict) or evidence_ref == state.goal_source
                    or evidence_ref in observations or evidence_ref in state.evidence_sources):
                raise CheckpointConflict('owner_atomic_dojo_bash_unavailable')
            budget.settle(reservation_id, outcome='success')
            observations[evidence_ref] = {'actor': state.actor, 'tool_id': 'native.bash',
                'arguments': deepcopy(arguments), 'observation_sha256': observation_digest(observation),
                'dojo_bash_observation': deepcopy(observation), 'dojo_session_id': session_id}
            state.evidence_sources.append(evidence_ref)
            state.inflight = ''
            del raw['metadata']['owner_read_pending_action']
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return result

    def mark_host_bash_dispatch(self, *, expected_version: int, reservation_id: str, arguments: dict) -> Checkpoint:
        """Record a single possible host-shell dispatch before starting it.

        A crash after this marker leaves an uncertain action, even if the
        process never started. The marker is not permission to execute: the
        caller must separately validate the actor, live claim and scope.
        """
        if not isinstance(reservation_id, str) or not reservation_id.strip() or not isinstance(arguments, dict):
            raise ValueError('owner_atomic_bash_dispatch_invalid')
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            metadata = raw['metadata']
            expected = {'reservation_id': reservation_id, 'kind': 'tool', 'tool_id': 'native.bash',
                        'arguments': arguments, 'claim_digest': _claim_digest(self.receipt)}
            if (state.status != 'running' or state.inflight != reservation_id
                    or reservation_id not in budget.usage()['in_flight']
                    or metadata.get('owner_read_pending_action') != expected
                    or 'owner_bash_dispatch_started' in metadata
                    or 'owner_bash_return_unreconciled' in metadata):
                raise CheckpointConflict('owner_atomic_bash_dispatch_unavailable')
            metadata['owner_bash_dispatch_started'] = {
                'schema_version': 1, 'reservation_id': reservation_id,
                'claim_digest': _claim_digest(self.receipt), 'arguments_sha256': _decision_digest(arguments),
            }
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return result

    def record_bash_return(self, *, expected_version: int, reservation_id: str,
                           arguments: dict, exit_code: int, timed_out: bool,
                           cancelled: bool, stdout: str, stderr: str,
                           output_bindings=None) -> Checkpoint:
        """Record bounded, caller-reported shell facts without settling effects.

        This unwired foundation never authorizes or dispatches a command and a
        process exit is not proof of an authorized effect. Only digests and
        lengths of output are stored by default. A trusted executor may opt in
        to storing output after redaction with its exact ExecutionBindings;
        the owner's existing checkpoint database then contains private output.
        Only a claim- and authorization-checked caller may read it. The action
        and reservation remain in
        flight, so restart cannot replay it or publish a successful outcome.
        A later actor-authorized reconciliation contract must determine the
        effects and settle it. The caller's report is not independent evidence.
        """
        if (not isinstance(reservation_id, str) or not reservation_id.strip()
                or not isinstance(arguments, dict) or type(exit_code) is not int
                or type(timed_out) is not bool or type(cancelled) is not bool
                or (timed_out and cancelled) or ((timed_out or cancelled) and exit_code != -1)
                or not isinstance(stdout, str) or not isinstance(stderr, str)
                # The executor caps characters, not UTF-8 bytes, and appends a
                # truncation marker. Preserve bounded multibyte output receipts.
                or len(stdout.encode('utf-8')) > 65536 or len(stderr.encode('utf-8')) > 65536):
            raise ValueError('owner_atomic_bash_return_invalid')
        redacted_output = None
        if output_bindings is not None:
            from alphonse.agent_v2.skill_bindings import ExecutionBindings
            if type(output_bindings) is not ExecutionBindings:
                raise ValueError('owner_atomic_bash_output_redactor_invalid')
            # The executor must supply the bindings used for the action, not a
            # model-supplied redactor. Redaction can expand strings; bound both
            # stored strings again before touching the database.
            redacted_output = output_bindings.redact({'stdout': stdout, 'stderr': stderr})
            if (not isinstance(redacted_output, dict) or set(redacted_output) != {'stdout', 'stderr'}
                    or any(not isinstance(value, str) or len(value.encode('utf-8')) > 65536
                           for value in redacted_output.values())):
                raise ValueError('owner_atomic_bash_output_invalid')
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            pending = {'reservation_id': reservation_id, 'kind': 'tool', 'tool_id': 'native.bash',
                       'arguments': arguments, 'claim_digest': _claim_digest(self.receipt)}
            if (state.status != 'running' or state.inflight != reservation_id
                    or raw['metadata'].get('owner_read_pending_action') != pending
                    or 'owner_bash_return_unreconciled' in raw['metadata']):
                raise CheckpointConflict('owner_atomic_bash_return_unavailable')
            raw['metadata']['owner_bash_return_unreconciled'] = {
                'schema_version': 1, 'reservation_id': reservation_id,
                'claim_digest': _claim_digest(self.receipt), 'arguments_sha256': _decision_digest(arguments),
                'exit_code': exit_code, 'timed_out': timed_out, 'cancelled': cancelled,
                'stdout_sha256': hashlib.sha256(stdout.encode('utf-8')).hexdigest(),
                'stderr_sha256': hashlib.sha256(stderr.encode('utf-8')).hexdigest(),
                'stdout_bytes': len(stdout.encode('utf-8')), 'stderr_bytes': len(stderr.encode('utf-8')),
                'effect_status': 'unverified',
            }
            if redacted_output is not None:
                raw['metadata']['owner_bash_return_unreconciled']['redacted_output'] = redacted_output
                raw['metadata']['owner_bash_return_unreconciled']['redacted_output_sha256'] = _decision_digest(redacted_output)
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return result

    def read_bash_return(self, *, expected_version: int, reservation_id: str, arguments: dict,
                         authorize, cancellation_checker) -> dict:
        """Read retained process facts under the same live claim, without settling.

        `authorize` is a trusted host callback that must revalidate the actor's
        access to the command's sources and any selected bindings. This is a
        point-in-time check, not proof that arbitrary effects were authorized or
        verified, nor a replacement-claim recovery or effect-reconciliation API.
        """
        if (not isinstance(reservation_id, str) or not reservation_id.strip()
                or not isinstance(arguments, dict) or not callable(authorize)
                or not callable(cancellation_checker)):
            raise ValueError('owner_atomic_bash_read_invalid')
        def permitted():
            try:
                return (cancellation_checker() is False
                        and authorize(self.task.user, self.task.task_id, deepcopy(arguments)) is True)
            except Exception:
                return False
        if not permitted():
            raise CheckpointConflict('owner_atomic_bash_read_unavailable')
        with self._transaction() as conn:
            _, raw, state, budget = self._load(conn, expected_version)
            metadata = raw['metadata']
            marker = metadata.get('owner_bash_return_unreconciled')
            pending = {'reservation_id': reservation_id, 'kind': 'tool', 'tool_id': 'native.bash',
                       'arguments': arguments, 'claim_digest': _claim_digest(self.receipt)}
            if (state.status != 'running' or state.inflight != reservation_id
                    or reservation_id not in budget.usage()['in_flight']
                    or metadata.get('owner_read_pending_action') != pending
                    or not isinstance(marker, dict) or marker.get('schema_version') != 1
                    or marker.get('reservation_id') != reservation_id
                    or marker.get('claim_digest') != _claim_digest(self.receipt)
                    or marker.get('arguments_sha256') != _decision_digest(arguments)
                    or marker.get('effect_status') != 'unverified'):
                raise CheckpointConflict('owner_atomic_bash_read_unavailable')
            output = marker.get('redacted_output')
            if (not isinstance(output, dict) or set(output) != {'stdout', 'stderr'}
                    or any(not isinstance(value, str) or len(value.encode('utf-8')) > 65536
                           for value in output.values())
                    or marker.get('redacted_output_sha256') != _decision_digest(output)
                    or type(marker.get('exit_code')) is not int
                    or type(marker.get('timed_out')) is not bool or type(marker.get('cancelled')) is not bool):
                raise CheckpointConflict('owner_atomic_bash_read_unavailable')
            result = {'exit_code': marker['exit_code'], 'timed_out': marker['timed_out'],
                      'cancelled': marker['cancelled'], 'effect_status': 'unverified', **deepcopy(output)}
            marker_digest = _decision_digest(marker)
        if not permitted():
            raise CheckpointConflict('owner_atomic_bash_read_unavailable')
        # The authorization callback can perform external I/O. A live claim at
        # the first read is not sufficient if it changes while that callback
        # runs. Bracket it with a second fenced snapshot and compare the exact
        # action and receipt before returning private output.
        with self._transaction() as conn:
            _, raw, state, budget = self._load(conn, expected_version)
            marker = raw['metadata'].get('owner_bash_return_unreconciled')
            if (state.status != 'running' or state.inflight != reservation_id
                    or reservation_id not in budget.usage()['in_flight']
                    or raw['metadata'].get('owner_read_pending_action') != pending
                    or not isinstance(marker, dict) or _decision_digest(marker) != marker_digest):
                raise CheckpointConflict('owner_atomic_bash_read_unavailable')
        try:
            cancelled = cancellation_checker() is not False
        except Exception:
            cancelled = True
        if cancelled:
            raise CheckpointConflict('owner_atomic_bash_read_unavailable')
        return result

    def settle_host_bash_return(self, *, expected_version: int, reservation_id: str,
                                arguments: dict, evidence_ref: str, authorize,
                                cancellation_checker) -> Checkpoint:
        """Settle the process return, leaving its effects explicitly unverified.

        This opt-in foundation does not authorize dispatch or publication. A
        caller must enforce access to the historical output each time it is
        used, and independently verify any claimed effect. The host must not
        automatically replay a started action after an interruption.
        """
        if not isinstance(evidence_ref, str) or not evidence_ref.strip():
            raise ValueError('owner_atomic_bash_observation_invalid')
        observation = self.read_bash_return(expected_version=expected_version,
            reservation_id=reservation_id, arguments=arguments, authorize=authorize,
            cancellation_checker=cancellation_checker)
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            metadata = raw['metadata']
            pending = {'reservation_id': reservation_id, 'kind': 'tool', 'tool_id': 'native.bash',
                       'arguments': arguments, 'claim_digest': _claim_digest(self.receipt)}
            marker = metadata.get('owner_bash_return_unreconciled')
            dispatch = metadata.get('owner_bash_dispatch_started')
            observations = metadata.get('owner_read_observations')
            history = metadata.get('owner_host_bash_history', [])
            if (state.status != 'running' or state.inflight != reservation_id
                    or reservation_id not in budget.usage()['in_flight']
                    or metadata.get('owner_read_pending_action') != pending
                    or not isinstance(marker, dict) or not isinstance(dispatch, dict)
                    or dispatch != {'schema_version': 1, 'reservation_id': reservation_id,
                                    'claim_digest': _claim_digest(self.receipt),
                                    'arguments_sha256': _decision_digest(arguments)}
                    or marker.get('reservation_id') != reservation_id
                    or marker.get('claim_digest') != _claim_digest(self.receipt)
                    or marker.get('arguments_sha256') != _decision_digest(arguments)
                    or marker.get('effect_status') != 'unverified'
                    or any(type(marker.get(key)) is not type(observation[key]) or marker[key] != observation[key]
                           for key in ('exit_code', 'timed_out', 'cancelled'))
                    or marker.get('redacted_output_sha256') != _decision_digest(
                        {'stdout': observation['stdout'], 'stderr': observation['stderr']})
                    or not isinstance(observations, dict) or not isinstance(history, list)
                    or evidence_ref == state.goal_source or evidence_ref in observations
                    or evidence_ref in state.evidence_sources
                    or any(item.get('reservation_id') == reservation_id for item in history if isinstance(item, dict))):
                raise CheckpointConflict('owner_atomic_bash_observation_unavailable')
            budget.settle(reservation_id, outcome='success')  # the process returned, not task success
            observations[evidence_ref] = {'actor': state.actor, 'tool_id': 'native.bash',
                'arguments': deepcopy(arguments), 'observation_sha256': observation_digest(observation),
                'host_bash_observation': deepcopy(observation)}
            history.append({'reservation_id': reservation_id, 'claim_digest': _claim_digest(self.receipt),
                            'arguments_sha256': _decision_digest(arguments),
                            'observation_sha256': observation_digest(observation),
                            'effect_status': 'unverified'})
            metadata['owner_host_bash_history'] = history
            state.evidence_sources.append(evidence_ref)
            state.inflight = ''
            del metadata['owner_read_pending_action']
            del metadata['owner_bash_dispatch_started']
            del metadata['owner_bash_return_unreconciled']
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return result

    def commit_owner_reminder(self, *, context, decision, projection_fingerprint, now_ms):
        """Atomically create one local reminder and select its truthful confirmation.

        The caller must serialize this entire operation with owner cancellation.
        The database transaction fences the live claim, selected decision,
        reminder insertion, budget settlement and terminal checkpoint together.
        """
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        from .owner_clock import OWNER_CLOCK_TOOL_ID, valid_clock_observation
        from .owner_reminder_create import OWNER_REMINDER_CREATE_TOOL_ID, validate_reminder_request
        from .prerequisite import MissingPrerequisite
        from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders

        if (context.messages is not self.queue or context.claim_receipt is not self.receipt
                or context.is_cancelled() is not False or not isinstance(decision, dict)
                or decision.get('kind') != 'tool' or decision.get('tool_id') != OWNER_REMINDER_CREATE_TOOL_ID
                or not isinstance(decision.get('arguments'), dict)):
            raise CheckpointConflict('owner_reminder_unavailable')
        args = decision['arguments']
        # Canonical user timezone is stored alongside the queue; a separate
        # external settings store cannot be fenced by this transaction.
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, self.version)
            marker = raw['metadata'].get('owner_read_pending_decision')
            observations = raw['metadata'].get('owner_read_observations')
            if (state.status != 'running' or state.inflight or context.is_cancelled() is not False
                    or not _matches_decision(marker, self.receipt, decision, projection_fingerprint)
                    or not isinstance(observations, dict) or set(observations) != set(state.evidence_sources)):
                raise CheckpointConflict('owner_reminder_unavailable')
            clock_bindings = [item for item in observations.values() if item.get('tool_id') == OWNER_CLOCK_TOOL_ID]
            execution = context.tool_execution_context(self.task)
            if not clock_bindings:
                # Nothing has been mutated: let the loop request the missing
                # observation and retry under the same claim and cancellation fence.
                raise MissingPrerequisite('owner_clock')
            if (not valid_clock_observation(self.task.user, {},
                    clock_bindings[-1].get('clock_observation'), context=execution)):
                raise CheckpointConflict('owner_reminder_clock_required')
            zone_row = conn.execute('SELECT timezone FROM v2_user_timezones WHERE user_id=?', (self.task.user,)).fetchone()
            setting = conn.execute("SELECT value FROM v2_user_settings WHERE key='timezone'").fetchone()
            zone = str((zone_row[0] if zone_row and zone_row[0] else None) or (setting[0] if setting else None) or 'UTC')
            if zone != clock_bindings[-1]['clock_observation']['timezone']:
                raise CheckpointConflict('owner_reminder_clock_required')
            if self.task.metadata.get('owner_clarification') is not None:
                # Preserve the distinct actor-bound turns and recheck their
                # provenance in the same transaction as any future effect.
                # This is not yet authority to interpret a follow-up as a
                # reminder: the existing validator remains in force.
                from .owner_clarification import load_bound_owner_exchange
                load_bound_owner_exchange(conn, self.task)
            canonical = validate_reminder_request(self.task, args, zone,
                observed_at=clock_bindings[-1]['clock_observation']['observed_at_utc'])
            now = datetime.now(timezone.utc)
            due = datetime.fromisoformat(canonical['due_at'])
            # Time spent interpreting the request counts against a relative
            # duration. Never save an already-expired reminder, but do not
            # reject a future one merely because less than a minute remains.
            if due <= now:
                raise ValueError('owner_reminder_time_expired')
            reservation = 'reminder-' + secrets.token_hex(16)
            budget.reserve(reservation, kind='tool', amounts={'actions': 1, 'model_calls': 0,
                'input_tokens': 0, 'output_tokens': 0}, now_ms=now_ms)
            OwnerDesktopReminders.ensure_table(conn)
            existing = conn.execute('SELECT 1 FROM owner_desktop_reminders WHERE owner_id=? AND request_id=?',
                                    (self.task.user, self.task.message_id)).fetchone()
            if existing is not None:
                raise CheckpointConflict('owner_reminder_request_already_exists')
            effect = OwnerDesktopReminders.create_locked(conn, self.task.user, canonical['message'],
                canonical['due_at'], self.task.message_id, now=now)
            if effect['state'] != 'pending' or context.is_cancelled() is not False:
                raise CheckpointConflict('owner_reminder_unavailable')
            observation = {'id': effect['id'], 'message': canonical['message'],
                           'due_at': effect['due_at'], 'state': effect['state']}
            ref = 'owner-reminder:' + effect['id']
            observations[ref] = {'actor': state.actor, 'tool_id': OWNER_REMINDER_CREATE_TOOL_ID,
                'arguments': deepcopy(args), 'observation_sha256': observation_digest(observation),
                'reminder_observation': deepcopy(observation)}
            state.evidence_sources.append(ref)
            local_due = due.astimezone(ZoneInfo(zone)).isoformat()
            state.status = 'completed'
            state.response = (f"Reminder saved: {canonical['message']} — {local_due} ({zone}). "
                              'Reminders appear when Desktop is open.')
            budget.settle(reservation, outcome='success')
            del raw['metadata']['owner_read_pending_decision']
            raw['metadata']['owner_read_selected_draft'] = {
                'schema_version': 1, 'kind': 'answer', 'decision_sha256': marker['decision_sha256'],
                'answer_sha256': hashlib.sha256(state.response.encode('utf-8')).hexdigest(),
                'projection_sha256': marker['projection_sha256'], 'model_reservation_id': marker['reservation_id'],
                'claim_digest': marker['claim_digest'], 'evidence_refs': list(state.evidence_sources),
                'evidence_bindings_sha256': _decision_digest(observations),
            }
            # The effect and its recoverable confirmation become durable together.
            # This transaction runs under the cancellation handoff lock.
            raw['metadata']['owner_read_ready_handoff'] = {
                'schema_version': 1, 'claim_digest': _claim_digest(self.receipt),
                'selected_sha256': _decision_digest(raw['metadata']['owner_read_selected_draft']),
            }
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return state

    def commit_owner_schedule_cancel(self, *, context, decision, projection_fingerprint, now_ms, observed):
        """Cancel a single observed owner schedule and checkpoint its result atomically."""
        from .owner_schedule_create import SCHEDULED_TASK_TOOL_ID
        from .owner_schedule_cancel import cancel_schedule_locked, valid_cancel_locked
        from .owner_schedules import OWNER_SCHEDULES_TOOL_ID
        if (self.scheduled_origin or context.messages is not self.queue or context.claim_receipt is not self.receipt
                or context.is_cancelled() is not False or not isinstance(decision, dict)
                or decision.get('kind') != 'tool' or decision.get('tool_id') != SCHEDULED_TASK_TOOL_ID
                or not isinstance(decision.get('arguments'), dict)
                or not isinstance(observed, tuple) or len(observed) != 2):
            raise CheckpointConflict('owner_schedule_cancel_unavailable')
        args = decision['arguments']
        evidence_ref, snapshot = observed
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, self.version)
            marker = raw['metadata'].get('owner_read_pending_decision')
            observations = raw['metadata'].get('owner_read_observations')
            binding = observations.get(evidence_ref) if isinstance(observations, dict) else None
            if (state.status != 'running' or state.inflight or context.is_cancelled() is not False
                    or not _matches_decision(marker, self.receipt, decision, projection_fingerprint)
                    or not isinstance(observations, dict) or set(observations) != set(state.evidence_sources)
                    or not isinstance(binding, dict) or binding.get('actor') != state.actor
                    or binding.get('tool_id') != OWNER_SCHEDULES_TOOL_ID or binding.get('arguments') != {}
                    or binding.get('observation_sha256') != observation_digest(snapshot)):
                raise CheckpointConflict('owner_schedule_cancel_unavailable')
            reservation = 'schedule-cancel-' + secrets.token_hex(16)
            budget.reserve(reservation, kind='tool', amounts={'actions': 1, 'model_calls': 0,
                'input_tokens': 0, 'output_tokens': 0}, now_ms=now_ms)
            effect = cancel_schedule_locked(conn, self.task, args, snapshot)
            if context.is_cancelled() is not False or not valid_cancel_locked(conn, state.actor, args, effect, task=self.task):
                raise CheckpointConflict('owner_schedule_cancel_unavailable')
            # A pre-cancellation list is necessarily stale after mutation. Its
            # digest and selected target are durably bound into the effect.
            for ref in list(state.evidence_sources):
                if observations[ref].get('tool_id') == OWNER_SCHEDULES_TOOL_ID:
                    state.evidence_sources.remove(ref)
                    del observations[ref]
            ref = 'owner-schedule-cancel:' + effect['id']
            observations[ref] = {'actor': state.actor, 'tool_id': SCHEDULED_TASK_TOOL_ID,
                'arguments': deepcopy(args), 'observation_sha256': observation_digest(effect),
                'schedule_observation': deepcopy(effect)}
            state.evidence_sources.append(ref)
            budget.settle(reservation, outcome='success')
            del raw['metadata']['owner_read_pending_decision']
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return state, ref, effect

    def commit_owner_schedule(self, *, context, decision, projection_fingerprint, now_ms):
        """Commit one owner-origin schedule, provenance and response together.

        This method is dormant until a trusted scheduled execution path exists.
        """
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        from .owner_clock import OWNER_CLOCK_TOOL_ID, execute_owner_clock, valid_clock_observation
        from .prerequisite import MissingPrerequisite
        from .owner_schedule_create import SCHEDULED_TASK_TOOL_ID, validate_schedule_request, valid_schedule_locked
        from .owner_schedule_origin import create_schedule_origin_locked

        if (context.messages is not self.queue or context.claim_receipt is not self.receipt
                or context.is_cancelled() is not False or not isinstance(decision, dict)
                or decision.get('kind') != 'tool' or decision.get('tool_id') != SCHEDULED_TASK_TOOL_ID
                or not isinstance(decision.get('arguments'), dict)):
            raise CheckpointConflict('owner_schedule_unavailable')
        args = decision['arguments']
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, self.version)
            marker = raw['metadata'].get('owner_read_pending_decision')
            observations = raw['metadata'].get('owner_read_observations')
            if (state.status != 'running' or state.inflight or context.is_cancelled() is not False
                    or not _matches_decision(marker, self.receipt, decision, projection_fingerprint)
                    or not isinstance(observations, dict) or set(observations) != set(state.evidence_sources)):
                raise CheckpointConflict('owner_schedule_unavailable')
            clocks = [item for item in observations.values() if item.get('tool_id') == OWNER_CLOCK_TOOL_ID]
            execution = context.tool_execution_context(self.task)
            zone_row = conn.execute('SELECT timezone FROM v2_user_timezones WHERE user_id=?', (self.task.user,)).fetchone()
            setting = conn.execute("SELECT value FROM v2_user_settings WHERE key='timezone'").fetchone()
            zone = str((zone_row[0] if zone_row and zone_row[0] else None) or (setting[0] if setting else None) or 'UTC')
            if (not clocks or not valid_clock_observation(self.task.user, {},
                    clocks[-1].get('clock_observation'), context=execution)
                    or zone != clocks[-1]['clock_observation']['timezone']):
                # The host owns this read-only prerequisite. It is recorded
                # as evidence, budgeted, and committed with the effect; no
                # model-chosen action or owner permission is fabricated.
                clock_reservation = 'prerequisite-clock-' + secrets.token_hex(16)
                budget.reserve(clock_reservation, kind='tool', amounts={'actions': 1, 'model_calls': 0,
                    'input_tokens': 0, 'output_tokens': 0}, now_ms=now_ms)
                clock = execute_owner_clock({}, context=execution)
                if (zone != clock.get('timezone') or not valid_clock_observation(self.task.user, {}, clock, context=execution)
                        or context.is_cancelled() is not False):
                    raise CheckpointConflict('owner_schedule_clock_required')
                clock_ref = 'owner-clock:' + secrets.token_hex(16)
                observations[clock_ref] = {'actor': state.actor, 'tool_id': OWNER_CLOCK_TOOL_ID,
                    'arguments': {}, 'observation_sha256': observation_digest(clock), 'clock_observation': deepcopy(clock)}
                state.evidence_sources.append(clock_ref)
                budget.settle(clock_reservation, outcome='success')
                clocks = [observations[clock_ref]]
            now = datetime.now(timezone.utc)
            try:
                due = validate_schedule_request(self.task, args, zone,
                    observed_at=clocks[-1]['clock_observation']['observed_at_utc'], now=now)
            except ValueError as exc:
                if str(exc) == 'owner_schedule_time_invalid':
                    raise MissingPrerequisite('owner_schedule_time') from None
                raise
            reservation = 'schedule-' + secrets.token_hex(16)
            budget.reserve(reservation, kind='tool', amounts={'actions': 1, 'model_calls': 0,
                'input_tokens': 0, 'output_tokens': 0}, now_ms=now_ms)
            effect = context.owner_schedule_store.create_task(owner_user_id=self.task.user,
                name=self.task.goal[:100], prompt=self.task.goal, schedule_kind='once', run_at=due,
                timezone_name=zone, delivery_mode='pdca', idempotency_key='owner-schedule:' + self.task.message_id,
                now=now, connection=conn)
            create_schedule_origin_locked(conn, self.task, effect)
            observation = {'id': effect.scheduled_task_id, 'run_at': due, 'prompt': self.task.goal}
            if context.is_cancelled() is not False or not valid_schedule_locked(conn, state.actor, args, observation, task=self.task):
                raise CheckpointConflict('owner_schedule_unavailable')
            ref = 'owner-schedule:' + effect.scheduled_task_id
            observations[ref] = {'actor': state.actor, 'tool_id': SCHEDULED_TASK_TOOL_ID,
                'arguments': deepcopy(args), 'observation_sha256': observation_digest(observation),
                'schedule_observation': deepcopy(observation)}
            state.evidence_sources.append(ref)
            state.status = 'completed'
            state.response = f"Task scheduled for {datetime.fromisoformat(due).astimezone(ZoneInfo(zone)).isoformat()} ({zone})."
            budget.settle(reservation, outcome='success')
            del raw['metadata']['owner_read_pending_decision']
            raw['metadata']['owner_read_selected_draft'] = {
                'schema_version': 1, 'kind': 'answer', 'decision_sha256': marker['decision_sha256'],
                'answer_sha256': hashlib.sha256(state.response.encode('utf-8')).hexdigest(),
                'projection_sha256': marker['projection_sha256'], 'model_reservation_id': marker['reservation_id'],
                'claim_digest': marker['claim_digest'], 'evidence_refs': list(state.evidence_sources),
                'evidence_bindings_sha256': _decision_digest(observations),
            }
            raw['metadata']['owner_read_ready_handoff'] = {
                'schema_version': 1, 'claim_digest': _claim_digest(self.receipt),
                'selected_sha256': _decision_digest(raw['metadata']['owner_read_selected_draft']),
            }
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return state
