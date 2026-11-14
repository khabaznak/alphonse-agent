"""Owner-claim-bound candidate checkpoints in the queue's SQLite database.

This is an experimental persistence boundary, not a registered processor.
The caller must validate read authorization and observation freshness at the
I/O boundary and again before using restored observations. Neither external
calls nor delivery are fenced by a SQLite transaction.
"""
from __future__ import annotations

from copy import deepcopy
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
from .single_loop_candidate import CandidateState
from .task_budget import BudgetExhausted, TaskBudget


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
            row = conn.execute("""SELECT o.owner_id,o.peer_uid,c.task_state_json,c.owner_id AS checkpoint_owner
                FROM v2_trusted_owner_ingress o
                JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=o.owner_id
                JOIN v2_users u ON u.user_id=o.owner_id AND u.role='admin' AND u.is_active=1
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
            rows = conn.execute("""SELECT role,content,owner_user_id,source_message_id FROM v2_conversation_events
                WHERE source_message_id IN (?,?)""", (f'inbound:{message_id}', f'outbound:owner:{message_id}')).fetchall()
            entries = {item['source_message_id']: (item['role'], item['content'], item['owner_user_id']) for item in rows}
            return entries == {f'inbound:{message_id}': ('user', raw['goal'], row['owner_id']),
                               f'outbound:owner:{message_id}': ('assistant', state.response, row['owner_id'])}
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
        self.version = None

    def _transaction(self, *, _finalize=None):
        return self.queue.trusted_local_owner_claim_transaction(self.receipt, self.task, clock=self.clock,
                                                               _finalize=_finalize)

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

    def _load(self, conn, version):
        row = self.store._row(conn, self.task.task_id)
        raw = self.store._validate_identity(row, self.task.task_id, self.task.user, version)
        if raw.get('message_id') != self.receipt.message_id:
            raise CheckpointConflict('owner_atomic_checkpoint_identity_mismatch')
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
            row, raw, state, budget = self._load(conn, expected_version)
            if 'owner_read_pending_decision' in raw['metadata']:
                raise CheckpointConflict('owner_atomic_pending_decision_unresolved')
            observations = raw['metadata'].get('owner_read_observations')
            if not isinstance(observations, dict) or set(observations) != set(state.evidence_sources):
                raise CheckpointConflict('owner_atomic_observation_bindings_invalid')
            return Checkpoint(row['version'], state, budget), deepcopy(observations)

    def take_desktop_draft(self, context: Any) -> dict:
        """Start one owner-only pickup after revalidating its sources.

        A started pickup is never replayed automatically: a lost IPC response
        leaves delivery uncertain. This does not confirm user receipt, fence
        external I/O, authorize other channels, or enable the normal daemon.
        """
        from .owner_read_recovery import restore_owner_read_observations
        expected_version = self.version
        restore_owner_read_observations(self, context, expected_version=expected_version, allow_completed=True)
        acknowledgement_token = secrets.token_hex(16)
        with self._transaction() as conn:
            row, raw, state, budget = self._load(conn, expected_version)
            metadata = raw['metadata']
            if 'owner_clarification' in self.task.metadata:
                from .owner_clarification import load_bound_owner_question
                load_bound_owner_question(conn, self.task.user, self.task.metadata['owner_clarification'])
            selected = metadata.get('owner_read_selected_draft')
            observations = metadata.get('owner_read_observations')
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
            metadata['owner_read_desktop_pickup_started'] = {
                'schema_version': 1, 'claim_digest': _claim_digest(self.receipt),
                'answer_sha256': selected['answer_sha256'], 'channel': 'desktop',
                'acknowledgement_sha256': hashlib.sha256(acknowledgement_token.encode('ascii')).hexdigest(),
                'status': 'started_unconfirmed',
            }
            result = self.store._write(conn, row, raw, state, budget)
            response = {'message_id': self.task.message_id, 'message': state.response,
                        'kind': selected['kind'],
                        'delivery_status': 'started_unconfirmed', 'acknowledgement_token': acknowledgement_token}
        self.version = result.version
        return response

    def acknowledge_desktop_draft(self, context: Any, token: str) -> None:
        """Record a client acknowledgement and conversation in one transaction.

        The acknowledgment does not prove a person read the response. The
        conversation store must share the claim database so the record cannot
        commit independently of the live-claim and checkpoint checks.
        """
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
                    or marker.get('answer_sha256') != hashlib.sha256(state.response.encode('utf-8')).hexdigest()):
                raise CheckpointConflict('owner_desktop_ack_unavailable')
            if marker['status'] == 'client_acknowledged':
                return
            incoming = conversation.record(owner_user_id=self.task.user, role='user',
                content=self.task.goal, source='desktop', source_message_id=f'inbound:{self.task.message_id}',
                connection=conn)
            outgoing = conversation.record(owner_user_id=self.task.user, role='assistant',
                content=state.response, source='desktop', source_message_id=f'outbound:owner:{self.task.message_id}',
                connection=conn)
            if incoming is None or outgoing is None:
                raise CheckpointConflict('owner_desktop_history_unavailable')
            marker['status'] = 'client_acknowledged'
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
                if marker is not None:
                    if not isinstance(marker, dict) or marker.get('claim_digest') != _claim_digest(self.receipt):
                        raise CheckpointConflict('owner_atomic_pending_decision_unavailable')
                    del raw['metadata']['owner_read_pending_decision']
            result = self.store._write(conn, row, raw, state, budget)
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
            if any(item['kind'] == 'model' and item['status'] == 'settled'
                   and any(item['actual'][key] > item['reserved'][key] for key in ('input_tokens', 'output_tokens'))
                   for item in budget.to_dict()['reservations'].values()):
                raise CheckpointConflict('owner_atomic_model_reservation_overrun')
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
            raw['metadata']['owner_read_pending_decision'] = {
                'reservation_id': reservation_id, 'claim_digest': _claim_digest(self.receipt),
                'decision_sha256': decision_digest, 'projection_sha256': projection_fingerprint,
            }
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return result

    def settle_read(self, *, expected_version: int, reservation_id: str, tool_id: str,
                    arguments: dict, evidence_ref: str, observation: dict) -> Checkpoint:
        """Store an already-validated read result, action binding, and budget together.

        This deliberately cannot settle a failed or ambiguous action. Observation
        validity and actor visibility must be established and revalidated by the
        host; this API does not assert those semantic properties.
        """
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
            budget.settle(reservation_id, outcome='success')
            observations[evidence_ref] = {'actor': state.actor, 'tool_id': tool_id,
                                          'arguments': deepcopy(arguments),
                                          'observation_sha256': observation_digest(observation)}
            state.evidence_sources.append(evidence_ref)
            state.inflight = ''
            del raw['metadata']['owner_read_pending_action']
            result = self.store._write(conn, row, raw, state, budget)
        self.version = result.version
        return result
