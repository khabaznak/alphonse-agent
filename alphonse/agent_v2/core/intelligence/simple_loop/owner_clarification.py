"""Validate an explicit link to a previously acknowledged owner question.

Only the local server supplies this binding. It adds quoted conversation
context to a new request; it does not resume or authorize the previous task.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from typing import Any


def resolve_owner_question(conn: Any, owner: str, message_id: str, *, allow_answer: bool = False,
                           _visited: frozenset[str] = frozenset()) -> tuple[str, str, str]:
    """Return the verified prior user turn, delivered question and digest.

    The caller holds a same-database transaction and is responsible for
    authenticating the current owner. Missing, deleted, changed, or unfinished
    sources must fail closed rather than silently removing reply context.
    """
    from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
    from .single_loop_checkpoint import STATE_KEY
    from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import TaskBudget

    if not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None:
        raise ValueError('owner_question_context_unavailable')
    if message_id in _visited or len(_visited) >= 100:
        raise ValueError('owner_question_context_unavailable')
    try:
        row = conn.execute('SELECT owner_id,task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                           (message_id,)).fetchone()
        origin = conn.execute('SELECT owner_id,peer_uid,payload_sha256 FROM v2_trusted_owner_ingress WHERE message_id=?',
                              (message_id,)).fetchone()
        if (row is None or row['owner_id'] != owner or origin is None
                or origin['owner_id'] != owner or origin['peer_uid'] != os.getuid()
                or not isinstance(origin['payload_sha256'], str)
                or re.fullmatch(r'[0-9a-f]{64}', origin['payload_sha256']) is None):
            raise ValueError('owner_question_context_unavailable')
        raw = json.loads(row['task_state_json'])
        from alphonse.agent_v2.core.messages.sqlite_queue import finalized_owner_history_matches
        if not finalized_owner_history_matches(conn, message_id, owner, raw):
            raise ValueError('owner_question_context_unavailable')
        metadata = raw['metadata']
        ancestor = metadata.get('owner_followup')
        if ancestor is not None:
            if (not isinstance(ancestor, dict) or set(ancestor) != {'message_id', 'sha256'}
                    or not isinstance(ancestor['sha256'], str)
                    or re.fullmatch(r'[0-9a-f]{64}', ancestor['sha256']) is None):
                raise ValueError('owner_question_context_unavailable')
            _, _, ancestor_digest = resolve_owner_question(conn, owner, ancestor['message_id'],
                allow_answer=True, _visited=_visited | {message_id})
            if not hmac.compare_digest(ancestor_digest, ancestor['sha256']):
                raise ValueError('owner_question_context_unavailable')
        envelope = metadata[STATE_KEY]
        state = CandidateState.restore(envelope['state'])
        budget = TaskBudget.from_dict(envelope['budget'])
        if not budget.within_work_limits():
            raise ValueError('owner_question_context_unavailable')
        selected = metadata['owner_read_selected_draft']
        pickup = metadata['owner_read_desktop_pickup_started']
        observations = metadata['owner_read_observations']
        allowed_metadata = {'intelligence_engine', 'intelligence_schema_version', 'routing_disposition',
                            'owner_clarification', 'owner_followup', 'owner_steering_included_projection', 'owner_host_bash_history', STATE_KEY, 'owner_read_observations',
                            'owner_read_selected_draft', 'owner_read_desktop_pickup_started'}
        if (raw.get('task_id') != message_id or raw.get('message_id') != message_id
                or raw.get('user') != owner or raw.get('intelligence_engine') != 'single_loop_experimental'
                or type(raw.get('intelligence_schema_version')) is not int
                or raw['intelligence_schema_version'] != 1 or state.task_id != message_id
                or type(metadata.get('intelligence_schema_version')) is not int
                or metadata['intelligence_schema_version'] != 1
                or metadata.get('intelligence_engine') != 'single_loop_experimental'
                or not isinstance(envelope, dict) or type(envelope.get('schema_version')) is not int
                or envelope['schema_version'] != 1
                or state.actor != owner or state.status not in ({'waiting_user', 'completed'} if allow_answer else {'waiting_user'}) or state.inflight
                or budget.usage()['in_flight'] or budget.usage()['unknown_usage']
                or set(metadata) - allowed_metadata
                or not isinstance(selected, dict) or selected.get('kind') != ('answer' if state.status == 'completed' else 'question')
                or type(selected.get('schema_version')) is not int or selected['schema_version'] != 1
                or not isinstance(pickup, dict) or pickup.get('status') != 'client_acknowledged'
                or type(pickup.get('schema_version')) is not int or pickup['schema_version'] != 1
                or pickup.get('channel') != 'desktop'
                or not isinstance(selected.get('claim_digest'), str)
                or re.fullmatch(r'[0-9a-f]{64}', selected['claim_digest']) is None
                or not isinstance(selected.get('decision_sha256'), str)
                or re.fullmatch(r'[0-9a-f]{64}', selected['decision_sha256']) is None
                or not isinstance(selected.get('projection_sha256'), str)
                or re.fullmatch(r'[0-9a-f]{64}', selected['projection_sha256']) is None
                or not isinstance(pickup.get('acknowledgement_sha256'), str)
                or re.fullmatch(r'[0-9a-f]{64}', pickup['acknowledgement_sha256']) is None
                or pickup.get('claim_digest') != selected['claim_digest']
                or not isinstance(observations, dict) or set(observations) != set(state.evidence_sources)
                or selected.get('evidence_refs') != state.evidence_sources
                or selected.get('evidence_bindings_sha256') != hashlib.sha256(json.dumps(
                    observations, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                    allow_nan=False).encode('utf-8')).hexdigest()
                or 'owner_read_pending_action' in metadata or 'owner_read_pending_decision' in metadata
                or pickup.get('answer_sha256') != selected.get('answer_sha256')
                or pickup.get('answer_sha256') != hashlib.sha256(state.response.encode('utf-8')).hexdigest()
                or conn.execute('SELECT 1 FROM v2_outbox WHERE task_id=? LIMIT 1', (message_id,)).fetchone()
                or conn.execute('SELECT 1 FROM v2_scheduled_handoffs WHERE message_id=?', (message_id,)).fetchone()
                or conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()):
            raise ValueError('owner_question_context_unavailable')
        history = metadata.get('owner_host_bash_history')
        has_bash_observations = any(isinstance(value, dict) and value.get('tool_id') == 'native.bash'
                                    for value in observations.values())
        if has_bash_observations != (history is not None):
            raise ValueError('owner_question_context_unavailable')
        if history is not None:
            # A returned process is historical evidence, never proof of its
            # external effect. Validate every persisted binding and completed
            # budget reservation; unresolved dispatch/return state is refused.
            from .owner_atomic_checkpoint import _decision_digest, observation_digest
            from .owner_host_bash import valid_host_output
            if (not isinstance(history, list) or not history
                    or 'owner_bash_dispatch_started' in metadata or 'owner_bash_return_unreconciled' in metadata):
                raise ValueError('owner_question_context_unavailable')
            bash_items = [value for value in observations.values() if isinstance(value, dict) and value.get('tool_id') == 'native.bash']
            if len(history) != len(bash_items):
                raise ValueError('owner_question_context_unavailable')
            unmatched = list(bash_items)
            seen = set()
            reservations = budget.to_dict()['reservations']
            for entry in history:
                if (not isinstance(entry, dict) or set(entry) != {'reservation_id', 'claim_digest', 'arguments_sha256', 'observation_sha256', 'effect_status'}
                        or entry['claim_digest'] != selected['claim_digest'] or entry['effect_status'] != 'unverified'
                        or not isinstance(entry['reservation_id'], str) or entry['reservation_id'] in seen):
                    raise ValueError('owner_question_context_unavailable')
                seen.add(entry['reservation_id'])
                reservation = reservations.get(entry['reservation_id'])
                if (not isinstance(reservation, dict) or reservation['kind'] != 'tool'
                        or reservation['status'] != 'settled' or reservation['outcome'] != 'success'):
                    raise ValueError('owner_question_context_unavailable')
                matches = [item for item in unmatched if item.get('actor') == owner
                    and isinstance(item.get('arguments'), dict)
                    and _decision_digest(item['arguments']) == entry['arguments_sha256']
                    and valid_host_output(item.get('host_bash_observation'))
                    and item.get('observation_sha256') == entry['observation_sha256']
                    and observation_digest(item['host_bash_observation']) == entry['observation_sha256']]
                if not matches:
                    raise ValueError('owner_question_context_unavailable')
                unmatched.remove(matches[0])
        rows = conn.execute('''SELECT role,content,owner_user_id,source_message_id,sequence
            FROM v2_conversation_events WHERE source_message_id IN (?,?)''',
            (f'inbound:{message_id}', f'outbound:owner:{message_id}')).fetchall()
        entries = {entry['source_message_id']: entry for entry in rows}
        incoming, outgoing = entries[f'inbound:{message_id}'], entries[f'outbound:owner:{message_id}']
        from .owner_clock import display_response
        if (len(rows) != 2 or incoming['owner_user_id'] != owner or outgoing['owner_user_id'] != owner
                or incoming['role'] != 'user' or outgoing['role'] != 'assistant'
                or incoming['content'] != raw['goal'] or outgoing['content'] != display_response(state.response, observations)
                or incoming['sequence'] >= outgoing['sequence']):
            raise ValueError('owner_question_context_unavailable')
        content = [incoming['content'], outgoing['content']]
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_steering'").fetchone():
            steers = conn.execute('''SELECT * FROM v2_owner_steering WHERE message_id=? AND owner_id=?
                ORDER BY sequence''', (message_id, owner)).fetchall()
            marker = metadata.get('owner_steering_included_projection')
            if steers:
                from .owner_atomic_checkpoint import _decision_digest
                if (not isinstance(marker, dict) or marker.get('reservation_id') != selected.get('model_reservation_id')
                        or marker.get('projection_sha256') != selected['projection_sha256']
                        or marker.get('steering_ids') != [steer['id'] for steer in steers]
                        or marker.get('steering_sha256') != _decision_digest([[steer['id'], steer['content_digest']] for steer in steers])):
                    raise ValueError('owner_question_context_unavailable')
            elif marker is not None:
                raise ValueError('owner_question_context_unavailable')
            corrections = []
            for steer in steers:
                digest = hashlib.sha256((steer['id'] + '\0' + steer['content']).encode()).hexdigest()
                if (steer['status'] != 'included_in_planning' or steer['peer_uid'] != os.getuid()
                        or steer['claim_digest'] != selected['claim_digest']
                        or not hmac.compare_digest(steer['content_digest'], digest)):
                    raise ValueError('owner_question_context_unavailable')
                corrections.append(steer['content'])
            if corrections:
                content[0] = json.dumps({'original_user_message': content[0],
                                         'owner_corrections_in_order': corrections}, ensure_ascii=False, sort_keys=True)
        elif metadata.get('owner_steering_included_projection') is not None:
            raise ValueError('owner_question_context_unavailable')
        payload = json.dumps([owner, message_id, *content], ensure_ascii=False,
                             separators=(',', ':')).encode('utf-8')
        if len(payload) > 16384:
            raise ValueError('owner_question_context_unavailable')
        return content[0], content[1], hashlib.sha256(payload).hexdigest()
    except Exception:
        raise ValueError('owner_question_context_unavailable') from None


def bind_owner_question(conn: Any, owner: str, message_id: str) -> dict[str, str]:
    _, _, digest = resolve_owner_question(conn, owner, message_id)
    return {'message_id': message_id, 'sha256': digest}


def bind_owner_followup(conn: Any, owner: str, message_id: str) -> dict[str, str]:
    """Bind an acknowledged prior exchange, including an answer or question."""
    _, _, digest = resolve_owner_question(conn, owner, message_id, allow_answer=True)
    return {'message_id': message_id, 'sha256': digest}


def load_bound_owner_followup(conn: Any, owner: str, binding: Any) -> tuple[str, str]:
    if (not isinstance(binding, dict) or set(binding) != {'message_id', 'sha256'}
            or not isinstance(binding['sha256'], str)
            or re.fullmatch(r'[0-9a-f]{64}', binding['sha256']) is None):
        raise ValueError('owner_followup_context_unavailable')
    prompt, response, digest = resolve_owner_question(conn, owner, binding['message_id'], allow_answer=True)
    if not hmac.compare_digest(digest, binding['sha256']):
        raise ValueError('owner_followup_context_unavailable')
    return prompt, response


def load_bound_owner_question(conn: Any, owner: str, binding: Any) -> tuple[str, str]:
    if (not isinstance(binding, dict) or set(binding) != {'message_id', 'sha256'}
            or not isinstance(binding['sha256'], str)
            or re.fullmatch(r'[0-9a-f]{64}', binding['sha256']) is None):
        raise ValueError('owner_question_context_unavailable')
    prompt, question, digest = resolve_owner_question(conn, owner, binding['message_id'])
    if not hmac.compare_digest(digest, binding['sha256']):
        raise ValueError('owner_question_context_unavailable')
    return prompt, question


@dataclass(frozen=True)
class BoundOwnerExchange:
    """A snapshot of separate owner turns, not authority to perform an action."""
    prior_message_id: str
    reply_message_id: str
    prior_user_message: str
    assistant_question: str
    owner_reply: str
    sha256: str


def load_bound_owner_exchange(conn: Any, task: Any) -> BoundOwnerExchange:
    """Revalidate both turns inside the caller's live owner-claim transaction.

    The queue's claim transaction authenticates the active owner and current
    ingress before and after the caller's work. This function does not replace
    that check, interpret either turn, or grant authority to create an effect.
    The returned digest binds the exact source turns without fabricating a
    concatenated owner utterance.
    """
    try:
        if not conn.in_transaction:
            raise ValueError('owner_question_context_unavailable')
        owner = task.user
        reply_id = task.message_id
        reply = task.goal
        metadata = task.metadata
        if (not isinstance(owner, str) or not owner or not isinstance(reply, str) or not reply.strip()
                or not isinstance(reply_id, str) or re.fullmatch(r'[0-9a-f]{32}', reply_id) is None
                or not isinstance(metadata, dict)):
            raise ValueError('owner_question_context_unavailable')
        binding = metadata.get('owner_clarification')
        if not isinstance(binding, dict) or binding.get('message_id') == reply_id:
            raise ValueError('owner_question_context_unavailable')
        # Effect-specific confirmation currently binds one exact preceding
        # owner utterance. A steered multi-turn intent needs its own explicit
        # effect contract; do not silently collapse it into that older format.
        if (conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_steering'").fetchone()
                and conn.execute('SELECT 1 FROM v2_owner_steering WHERE message_id=? AND owner_id=? LIMIT 1',
                                 (binding.get('message_id'), owner)).fetchone()):
            raise ValueError('owner_question_context_unavailable')
        # Defense in depth. The enclosing claim transaction checks this
        # immutable ingress record, peer and payload digest at both ends.
        row = conn.execute('''SELECT prompt,user_id,metadata_json FROM v2_inbound_messages
                              WHERE message_id=?''', (reply_id,)).fetchone()
        if (row is None or row['user_id'] != owner or row['prompt'] != reply
                or json.loads(row['metadata_json']) != metadata):
            raise ValueError('owner_question_context_unavailable')
        prior, question = load_bound_owner_question(conn, owner, binding)
        payload = json.dumps([owner, binding['message_id'], binding['sha256'], prior,
                              question, reply_id, reply], ensure_ascii=False,
                             separators=(',', ':')).encode('utf-8')
        if len(payload) > 32768:
            raise ValueError('owner_question_context_unavailable')
        return BoundOwnerExchange(binding['message_id'], reply_id, prior, question, reply,
                                  hashlib.sha256(payload).hexdigest())
    except Exception:
        raise ValueError('owner_question_context_unavailable') from None
