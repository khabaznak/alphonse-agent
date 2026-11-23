"""Experimental claim-bound question and choice records; not wired to IPC.

The caller must independently verify its local peer and ensure the Boolean
choice comes from an explicit owner control. Same-UID host processes can
impersonate Desktop in trusted-local mode. These records neither resume work
nor independently authorize a grant.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from .owner_atomic_checkpoint import OwnerAtomicCheckpoint, _decision_digest, _claim_digest
from .single_loop_checkpoint import CheckpointConflict


def render_window_question(pause, window):
    """The acknowledged text binds the exact work window to the choice."""
    amounts = window['increment']
    assessment = pause.get('progress_assessment', 'unknown')
    progress = {'progress': 'A preliminary review found evidence of progress. ',
                'no_progress': 'A preliminary review did not find evidence of progress. ',
                'unknown': 'Progress could not be assessed. '}[assessment]
    # Old questions must retain their exact text for durable recovery.
    if pause.get('schema_version') == 1:
        progress = ''
    return (progress + f"This request has reached a checkpoint after {pause['counter']} "
            f"{pause['metric'].replace('_', ' ')}. Increase this request’s work allowance by "
            f"{amounts['actions']} tool actions, {amounts['model_calls']} model calls, "
            f"{amounts['input_tokens']} input tokens, {amounts['output_tokens']} output tokens "
            f"and {window['elapsed_increment_ms'] // 1000} seconds of work?")


def _check(checkpoint, expected_version, cancellation_checker):
    if (not isinstance(checkpoint, OwnerAtomicCheckpoint) or checkpoint.scheduled_origin
            or checkpoint.telegram_origin or type(expected_version) is not int or expected_version < 1
            or not callable(cancellation_checker)):
        raise ValueError('owner_continuation_question_unavailable')


def _paused(checkpoint, conn, expected_version, cancellation_checker):
    row, raw, state, budget = checkpoint._load(conn, expected_version, allow_paused=True)
    md = raw['metadata']
    pause = md.get('owner_continuation_pause')
    keys = {'schema_version', 'claim_digest', 'metric', 'interval', 'counter', 'previous_counter', 'paused_at_ms',
            'observations_sha256', 'state_sha256'}
    if isinstance(pause, dict) and pause.get('schema_version') == 2:
        keys.update({'progress_assessment', 'evidence_refs'})
    lineage = md.get('owner_continuation_reclaim')
    current = _claim_digest(checkpoint.receipt)
    valid_lineage = (isinstance(lineage, dict) and set(lineage) == {'schema_version', 'original_claim_digest',
        'current_claim_digest', 'grant_sha256', 'pause_sha256'} and type(lineage.get('schema_version')) is int
        and lineage['schema_version'] == 1 and isinstance(pause, dict)
        and lineage.get('original_claim_digest') == pause.get('claim_digest')
        and lineage.get('current_claim_digest') == current and lineage.get('pause_sha256') == _decision_digest(pause)
        and lineage.get('grant_sha256') == _decision_digest(md.get('owner_continuation_grant')))
    if (cancellation_checker() is not False or not isinstance(pause, dict) or set(pause) != keys
            or type(pause['schema_version']) is not int or pause['schema_version'] not in {1, 2}
            or (pause['schema_version'] == 2 and pause.get('progress_assessment') not in {'progress', 'no_progress', 'unknown'})
            or (pause['schema_version'] == 2 and pause.get('evidence_refs') != state.evidence_sources)
            or (pause['claim_digest'] != current and not valid_lineage)
            or pause['metric'] not in {'actions', 'model_calls', 'input_tokens', 'output_tokens'}
            or type(pause['interval']) is not int or pause['interval'] <= 0
            or type(pause['paused_at_ms']) is not int or pause['paused_at_ms'] < 0
            or type(pause['counter']) is not int or type(pause['previous_counter']) is not int
            or pause['previous_counter'] < 0 or pause['counter'] - pause['previous_counter'] < pause['interval']
            or budget.usage()['charged'][pause['metric']] != pause['counter']
            or state.status != 'running' or state.inflight
            or budget.usage()['in_flight'] or budget.usage()['unknown_usage']
            or not isinstance(md.get('owner_read_observations'), dict)
            or set(md['owner_read_observations']) != set(state.evidence_sources)
            or pause.get('state_sha256') != _decision_digest(state.snapshot())
            or pause.get('observations_sha256') != _decision_digest(md.get('owner_read_observations'))):
        raise CheckpointConflict('owner_continuation_question_unavailable')
    return row, raw, state, budget, pause


def valid_question(q, pause, status):
    keys = {'schema_version', 'pause_version', 'pause_sha256', 'question', 'question_sha256',
            'acknowledgement_sha256', 'status'}
    return (isinstance(q, dict) and set(q) == keys
            and type(q['schema_version']) is int and q['schema_version'] == 1
            and type(q['pause_version']) is int and q['pause_version'] > 0
            and q['status'] == status and q['pause_sha256'] == _decision_digest(pause)
            and isinstance(q['question'], str) and bool(q['question'].strip())
            and len(q['question'].encode('utf-8')) <= 4096
            and q['question_sha256'] == hashlib.sha256(q['question'].encode('utf-8')).hexdigest()
            and isinstance(q['acknowledgement_sha256'], str)
            and len(q['acknowledgement_sha256']) == 64)


def start_question(checkpoint, *, expected_version, question, cancellation_checker):
    """Record one delivery attempt before exposing its text and token."""
    _check(checkpoint, expected_version, cancellation_checker)
    if not isinstance(question, str) or not question.strip() or len(question.encode('utf-8')) > 4096:
        raise ValueError('owner_continuation_question_unavailable')
    token = secrets.token_hex(16)
    with checkpoint._transaction() as conn:
        row, raw, state, budget, pause = _paused(checkpoint, conn, expected_version, cancellation_checker)
        md = raw['metadata']
        if any(k in md for k in ('owner_continuation_question', 'owner_continuation_choice', 'owner_continuation_grant')):
            raise CheckpointConflict('owner_continuation_question_unavailable')
        md['owner_continuation_question'] = {
            'schema_version': 1, 'pause_version': expected_version,
            'pause_sha256': _decision_digest(pause), 'question': question,
            'question_sha256': hashlib.sha256(question.encode('utf-8')).hexdigest(),
            'acknowledgement_sha256': hashlib.sha256(token.encode('ascii')).hexdigest(),
            'status': 'started_unconfirmed',
        }
        result = checkpoint.store._write(conn, row, raw, state, budget)
        if cancellation_checker() is not False:
            raise CheckpointConflict('owner_continuation_question_unavailable')
    checkpoint.version = result.version
    return result.version, token


def acknowledge_question(checkpoint, *, expected_version, token, cancellation_checker):
    """Record Desktop acknowledgement; a lost delivery is never retried here."""
    _check(checkpoint, expected_version, cancellation_checker)
    if not isinstance(token, str):
        raise ValueError('owner_continuation_question_unavailable')
    with checkpoint._transaction() as conn:
        row, raw, state, budget, pause = _paused(checkpoint, conn, expected_version, cancellation_checker)
        q = raw['metadata'].get('owner_continuation_question')
        digest = hashlib.sha256(token.encode('utf-8')).hexdigest()
        if (not valid_question(q, pause, 'started_unconfirmed')
                or not hmac.compare_digest(q['acknowledgement_sha256'], digest)):
            raise CheckpointConflict('owner_continuation_question_unavailable')
        q['status'] = 'acknowledged'
        result = checkpoint.store._write(conn, row, raw, state, budget)
        if cancellation_checker() is not False:
            raise CheckpointConflict('owner_continuation_question_unavailable')
    checkpoint.version = result.version
    return result.version


def record_choice(checkpoint, *, expected_version, peer_uid, approve, cancellation_checker):
    """Record an explicit UI choice, bound to the acknowledged question."""
    _check(checkpoint, expected_version, cancellation_checker)
    if type(peer_uid) is not int or peer_uid != os.getuid() or type(approve) is not bool:
        raise ValueError('owner_continuation_choice_unavailable')
    with checkpoint._transaction() as conn:
        row, raw, state, budget, pause = _paused(checkpoint, conn, expected_version, cancellation_checker)
        md = raw['metadata']
        q = md.get('owner_continuation_question')
        if (not valid_question(q, pause, 'acknowledged')
                or 'owner_continuation_choice' in md or 'owner_continuation_grant' in md):
            raise CheckpointConflict('owner_continuation_choice_unavailable')
        md['owner_continuation_choice'] = {
            'schema_version': 1, 'owner_id': checkpoint.task.user, 'message_id': checkpoint.task.message_id,
            'pause_version': q['pause_version'], 'pause_sha256': q['pause_sha256'],
            'question_sha256': q['question_sha256'], 'approved': approve,
        }
        result = checkpoint.store._write(conn, row, raw, state, budget)
        if cancellation_checker() is not False:
            raise CheckpointConflict('owner_continuation_choice_unavailable')
    checkpoint.version = result.version
    return result.version
