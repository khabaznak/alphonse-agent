"""Inactive trusted-local continuation grants; these do not resume work.

The caller must independently verify the Unix peer, as for owner ingress.
The trusted-local boundary is the OS account: a same-UID process, including
host Bash, can impersonate the Desktop. This is not a release-grade boundary.
"""
from __future__ import annotations

from .owner_atomic_checkpoint import _decision_digest
from .single_loop_checkpoint import CheckpointConflict


def _valid_approval(checkpoint, pause, question, choice):
    from .owner_continuation_question import valid_question
    if (not isinstance(pause, dict) or type(pause.get('schema_version')) is not int
            or pause['schema_version'] not in {1, 2} or not valid_question(question, pause, 'acknowledged')
            or not isinstance(choice, dict)
            or set(choice) != {'schema_version', 'owner_id', 'message_id', 'pause_version', 'pause_sha256',
                               'question_sha256', 'approved'}
            or type(choice['schema_version']) is not int or choice['schema_version'] != 1
            or choice['approved'] is not True or choice['owner_id'] != checkpoint.task.user
            or choice['message_id'] != checkpoint.task.message_id
            or type(choice['pause_version']) is not int or choice['pause_version'] != question['pause_version']
            or choice['pause_sha256'] != question['pause_sha256']
            or choice['question_sha256'] != question['question_sha256']):
        return False
    return True


def issue_grant(checkpoint, *, expected_version, now_ms, lifetime_ms, cancellation_checker):
    """Record an expiring one-window grant from an acknowledged owner choice.

    ``now_ms`` must come from the trusted host clock, not the Desktop client.
    This local-only experimental stage does not release, requeue, or resume the
    task. The input/choice helper requires a live trusted-local owner claim.
    Same-UID host processes are trusted in this development mode.
    """
    from .owner_continuation_question import _check, _paused
    _check(checkpoint, expected_version, cancellation_checker)
    if type(now_ms) is not int or now_ms < 0 or type(lifetime_ms) is not int or lifetime_ms <= 0:
        raise ValueError('owner_continuation_grant_unavailable')
    with checkpoint._transaction() as conn:
        row, raw, state, budget, pause = _paused(checkpoint, conn, expected_version, cancellation_checker)
        md = raw['metadata']
        question, choice = md.get('owner_continuation_question'), md.get('owner_continuation_choice')
        if (not _valid_approval(checkpoint, pause, question, choice)
                or 'owner_continuation_grant' in md or 'owner_continuation_handoff' in md
                or pause.get('metric') not in {'actions', 'model_calls', 'input_tokens', 'output_tokens'}
                or type(pause.get('interval')) is not int or pause['interval'] <= 0
                or type(pause.get('counter')) is not int or pause['counter'] < 0
                or budget.usage()['charged'][pause['metric']] != pause['counter']):
            raise CheckpointConflict('owner_continuation_grant_unavailable')
        md['owner_continuation_grant'] = {
            'schema_version': 1, 'owner_id': checkpoint.task.user, 'message_id': checkpoint.task.message_id,
            'pause_version': choice['pause_version'], 'pause_sha256': choice['pause_sha256'],
            'choice_sha256': _decision_digest(choice), 'metric': pause['metric'],
            'checkpoint': pause['counter'], 'through': pause['counter'] + pause['interval'],
            'issued_at_ms': now_ms, 'expires_at_ms': now_ms + lifetime_ms, 'status': 'issued',
        }
        result = checkpoint.store._write(conn, row, raw, state, budget)
        if cancellation_checker() is not False:
            raise CheckpointConflict('owner_continuation_grant_unavailable')
    checkpoint.version = result.version
    return result.version


def prepare_handoff(checkpoint, *, expected_version, now_ms, cancellation_checker):
    """Consume a matching, unexpired grant into a held handoff.

    This records preparation only. It intentionally leaves the original queue
    claim untouched until a worker can revalidate evidence and accept a fresh
    claim without repeating previously dispatched work.
    """
    from .owner_continuation_question import _check, _paused
    _check(checkpoint, expected_version, cancellation_checker)
    if type(now_ms) is not int or now_ms < 0:
        raise ValueError('owner_continuation_handoff_unavailable')
    with checkpoint._transaction() as conn:
        row, raw, state, budget, pause = _paused(checkpoint, conn, expected_version, cancellation_checker)
        md = raw['metadata']
        grant, choice = md.get('owner_continuation_grant'), md.get('owner_continuation_choice')
        if (not _valid_approval(checkpoint, pause, md.get('owner_continuation_question'), choice)
                or not isinstance(grant, dict) or set(grant) != {'schema_version', 'owner_id', 'message_id',
                    'pause_version', 'pause_sha256', 'choice_sha256', 'metric', 'checkpoint', 'through',
                    'issued_at_ms', 'expires_at_ms', 'status'}
                or type(grant.get('schema_version')) is not int or grant['schema_version'] != 1
                or grant.get('status') != 'issued' or grant.get('pause_version') != choice['pause_version']
                or grant.get('owner_id') != checkpoint.task.user or grant.get('message_id') != checkpoint.task.message_id
                or grant.get('pause_sha256') != _decision_digest(pause)
                or not isinstance(choice, dict) or grant.get('choice_sha256') != _decision_digest(choice)
                or type(grant.get('issued_at_ms')) is not int or type(grant.get('expires_at_ms')) is not int
                or not grant['issued_at_ms'] <= now_ms < grant['expires_at_ms']
                or grant.get('metric') != pause.get('metric') or grant.get('checkpoint') != pause.get('counter')
                or grant.get('through') != pause.get('counter') + pause.get('interval')
                or 'owner_continuation_handoff' in md):
            raise CheckpointConflict('owner_continuation_handoff_unavailable')
        grant['status'] = 'consumed'
        md['owner_continuation_handoff'] = {'schema_version': 1, 'status': 'held',
            'grant_sha256': _decision_digest(grant), 'original_claim_digest': pause['claim_digest'],
            'prepared_at_ms': now_ms}
        result = checkpoint.store._write(conn, row, raw, state, budget)
        if cancellation_checker() is not False:
            raise CheckpointConflict('owner_continuation_handoff_unavailable')
    checkpoint.version = result.version
    return result.version
