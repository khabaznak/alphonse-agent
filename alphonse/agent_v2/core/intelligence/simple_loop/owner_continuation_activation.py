"""Activate a quiescent, owner-approved local continuation.

The worker must hold the original claim or a fenced fresh claim bound to the
pause, and must revalidate evidence before using the renewed work budget. The
trusted-local OS user is not a release-grade authorization boundary.
"""

from __future__ import annotations

from copy import deepcopy

from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
from .owner_atomic_checkpoint import _decision_digest
from .owner_continuation_question import _check, _paused
from .owner_continuation_grant import _valid_approval
from .single_loop_checkpoint import CheckpointConflict
from .task_budget import DIMENSIONS


def activate_held_same_claim(checkpoint, *, expected_version, context, cancellation_checker):
    """Atomically activate a held owner grant under a valid linked claim.

    A prepared handoff is NOT itself a worker dispatch. Time is read from the
    fenced host clock; configured work-window increments are separately bounded.
    The full budget ledger and source bindings are retained without replay.
    This does not itself acquire a worker claim or dispatch work.
    """
    _check(checkpoint, expected_version, cancellation_checker)
    # Perform source I/O while still paused; the subsequent claim-bound
    # transaction rechecks the exact checkpoint and cancellation afterwards.
    from .owner_read_recovery import restore_owner_read_observations
    restore_owner_read_observations(checkpoint, context, expected_version=expected_version)
    with checkpoint._transaction() as conn:
        row, raw, state, budget, pause = _paused(checkpoint, conn, expected_version, cancellation_checker)
        md = raw['metadata']
        question = md.get('owner_continuation_question')
        choice = md.get('owner_continuation_choice')
        grant = md.get('owner_continuation_grant')
        handoff = md.get('owner_continuation_handoff')
        now = checkpoint.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise CheckpointConflict('owner_continuation_activation_unavailable')
        now_ms = int(now.timestamp() * 1000)
        digest = pause.get('claim_digest')
        current_digest = _claim_digest(checkpoint.receipt)
        reclaim = md.get('owner_continuation_reclaim')
        valid_reclaim = (isinstance(reclaim, dict) and reclaim.get('schema_version') == 1
            and reclaim.get('original_claim_digest') == digest and reclaim.get('current_claim_digest') == current_digest
            and reclaim.get('pause_sha256') == _decision_digest(pause)
            and reclaim.get('grant_sha256') == _decision_digest(grant))
        metric = pause.get('metric')
        interval = pause.get('interval')
        started = pause.get('paused_at_ms')
        if (not _valid_approval(checkpoint, pause, question, choice)
                or not isinstance(grant, dict) or set(grant) != {'schema_version', 'owner_id', 'message_id',
                    'pause_version', 'pause_sha256', 'choice_sha256', 'metric', 'checkpoint', 'through',
                    'issued_at_ms', 'expires_at_ms', 'status'}
                or type(grant.get('schema_version')) is not int or grant['schema_version'] != 1
                or grant.get('status') != 'consumed'
                or grant.get('owner_id') != checkpoint.task.user or grant.get('message_id') != checkpoint.task.message_id
                or grant.get('pause_version') != question.get('pause_version')
                or grant.get('pause_sha256') != _decision_digest(pause)
                or grant.get('choice_sha256') != _decision_digest(choice)
                or not isinstance(handoff, dict) or set(handoff) != {'schema_version', 'status', 'grant_sha256',
                    'original_claim_digest', 'prepared_at_ms'}
                or type(handoff.get('schema_version')) is not int or handoff['schema_version'] != 1
                or handoff.get('status') != 'held'
                or handoff.get('grant_sha256') != _decision_digest(grant)
                or handoff.get('original_claim_digest') != digest
                or (digest != current_digest and not valid_reclaim)
                or metric not in DIMENSIONS or type(interval) is not int or interval <= 0
                or type(started) is not int or started < 0 or started > now_ms
                or type(grant.get('checkpoint')) is not int or grant['checkpoint'] != pause.get('counter')
                or grant.get('metric') != metric or grant.get('through') != grant['checkpoint'] + interval
                or budget.usage()['charged'][metric] != grant['checkpoint']
                or type(grant.get('issued_at_ms')) is not int or type(grant.get('expires_at_ms')) is not int
                or type(handoff.get('prepared_at_ms')) is not int
                or not started <= grant['issued_at_ms'] <= handoff['prepared_at_ms'] <= now_ms < grant['expires_at_ms']):
            raise CheckpointConflict('owner_continuation_activation_unavailable')
        increments = {key: interval if key == metric else 0 for key in DIMENSIONS}
        caps = increments
        elapsed_increment = elapsed_cap = 0
        configured_window = getattr(context, 'owner_continuation_budget_window', None)
        if configured_window is not None:
            from .owner_continuation_question import render_window_question
            if question.get('question') != render_window_question(pause, configured_window):
                raise CheckpointConflict('owner_continuation_activation_unavailable')
            keys = {'increment','increment_caps','elapsed_increment_ms','elapsed_increment_cap_ms',
                    'total_caps','total_elapsed_cap_ms'}
            if not isinstance(configured_window, dict) or set(configured_window) != keys:
                raise CheckpointConflict('owner_continuation_activation_unavailable')
            increments = configured_window['increment']
            caps = configured_window['increment_caps']
            totals = configured_window['total_caps']
            elapsed_increment, elapsed_cap = configured_window['elapsed_increment_ms'], configured_window['elapsed_increment_cap_ms']
            total_elapsed = configured_window['total_elapsed_cap_ms']
            if (not all(isinstance(v, dict) and set(v) == set(DIMENSIONS) for v in (increments, caps))
                    or (totals is not None and (not isinstance(totals, dict) or set(totals) != set(DIMENSIONS)))
                    or any(type(increments[k]) is not int or type(caps[k]) is not int
                        or increments[k] < 0 or caps[k] < 0
                        or (totals is not None and (type(totals[k]) is not int or totals[k] < 0
                            or budget.limits.total[k] + increments[k] > totals[k])) for k in DIMENSIONS)
                    or increments[metric] != interval
                    or type(elapsed_increment) is not int or elapsed_increment < 0
                    or (total_elapsed is not None and (type(total_elapsed) is not int or total_elapsed < 0
                        or budget.limits.elapsed_ms + elapsed_increment > total_elapsed))):
                raise CheckpointConflict('owner_continuation_activation_unavailable')
        try:
            budget.apply_continuation_window(grant_id=_decision_digest(grant), pause_started_ms=started,
                resumed_ms=now_ms, increment=increments, increment_caps=caps,
                elapsed_increment_ms=elapsed_increment, elapsed_increment_cap_ms=elapsed_cap)
        except ValueError:
            raise CheckpointConflict('owner_continuation_activation_unavailable') from None
        history = md.setdefault('owner_continuation_history', [])
        if not isinstance(history, list):
            raise CheckpointConflict('owner_continuation_activation_unavailable')
        prior_crashes = md.pop('owner_continuation_crash_lineage', [])
        if not isinstance(prior_crashes, list):
            raise CheckpointConflict('owner_continuation_activation_unavailable')
        history.append(deepcopy({'pause': pause, 'question': question, 'choice': choice,
                                'grant': grant, 'handoff': handoff, 'reclaim': reclaim,
                                'crash_before_pause': prior_crashes, 'resumed_at_ms': now_ms}))
        for key in ('owner_continuation_pause', 'owner_continuation_question', 'owner_continuation_choice',
                    'owner_continuation_grant', 'owner_continuation_handoff'):
            del md[key]
        md.pop('owner_continuation_reclaim', None)
        result = checkpoint.store._write(conn, row, raw, state, budget)
        if cancellation_checker() is not False:
            raise CheckpointConflict('owner_continuation_activation_unavailable')
    checkpoint.version = result.version
    return result
