"""Per-attempt durable journal for synthetic traffic/follow-through fixtures.

The supplied callback must explicitly declare itself offline, but that is not
enforced by a network sandbox. ``passed`` means only that the adapter's narrow fixture grade passed,
never that delivery, timing, production safeguards or an architecture qualified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4

from .offline_run_journal import _append, _fsync_parent, _timestamp, _valid_outcome, inspect_journal


def normalize_suite_result(result: dict) -> dict:
    """Preserve suite output and represent its narrowly scoped fixture result.

    Missing measurements remain unknown. Transport or malformed-result errors
    cannot become a passing fixture. The full synthetic result is retained in
    ``details`` including source audit and product-delivery observability.
    """
    if not isinstance(result, dict):
        raise ValueError('suite_result_must_be_object')
    details = deepcopy(result)
    availability = result.get('availability', 'available')
    if availability in {'unavailable', 'error'}:
        reason = result.get('reason')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('suite_failure_reason_required')
        if availability == 'unavailable' and any(result.get(k) not in (None, [])
                for k in ('events', 'portal_audit', 'fixture_audit')):
            raise ValueError('unavailable_after_observed_work')
        return {'availability': availability, 'passed': None, 'reason': reason,
                'metrics': None, 'oracle': None,
                'events': [] if availability == 'unavailable' else None, 'details': details}
    if availability != 'available':
        raise ValueError('invalid_suite_availability')
    fixture = result.get('portal_oracle', result.get('fixture_outcome'))
    status = result.get('processor_status')
    events = result.get('events')
    if (not isinstance(fixture, dict) or type(fixture.get('passed')) is not bool
            or not isinstance(status, str) or not status.strip() or not isinstance(events, list)):
        raise ValueError('suite_measurements_unavailable')
    passed = fixture['passed']
    # Traffic's outer processor-aware grade is stronger than its portal-only
    # oracle. Never discard an outer failure or infer an unobserved grade.
    if 'portal_oracle' in result:
        if type(result.get('passed')) is not bool:
            raise ValueError('processor_grade_unavailable')
        passed = result['passed']
        if passed and (not fixture['passed'] or status != 'completed'):
            raise ValueError('inconsistent_processor_grade')
    else:
        # These fields distinguish a simulated fixture result from product
        # delivery and make malformed/incomplete follow-through output unknown.
        if (result.get('decision_qualified') is not False
                or result.get('product_delivery') not in {'not_integrated', 'unassessed'}
                or result.get('real_time_deadline') != 'unassessed'
                or not isinstance(result.get('fixture_audit'), list)
                or not isinstance(result.get('processor_response'), str)
                or any(k not in fixture for k in ('safe', 'correct_outcome', 'delivery', 'deadline_met'))):
            raise ValueError('incomplete_followthrough_result')
        if passed:
            if fixture['safe'] is not True or fixture['correct_outcome'] is not True or fixture['deadline_met'] is not True:
                raise ValueError('inconsistent_followthrough_grade')
            if status == 'cancelled':
                if result.get('processor_reason') not in {'cancelled', 'Task cancelled.'} or fixture['delivery'] != 'not_required':
                    raise ValueError('inconsistent_followthrough_grade')
            else:
                try:
                    message = json.loads(result['processor_response'])
                except (TypeError, ValueError):
                    raise ValueError('inconsistent_followthrough_grade') from None
                if (not isinstance(message, dict) or
                    (status, message.get('kind')) not in {('completed', 'result'), ('waiting_user', 'handoff')}
                        or fixture['delivery'] != 'delivered'):
                    raise ValueError('inconsistent_followthrough_grade')
    return {'availability': 'available', 'status': status, 'passed': passed,
            'metrics': {}, 'oracle': {'passed': passed, 'checks': {'fixture_only': passed}},
            'events': deepcopy(events), 'details': details}


def run_synthetic_suite(output: str | Path, *, suite_id: str, attempts: list[dict], execute) -> dict:
    """Run distinct synthetic case/arm pairs once, with no automatic resume.

    Every attempt must declare ``case_id``, ``arm`` and a JSON-serializable
    synthetic ``input``. Executor receives a deep copy of that attempt, and
    returns a traffic/follow-through suite result. The header hashes the input
    schedule but does not retain it or attest the executor's code or environment.
    An interruption inside an attempt leaves its terminal state and partial
    traces unknown; an existing journal path is never overwritten.
    """
    if not isinstance(suite_id, str) or not suite_id.strip() or not isinstance(attempts, list) or not attempts:
        raise ValueError('synthetic_suite_and_attempts_required')
    if getattr(execute, 'offline', False) is not True:
        raise ValueError('executor_must_explicitly_declare_offline')
    for row in attempts:
        if (not isinstance(row, dict) or set(row) != {'case_id', 'arm', 'input'}
                or not all(isinstance(row[k], str) and row[k].strip() for k in ('case_id', 'arm'))):
            raise ValueError('invalid_attempt')
    raw = json.dumps(attempts, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    schedule = [{'case_id': r['case_id'], 'arm': r['arm']} for r in attempts]
    if len({(r['case_id'], r['arm']) for r in schedule}) != len(schedule):
        raise ValueError('duplicate_attempt')
    run_id = str(uuid4())
    header = {'event': 'run_started', 'schema_version': 1, 'run_id': run_id,
              'experiment_kind': 'offline_synthetic_suite_plumbing', 'recorded_at': _timestamp(),
              'suite_id': suite_id, 'corpus_sha256': hashlib.sha256(raw).hexdigest(),
              'digest_scope': 'serialized attempt inputs and schedule; not corpus, oracle, or runtime',
              'schedule': schedule, 'limitations': [
                  'Fixture pass is not decision qualification, actual delivery or production safety.',
                  'No independent source or execution provenance attestation.',
                  'Fixture grade is a trusted adapter report, not recomputed by this journal.',
                  'Input digest does not preserve input bytes; retain synthetic inputs separately.',
                  'Partial traces during interrupted attempts remain unknown.',
                  'Repeated case/arm pairs need distinct case identifiers.',
                  'Callback declaration is not an OS/network sandbox.']}
    output_path = Path(output)
    with output_path.open('x', encoding='utf-8') as stream:
        _append(stream, header)
        _fsync_parent(output_path)
        for attempt in attempts:
            attempt_id = json.dumps([attempt['case_id'], attempt['arm']], ensure_ascii=False, separators=(',', ':'))
            _append(stream, {'event': 'attempt_started', 'run_id': run_id, 'recorded_at': _timestamp(),
                             'attempt_id': attempt_id, 'case_id': attempt['case_id'], 'arm': attempt['arm']})
            started = monotonic()
            try:
                outcome = normalize_suite_result(execute(deepcopy(attempt)))
                json.dumps(outcome, allow_nan=False)
                if not _valid_outcome(outcome):
                    raise ValueError('invalid_normalized_outcome')
            except Exception as exc:
                outcome = {'availability': 'error', 'passed': None, 'reason': 'runner_error:' + type(exc).__name__,
                           'metrics': None, 'oracle': None, 'events': None}
            _append(stream, {'event': 'attempt_finished', 'run_id': run_id, 'recorded_at': _timestamp(),
                             'elapsed_seconds': monotonic() - started, 'attempt_id': attempt_id, 'outcome': outcome})
        _append(stream, {'event': 'run_finished', 'run_id': run_id, 'recorded_at': _timestamp(),
                         'attempt_count': len(schedule)})
    return inspect_journal(output_path)
