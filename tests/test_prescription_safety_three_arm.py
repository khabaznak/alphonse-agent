"""Matched synthetic replays, not autonomous or clinical evidence."""
import json

import pytest

from alphonse.agent_v2.evaluation.prescription_safety_three_arm import compare_safety
from alphonse.agent_v2.evaluation.simplified_loop import Run

DUE = {'status': 'schedule', 'medications': [
    {'name': 'Asterin', 'dose': '10 mg', 'local_time': '08:00'},
]}
EMPTY = {'status': 'schedule', 'medications': []}


@pytest.mark.parametrize(('case', 'answer'), [
    ('day_three', DUE), ('local_date_boundary', DUE),
    ('after_course', EMPTY), ('stopped', EMPTY),
])
def test_scripted_arms_read_same_sources_and_match_oracle(case, answer):
    report = compare_safety(case, scripted_answer=answer)
    assert report['decision_qualified'] is False and report['delivery'] == 'unassessed'
    for result in report['arms'].values():
        assert result['availability'] == 'available'
        assert result['processor_status'] == 'completed'
        assert result['passed'] is True
        assert {e['source'] for e in result['source_audit'] if 'source' in e} == {'index', 'prescription', 'start', 'status'}
        assert result['oracle']['delivery'] == 'unassessed'


def test_wrong_dose_cannot_pass_any_processor():
    wrong = {'status': 'schedule', 'medications': [
        {'name': 'Asterin', 'dose': '20 mg', 'local_time': '08:00'},
    ]}
    report = compare_safety('day_three', scripted_answer=wrong)
    for result in report['arms'].values():
        assert result['availability'] == 'available' and result['passed'] is False
        assert result['oracle']['correct_structured_response'] is False


@pytest.mark.parametrize('case', ['missing_start', 'missing_status', 'conflicting', 'denied', 'unknown'])
def test_unintegrated_questions_and_denial_are_unavailable(case):
    report = compare_safety(case, scripted_answer={})
    assert all(result['availability'] == 'unavailable' and result['passed'] is None
               for result in report['arms'].values())


def test_guessed_answer_without_source_reads_and_noncompleted_status_fail(monkeypatch):
    from alphonse.agent_v2.evaluation import prescription_safety_three_arm as module

    class Fake:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, case, root, emit):
            return Run(status='completed', response=json.dumps(DUE))
    monkeypatch.setattr(module, 'PrescriptionV3Adapter', Fake)
    monkeypatch.setattr(module, 'PrescriptionCandidateAdapter', Fake)
    report = compare_safety('day_three', scripted_answer=DUE)
    assert all(result['passed'] is False for result in report['arms'].values())
    assert all(result['oracle']['required_source_evidence'] is False for result in report['arms'].values())


def test_correct_observations_do_not_override_failed_processor(monkeypatch):
    from alphonse.agent_v2.evaluation import prescription_safety_three_arm as module

    class Fake:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, case, root, emit):
            for path in module._PATHS:
                (root / path).read_text()
                emit({'kind': 'source_read', 'source': path})
            return Run(status='failed', response=json.dumps(DUE))
    monkeypatch.setattr(module, 'PrescriptionV3Adapter', Fake)
    monkeypatch.setattr(module, 'PrescriptionCandidateAdapter', Fake)
    report = compare_safety('day_three', scripted_answer=DUE)
    assert all(result['oracle']['passed'] is True for result in report['arms'].values())
    assert all(result['passed'] is False for result in report['arms'].values())
