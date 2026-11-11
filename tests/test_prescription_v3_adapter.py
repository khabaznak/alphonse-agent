"""Synthetic replays only: no autonomous or clinical performance claims."""
import json
from pathlib import Path

from alphonse.agent_v2.evaluation.prescription_v3_adapter import PrescriptionV3Adapter
from alphonse.agent_v2.evaluation.prescription_three_arm import compare_prescriptions
from alphonse.agent_v2.evaluation.simplified_loop import _run, load_frozen

CORPUS = Path(__file__).parent / 'fixtures/single_loop_comparison/prescription_schedule_development_v1.json'
PATHS = {'household/index.md', 'household/prescription.md', 'household/treatment-start.md'}


def answer(*names):
    return json.dumps({'due_medications': list(names)})


def test_three_scripted_arms_run_on_same_synthetic_cases():
    cases, _ = load_frozen(CORPUS)
    scripts = {case['id']: answer(*([] if 'after-course' in case['id'] else ['Asterin']))
               for case in cases['cases']}
    report = compare_prescriptions(CORPUS, scripted_answers=scripts)
    assert report['experiment_kind'] == 'offline_scripted_prescription_three_arm_plumbing'
    for row in report['cases']:
        for result in row['arms'].values():
            assert result['availability'] == 'available'
            assert result['passed'] is True
            assert set(result['oracle']['source_read_counts']) == PATHS
            assert result['metrics']['provider_input_tokens'] is None


def test_wrong_presupplied_answer_fails_independent_oracle_in_every_arm():
    case = load_frozen(CORPUS)[0]['cases'][0]
    report = compare_prescriptions(CORPUS, scripted_answers={case['id']: answer('Borelin')})
    for result in report['cases'][0]['arms'].values():
        assert result['status'] == 'completed'
        assert result['passed'] is False
        assert result['oracle']['checks']['medication_schedule'] is False
    for row in report['cases'][1:]:
        for result in row['arms'].values():
            assert result['availability'] == 'unavailable'
            assert result['passed'] is None


def test_denies_unexpected_fixture_and_does_not_run_oracle_input():
    case = load_frozen(CORPUS)[0]['cases'][0]
    adapter = PrescriptionV3Adapter(projection_enabled=False, scripted_answers={case['id']: answer('Asterin')})
    case = dict(case, fixture={'files': {'unexpected.txt': 'anything'}})
    result = _run(case, adapter)
    assert result['availability'] == 'unavailable'
