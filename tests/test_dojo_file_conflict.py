import hashlib

from alphonse.agent_v2.evaluation.dojo_file_conflict import FILES, grade
from alphonse.agent_v2.evaluation.dojo_household_shopping import fixture_bytes


def observation():
    return {'status': 'observed', 'scan_complete': True, 'invalid_json': [],
            'source_hashes': {k: hashlib.sha256(fixture_bytes(v)).hexdigest() for k, v in FILES.items()},
            'new_json': {'new-list.json': {'rice': 2, 'tomatoes': 3}}}


def test_conflict_case_checks_new_file_and_preservation():
    value = observation()
    assert grade(value)['effect_passed'] is True
    value['source_hashes']['shopping-list.json'] = 'changed'
    assert grade(value)['effect_passed'] is False
    value = observation()
    value['new_json']['new-list.json'] = {'rice': 2, 'tomatoes': 4}
    assert grade(value)['effect_passed'] is False


def test_accurate_zero_for_covered_meal_plan_item_is_not_invented_failure():
    value = observation()
    value['new_json']['new-list.json']['beans'] = 0
    scored = grade(value)
    assert scored['effect_passed'] is True
    assert scored['matching_files_with_zero_amounts'] == ['new-list.json']
    value['new_json']['new-list.json']['unrelated'] = 0
    assert grade(value)['effect_passed'] is False


def test_incomplete_observation_does_not_claim_success_or_unseen_failure():
    value = observation()
    value['source_hashes'] = None
    assert grade(value)['effect_passed'] is None
    value = observation()
    value['scan_complete'] = False
    assert grade(value)['effect_passed'] is None
    assert grade({'status': 'unavailable'})['effect_passed'] is None
