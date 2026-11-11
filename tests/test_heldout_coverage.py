"""Structural held-out coverage, not a claim of independently reviewed cases."""
from copy import deepcopy
from hashlib import sha256

import pytest

from alphonse.agent_v2.evaluation.heldout_coverage import ARMS, REQUIRED_TAGS, assess_coverage


def metadata():
    """Invented metadata only; no usable task or oracle appears here."""
    tags = sorted(REQUIRED_TAGS)
    cases = []
    for i in range(24):
        family = 'traffic' if i < 4 else 'prescription' if i < 8 else 'other'
        cases.append({'id': f'private-id-{i}', 'author': 'private-author',
            'family': family, 'feasible': True, 'tags': [tags[i % len(tags)]],
            'task_sha256': sha256(f'task-{i}'.encode()).hexdigest(),
            'oracle_sha256': sha256(f'oracle-{i}'.encode()).hexdigest(),
            'oracle_implemented': True, 'oracle_reviewed': True,
            'supported_arms': sorted(ARMS)})
    # Multiple requirements can be asserted for one case. This is only an
    # opaque structural test; it proves no actual scenario satisfies them.
    cases[0]['tags'] = tags
    return cases


def test_public_report_contains_only_aggregate_metadata_and_never_qualifies():
    cases = metadata()
    report = assess_coverage(cases)
    assert report['structural_blockers'] == []
    assert report['qualification'] == 'unassessed'
    assert report['case_count'] == 24
    assert report['feasible_prescription_cases'] == report['feasible_traffic_cases'] == 4
    assert 'authorship_and_sequestration' in report['always_unverified']
    serialized = repr(report)
    assert 'private-id-' not in serialized and 'private-author' not in serialized
    assert cases[0]['oracle_sha256'] not in serialized


def test_missing_cases_tags_oracles_and_arms_are_blockers_not_passes():
    cases = metadata()[:5]
    cases[0]['tags'] = ['conversation']
    cases[0]['oracle_implemented'] = False
    cases[0]['oracle_reviewed'] = False
    cases[0]['supported_arms'] = []
    report = assess_coverage(cases)
    assert {'minimum_case_count_not_met', 'prescription_feasible_minimum_not_met',
            'required_coverage_missing', 'oracle_implementation_incomplete',
            'oracle_review_incomplete', 'case_arm_support_incomplete'} <= set(report['structural_blockers'])
    assert report['cases_with_unimplemented_oracles'] == 1
    assert report['cases_with_unsupported_arms'] == 1
    assert report['qualification'] == 'unassessed'


def test_duplicate_case_or_task_never_counts_as_independent_coverage():
    cases = metadata()
    cases[-1]['id'] = cases[0]['id']
    with pytest.raises(ValueError, match='invalid_or_duplicate_case_id'):
        assess_coverage(cases)
    cases = metadata()
    cases[-1]['task_sha256'] = cases[0]['task_sha256']
    with pytest.raises(ValueError, match='duplicate_task_digest'):
        assess_coverage(cases)


@pytest.mark.parametrize('field,value', [
    ('family', 'unrecognized'), ('family', ['private-case']),
    ('family', {'private-case': True}), ('feasible', 1), ('tags', ['not_a_category']),
    ('tags', ['conversation', 'conversation']), ('oracle_implemented', 'yes'),
    ('supported_arms', ['external', 'external']), ('task_sha256', 'abc'),
])
def test_invalid_private_metadata_is_rejected_without_leaking_values(field, value):
    cases = deepcopy(metadata())
    cases[0][field] = value
    with pytest.raises(ValueError) as error:
        assess_coverage(cases)
    assert 'private-id' not in str(error.value)
    assert 'private-author' not in str(error.value)
    assert 'private-case' not in str(error.value)
