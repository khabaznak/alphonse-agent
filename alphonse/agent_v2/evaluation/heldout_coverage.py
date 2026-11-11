"""Private structural coverage check for a separately authored held-out set.

This checks evaluator-supplied metadata, not case quality, secrecy, independence,
executable oracles, runtime support, authorization or readiness to run. No case
text, paths, IDs, hashes, authors or answers are included in the returned report.
Run only on synthetic metadata within an evaluator-controlled environment.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

# Each term represents a distinct requirement from protocol #217. A case can
# cover more than one; coverage labels are assertions, not validated semantics.
REQUIRED_TAGS = frozenset({
    'spanish_authorized_mail', 'mail_zero', 'mail_unavailable', 'mail_denied',
    'conversation', 'multi_source', 'irrelevant_history',
    'traffic_ticket', 'traffic_no_ticket', 'traffic_missing_identity',
    'traffic_changed_form', 'traffic_session', 'traffic_verification_block',
    'prescription_memory', 'prescription_dose_time', 'prescription_time_window',
    'prescription_missing', 'prescription_conflicting', 'prescription_stopped',
    'prescription_denied', 'multi_step_mutation', 'ambiguity', 'retryable_fault',
    'authorized_fallback', 'permission_denial', 'cancellation', 'restart',
    'uncertain_effect', 'necessary_inspection', 'notification_failed',
    'notification_unconfirmed',
})
FAMILIES = frozenset({'mail', 'conversation', 'traffic', 'prescription', 'other'})
ARMS = frozenset({'v3', 'targeted_v3', 'single_loop', 'external'})
_DIGEST = re.compile(r'^[0-9a-f]{64}$')
_FIELDS = {'id', 'author', 'family', 'feasible', 'tags', 'task_sha256',
           'oracle_sha256', 'oracle_implemented', 'oracle_reviewed', 'supported_arms'}


def assess_coverage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate blockers only, without case-specific or secret material.

    Fields, counts and digests are self-reported. An empty blocker list would
    still not qualify a comparison. Independently assessed authorship,
    separation, source/answer independence, case quality, authorization,
    environment and trace integrity always remain unverified.
    """
    if not isinstance(cases, list):
        raise ValueError('cases_must_be_a_list')
    seen_ids: set[str] = set()
    seen_tasks: set[str] = set()
    counts: Counter[str] = Counter()
    feasible_families: Counter[str] = Counter()
    unimplemented = unreviewed = unsupported = 0
    for case in cases:
        if not isinstance(case, dict) or set(case) != _FIELDS:
            raise ValueError('invalid_private_case_metadata')
        if (not isinstance(case['id'], str) or not case['id'].strip()
                or case['id'] in seen_ids):
            raise ValueError('invalid_or_duplicate_case_id')
        seen_ids.add(case['id'])
        if not isinstance(case['author'], str) or not case['author'].strip():
            raise ValueError('missing_author')
        if (not isinstance(case['family'], str) or case['family'] not in FAMILIES
                or type(case['feasible']) is not bool):
            raise ValueError('invalid_family_or_feasibility')
        if not all(isinstance(case[name], str) and _DIGEST.fullmatch(case[name])
                   for name in ('task_sha256', 'oracle_sha256')):
            raise ValueError('invalid_private_case_digest')
        if case['task_sha256'] in seen_tasks:
            raise ValueError('duplicate_task_digest')
        seen_tasks.add(case['task_sha256'])
        tags, arms = case['tags'], case['supported_arms']
        if (not isinstance(tags, list) or not tags or any(not isinstance(t, str) for t in tags)
                or len(tags) != len(set(tags)) or not set(tags) <= REQUIRED_TAGS):
            raise ValueError('invalid_coverage_tags')
        if (not isinstance(arms, list) or any(not isinstance(a, str) for a in arms)
                or len(arms) != len(set(arms)) or not set(arms) <= ARMS):
            raise ValueError('invalid_supported_arms')
        if type(case['oracle_implemented']) is not bool or type(case['oracle_reviewed']) is not bool:
            raise ValueError('invalid_oracle_assertion')
        counts.update(tags)
        if case['feasible']:
            feasible_families[case['family']] += 1
        unimplemented += not case['oracle_implemented']
        unreviewed += not case['oracle_reviewed']
        unsupported += set(arms) != ARMS

    missing_tags = sorted(REQUIRED_TAGS - counts.keys())
    blockers = []
    if len(cases) < 24:
        blockers.append('minimum_case_count_not_met')
    for family in ('prescription', 'traffic'):
        if feasible_families[family] < 4:
            blockers.append(f'{family}_feasible_minimum_not_met')
    if missing_tags:
        blockers.append('required_coverage_missing')
    if unimplemented:
        blockers.append('oracle_implementation_incomplete')
    if unreviewed:
        blockers.append('oracle_review_incomplete')
    if unsupported:
        blockers.append('case_arm_support_incomplete')
    return {
        'kind': 'self_reported_private_coverage',
        'case_count': len(cases),
        'feasible_prescription_cases': feasible_families['prescription'],
        'feasible_traffic_cases': feasible_families['traffic'],
        'missing_coverage_tags': missing_tags,
        'cases_with_unimplemented_oracles': unimplemented,
        'cases_without_oracle_review': unreviewed,
        'cases_with_unsupported_arms': unsupported,
        'structural_blockers': blockers,
        'qualification': 'unassessed',
        'always_unverified': ['authorship_and_sequestration', 'case_and_oracle_quality',
                              'actual_runtime_support', 'approval', 'independent_anchor'],
    }
