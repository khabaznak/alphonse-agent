"""Integrity checks for pre-registration; not a live experiment."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from alphonse.agent_v2.evaluation.decision_manifest import (
    bytes_digest, canonical_digest, freeze_manifest, validate_manifest,
)

PROTOCOL = b'protocol'
CORPUS = b'synthetic corpus'
ORACLE = b'synthetic independent oracle'
HASH = bytes_digest(b'configuration artifact')


def config():
    return {'status': 'available', 'revision': 'abc123', 'source_sha256': HASH,
            'prompt_sha256': HASH, 'tool_catalog_sha256': HASH,
            'authorization_policy_sha256': HASH, 'provider': 'offline',
            'model': 'scripted', 'runtime': 'python', 'decoding': {'temperature': 0},
            'limits': {'max_calls': 4}, 'timeout_seconds': 60, 'timezone': 'UTC',
            'environment': {'type': 'disposable'}, 'network': 'none',
            'capabilities': ['read']}


def arms():
    return {'v3': config(), 'targeted_v3': {'status': 'unavailable', 'reason': 'not built'},
            'single_loop': config(), 'external': {'status': 'unavailable', 'reason': 'not measured'}}


def make(**kwargs):
    args = dict(protocol_sha256=bytes_digest(PROTOCOL), corpus_sha256=bytes_digest(CORPUS),
                oracle_sha256=bytes_digest(ORACLE), case_ids=['case-a', 'case-b'],
                arms=arms(), experiment_kind='development', seed='seed', repetitions=5)
    args.update(kwargs)
    return freeze_manifest(**args)


def check(manifest, **kwargs):
    args = dict(trusted_manifest_sha256=manifest['manifest_sha256'], protocol=PROTOCOL,
                corpus=CORPUS, oracle=ORACLE, current_arms=arms(),
                arm_artifacts={arm: {key: b'configuration artifact' for key in
                    ('source', 'prompt', 'tool_catalog', 'authorization_policy')}
                    for arm in ('v3', 'single_loop')})
    args.update(kwargs)
    return validate_manifest(manifest, **args)


def test_reproducible_paired_schedule_and_explicit_blockers():
    manifest = make()
    assert manifest == make()
    assert len(manifest['schedule']) == 2 * 5 * 4
    for case in manifest['case_ids']:
        for repetition in range(1, 6):
            rows = [r for r in manifest['schedule'] if r['case_id'] == case and r['repetition'] == repetition]
            assert {r['arm'] for r in rows} == set(arms())
            assert [r['position'] for r in rows] == list(range(4))
    blockers = check(manifest)
    assert 'development_only' in blockers
    assert 'arm_unavailable:external' in blockers
    assert 'independent_authorship_and_sequestration_unverified' in blockers


def test_detect_manifest_edit_even_if_self_hash_is_recomputed():
    original = make()
    altered = deepcopy(original)
    altered['seed'] = 'new seed'
    altered['manifest_sha256'] = canonical_digest({k: v for k, v in altered.items() if k != 'manifest_sha256'})
    with pytest.raises(ValueError, match='manifest_anchor_mismatch'):
        check(altered, trusted_manifest_sha256=original['manifest_sha256'])
    with pytest.raises(ValueError, match='structure_or_schedule'):
        check(altered)


@pytest.mark.parametrize('field,value', [('protocol', b'changed'), ('corpus', b'changed'), ('oracle', b'changed')])
def test_detect_changed_artifacts(field, value):
    with pytest.raises(ValueError, match='frozen_artifact_changed'):
        check(make(), **{field: value})


def test_detect_changed_runtime_configuration():
    current = arms()
    current['v3']['model'] = 'different'
    with pytest.raises(ValueError, match='frozen_arm_configuration_changed'):
        check(make(), current_arms=current)


def test_detect_changed_or_missing_arm_artifacts():
    artifacts = {arm: {key: b'configuration artifact' for key in
                 ('source', 'prompt', 'tool_catalog', 'authorization_policy')}
                 for arm in ('v3', 'single_loop')}
    artifacts['v3']['prompt'] = b'changed'
    with pytest.raises(ValueError, match='frozen_arm_artifact_changed'):
        check(make(), arm_artifacts=artifacts)
    with pytest.raises(ValueError, match='missing_arm_artifacts'):
        check(make(), arm_artifacts={})


def test_reject_missing_pins_duplicate_cases_and_unknown_arm():
    missing = arms()
    del missing['v3']['prompt_sha256']
    with pytest.raises(ValueError, match='incomplete_arm'):
        make(arms=missing)
    with pytest.raises(ValueError, match='invalid_case_ids'):
        make(case_ids=['a', 'a'])
    with pytest.raises(ValueError, match='required_arms'):
        make(arms={'v3': config()})


def test_digest_does_not_claim_complete_heldout_eligibility():
    plan = make(experiment_kind='heldout', case_ids=[f'c{i}' for i in range(24)])
    blockers = check(plan)
    assert 'minimum_heldout_sample_not_met' not in blockers
    assert 'coverage_and_independent_oracles_unverified' in blockers
    assert 'authorization_and_approval_unverified' in blockers
    assert 'arm_unavailable:targeted_v3' in blockers


def test_development_example_matches_artifacts_and_remains_blocked():
    root = Path(__file__).resolve().parents[1]
    fixtures = root / 'tests/fixtures/single_loop_comparison'
    plan = json.loads((fixtures / 'prescription_schedule_development_plan_v1.json').read_text())
    source = (fixtures / 'prescription_schedule_development_v1.json').read_bytes()
    blockers = validate_manifest(plan,
        trusted_manifest_sha256='7b3d33e5998d1bc9b87fc94e3dd5cb4ee389c635ec02e49e645e4e9a24d29559',
        protocol=(root / 'docs/v3-hierarchical-capd/27-agent-architecture-decision-protocol.md').read_bytes(),
        corpus=source, oracle=source, current_arms=plan['arms'], arm_artifacts={})
    assert 'development_only' in blockers
    assert sum(x.startswith('arm_unavailable:') for x in blockers) == 4
