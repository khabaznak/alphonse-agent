"""The development snapshot is not a live comparison or held-out gate."""
from copy import deepcopy

import pytest

from alphonse.agent_v2.evaluation.decision_manifest import freeze_manifest, validate_manifest
from alphonse.agent_v2.evaluation.read_only_development_manifest import snapshot


def test_current_snapshot_covers_three_scripted_arms_and_one_unavailable():
    plan, protocol, corpus, artifacts = snapshot()
    assert len(plan['case_ids']) == 3
    assert plan['arms']['v3']['projection_flag'] == 'off'
    assert plan['arms']['targeted_v3']['projection_flag'] == 'on'
    assert plan['arms']['single_loop']['timeout_seconds'] == 'unbounded'
    assert plan['arms']['external']['status'] == 'unavailable'
    blockers = validate_manifest(plan, trusted_manifest_sha256=plan['manifest_sha256'],
        protocol=protocol, corpus=corpus, oracle=corpus, current_arms=plan['arms'],
        arm_artifacts=artifacts)
    assert {'development_only', 'arm_unavailable:external',
            'case_ids_and_runtime_provenance_unverified'} <= set(blockers)
    changed = deepcopy(artifacts)
    changed['v3']['source'] += b'changed'
    with pytest.raises(ValueError, match='frozen_arm_artifact_changed'):
        validate_manifest(plan, trusted_manifest_sha256=plan['manifest_sha256'],
            protocol=protocol, corpus=corpus, oracle=corpus, current_arms=plan['arms'],
            arm_artifacts=changed)


def test_confirmatory_manifest_cannot_claim_unbounded_timeout():
    plan, _, _, _ = snapshot()
    with pytest.raises(ValueError, match='invalid_timeout'):
        freeze_manifest(protocol_sha256=plan['protocol_sha256'], corpus_sha256=plan['corpus_sha256'],
            oracle_sha256=plan['oracle_sha256'], case_ids=plan['case_ids'], arms=plan['arms'],
            experiment_kind='heldout', seed=plan['seed'], repetitions=plan['repetitions'])
