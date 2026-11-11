"""Freeze a *description* of a decision experiment, without running it.

The returned SHA-256 must be anchored independently before exposure or execution.
A digest detects changes relative to that anchor; it supplies neither secrecy,
authorization, independent review nor proof the declared inputs were used. The
caller must retain the corresponding corpus and artifact bytes privately and
compare them at run time. This module does not make a run decision-grade.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any

_DIGEST = re.compile(r'^[0-9a-f]{64}$')
_ARMS = {'v3', 'targeted_v3', 'single_loop', 'external'}
_CONFIG_FIELDS = {
    'revision', 'source_sha256', 'prompt_sha256', 'tool_catalog_sha256',
    'authorization_policy_sha256', 'provider', 'model', 'runtime',
    'decoding', 'limits', 'timeout_seconds', 'timezone', 'environment',
    'network', 'capabilities',
}


def canonical_digest(value: Any) -> str:
    """Hash JSON data deterministically; reject non-JSON floats and objects."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _schedule(case_ids: list[str], arms: list[str], repeats: int, seed: str) -> list[dict[str, Any]]:
    # Sort by a stable cryptographic digest instead of relying on interpreter-
    # specific random implementations. Pair every arm on each case/replicate.
    rows = []
    for case_id in case_ids:
        for repetition in range(1, repeats + 1):
            order = sorted(arms, key=lambda arm: canonical_digest([seed, case_id, repetition, arm]))
            for position, arm in enumerate(order):
                rows.append({'case_id': case_id, 'repetition': repetition,
                             'arm': arm, 'position': position})
    return rows


def freeze_manifest(*, protocol_sha256: str, corpus_sha256: str, oracle_sha256: str,
                    case_ids: list[str], arms: dict[str, dict[str, Any]],
                    experiment_kind: str, seed: str, repetitions: int = 5) -> dict[str, Any]:
    """Create a content-bound plan. Configurations may explicitly be unavailable.

    Corpus and oracle hashes are commitments only; no sensitive case content is
    embedded. Execution requires a separate independent corpus/coverage review.
    A configuration marked unavailable has exactly ``status`` and ``reason``.
    Available configurations must pin every required field; values are never
    inferred from ambient credentials or process environment.
    """
    if experiment_kind not in {'development', 'heldout'}:
        raise ValueError('invalid_experiment_kind')
    if not all(_is_digest(x) for x in (protocol_sha256, corpus_sha256, oracle_sha256)):
        raise ValueError('invalid_artifact_digest')
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError('missing_schedule_seed')
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('invalid_repetitions')
    if (not isinstance(case_ids, list) or not case_ids or
            any(not isinstance(x, str) or not x.strip() for x in case_ids) or
            len(case_ids) != len(set(case_ids))):
        raise ValueError('invalid_case_ids')
    if not isinstance(arms, dict) or set(arms) != _ARMS:
        raise ValueError('required_arms_missing_or_extra')
    for config in arms.values():
        if not isinstance(config, dict):
            raise ValueError('invalid_arm_configuration')
        if config.get('status') == 'unavailable':
            if set(config) != {'status', 'reason'} or not isinstance(config['reason'], str) or not config['reason'].strip():
                raise ValueError('invalid_unavailable_arm')
        elif config.get('status') == 'available':
            if not _CONFIG_FIELDS <= set(config):
                raise ValueError('incomplete_arm_configuration')
            if any(config[k] is None or config[k] == '' or config[k] == {} or config[k] == [] for k in _CONFIG_FIELDS):
                raise ValueError('empty_arm_configuration')
            if not all(_is_digest(config[k]) for k in ('source_sha256', 'prompt_sha256', 'tool_catalog_sha256', 'authorization_policy_sha256')):
                raise ValueError('invalid_arm_artifact_digest')
            timeout = config['timeout_seconds']
            # An unbounded adapter may be described truthfully for development
            # diagnostics, but cannot be frozen as a confirmatory arm.
            unbounded_development = experiment_kind == 'development' and timeout == 'unbounded'
            if not unbounded_development and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float('inf')):
                raise ValueError('invalid_timeout')
        else:
            raise ValueError('invalid_arm_status')
    plan = {'schema_version': 1, 'experiment_kind': experiment_kind,
            'protocol_sha256': protocol_sha256, 'corpus_sha256': corpus_sha256,
            'oracle_sha256': oracle_sha256, 'case_ids': list(case_ids),
            'arms': deepcopy(arms), 'seed': seed, 'repetitions': repetitions,
            'schedule': _schedule(case_ids, sorted(arms), repetitions, seed)}
    return {**plan, 'manifest_sha256': canonical_digest(plan)}


def validate_manifest(manifest: dict[str, Any], *, trusted_manifest_sha256: str,
                      protocol: bytes, corpus: bytes, oracle: bytes,
                      current_arms: dict[str, dict[str, Any]],
                      arm_artifacts: dict[str, dict[str, bytes]]) -> list[str]:
    """Verify manifest and actual artifact bytes against an independent anchor.

    Returns a list of blockers, not a decision or permission to run. Caller must
    determine whether the supplied artifacts and configuration actually describe
    the executing environment and cover all relevant files.
    Coverage, independent authorship, secrecy, authorization, and human review
    remain unverified and therefore always block confirmatory qualification.
    """
    if not _is_digest(trusted_manifest_sha256):
        raise ValueError('invalid_trusted_manifest_digest')
    if not isinstance(manifest, dict) or manifest.get('manifest_sha256') != trusted_manifest_sha256:
        raise ValueError('manifest_anchor_mismatch')
    unsigned = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if canonical_digest(unsigned) != trusted_manifest_sha256:
        raise ValueError('manifest_hash_mismatch')
    try:
        rebuilt = freeze_manifest(protocol_sha256=manifest['protocol_sha256'],
            corpus_sha256=manifest['corpus_sha256'], oracle_sha256=manifest['oracle_sha256'],
            case_ids=manifest['case_ids'], arms=manifest['arms'],
            experiment_kind=manifest['experiment_kind'], seed=manifest['seed'],
            repetitions=manifest['repetitions'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid_manifest') from exc
    if rebuilt != manifest:
        raise ValueError('manifest_structure_or_schedule_mismatch')
    if (bytes_digest(protocol) != manifest['protocol_sha256'] or
            bytes_digest(corpus) != manifest['corpus_sha256'] or
            bytes_digest(oracle) != manifest['oracle_sha256']):
        raise ValueError('frozen_artifact_changed')
    if current_arms != manifest['arms']:
        raise ValueError('frozen_arm_configuration_changed')
    available = {name for name, cfg in manifest['arms'].items() if cfg['status'] == 'available'}
    if not isinstance(arm_artifacts, dict) or set(arm_artifacts) != available:
        raise ValueError('missing_arm_artifacts')
    artifact_names = ('source', 'prompt', 'tool_catalog', 'authorization_policy')
    for arm in available:
        artifacts = arm_artifacts[arm]
        if not isinstance(artifacts, dict) or set(artifacts) != set(artifact_names):
            raise ValueError('missing_arm_artifacts')
        for name in artifact_names:
            if not isinstance(artifacts[name], bytes) or bytes_digest(artifacts[name]) != manifest['arms'][arm][name + '_sha256']:
                raise ValueError('frozen_arm_artifact_changed')
    blockers = ['coverage_and_independent_oracles_unverified',
                'case_ids_and_runtime_provenance_unverified',
                'independent_authorship_and_sequestration_unverified',
                'authorization_and_approval_unverified', 'human_review_unverified']
    if manifest['experiment_kind'] != 'heldout':
        blockers.append('development_only')
    if len(manifest['case_ids']) < 24 or manifest['repetitions'] < 5:
        blockers.append('minimum_heldout_sample_not_met')
    blockers.extend('arm_unavailable:' + arm for arm, cfg in manifest['arms'].items()
                    if cfg['status'] == 'unavailable')
    return blockers
