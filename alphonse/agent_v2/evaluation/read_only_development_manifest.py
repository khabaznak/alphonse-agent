"""Describe and check the offline, three-arm read-only development fixture.

This snapshots selected local bytes and runtime metadata, not a full dependency
closure or execution attestation. It does not execute the comparison. The frozen
schedule is a proposed plan, not a record of actual trials or held-out evidence.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys

from .decision_manifest import bytes_digest, freeze_manifest, validate_manifest

ROOT = Path(__file__).resolve().parents[3]
CORPUS = 'tests/fixtures/single_loop_comparison/read_only_v2.json'
PROTOCOL = 'docs/v3-hierarchical-capd/27-agent-architecture-decision-protocol.md'
ADAPTER = 'alphonse/agent_v2/evaluation/candidate_comparison.py'
SCRIPT = 'alphonse/agent_v2/evaluation/deterministic_adapters.py'


def _inventory(root: Path) -> bytes:
    # Hash the contents, including untracked .py files, and their relative paths.
    # This is still not an exhaustive inventory of package data, installed
    # dependencies, native libraries, environment or OS resources.
    paths = sorted((root / 'alphonse/agent_v2').rglob('*.py'))
    paths.extend(root / name for name in ('pyproject.toml', 'requirements.txt'))
    inventory = {str(path.relative_to(root)): bytes_digest(path.read_bytes()) for path in paths}
    return json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()


def snapshot(root: Path = ROOT):
    corpus = (root / CORPUS).read_bytes()
    protocol = (root / PROTOCOL).read_bytes()
    parsed = json.loads(corpus)
    source = _inventory(root)
    adapter = (root / ADAPTER).read_bytes()
    script = (root / SCRIPT).read_bytes()
    # The oracle and candidate script both live inside this public corpus.
    prompt = json.dumps({'corpus_sha256': bytes_digest(corpus),
                         'script_sha256': bytes_digest(script),
                         'adapter_sha256': bytes_digest(adapter)}, sort_keys=True).encode()
    # The scripted V3 registry is defined in SCRIPT and wrapped in ADAPTER;
    # the candidate fixture's registry and policy are defined in ADAPTER.
    registry_policy = json.dumps({'adapter_sha256': bytes_digest(adapter),
                                  'script_sha256': bytes_digest(script)}, sort_keys=True).encode()
    artifacts = {name: {'source': source, 'prompt': prompt,
                       'tool_catalog': registry_policy, 'authorization_policy': registry_policy}
                 for name in ('v3', 'targeted_v3', 'single_loop')}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    arms = {}
    for name in artifacts:
        code = artifacts[name]
        common = {'status': 'available', 'revision': revision,
            **{key + '_sha256': bytes_digest(value) for key, value in code.items()},
            'provider': 'offline scripted doubles', 'model': 'no model call',
            'runtime': {'python': sys.version, 'platform': platform.platform()},
            'decoding': 'not applicable: scripted deterministic actions',
            'timezone': 'not pinned by adapter; not relevant to this fixture',
            'network': 'no provider calls; not an OS network sandbox',
            'capabilities': ['synthetic read', 'synthetic denial', 'pre-execution cancellation'],
            'environment': 'disposable fixture directory; dependencies and host environment not fully pinned'}
        if name == 'single_loop':
            common.update(timeout_seconds='unbounded',
                          limits='candidate TaskBudget fixture limits; no wall-clock process timeout')
        else:
            common.update(timeout_seconds=30, limits='isolated worker timeout; scripted V3 fixture limits',
                          projection_flag='on' if name == 'targeted_v3' else 'off')
        arms[name] = common
    arms['external'] = {'status': 'unavailable', 'reason': 'No measured or approved external agent'}
    manifest = freeze_manifest(protocol_sha256=bytes_digest(protocol), corpus_sha256=bytes_digest(corpus),
        oracle_sha256=bytes_digest(corpus), case_ids=[case['id'] for case in parsed['cases']],
        arms=arms, experiment_kind='development', seed='read-only-v2-development', repetitions=5)
    return manifest, protocol, corpus, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'verify'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--expected-digest', help='independently recorded manifest digest (required for verify)')
    args = parser.parse_args()
    plan, protocol, corpus, artifacts = snapshot()
    if args.action == 'freeze':
        with args.path.open('x', encoding='utf-8') as file:
            json.dump(plan, file, ensure_ascii=False, indent=2)
            file.write('\n')
        print('manifest_sha256=' + plan['manifest_sha256'])
        print('Store the digest independently; this command does not establish an anchor.')
    else:
        if not args.expected_digest:
            parser.error('--expected-digest is required for verify')
        manifest = json.loads(args.path.read_text())
        blockers = validate_manifest(manifest, trusted_manifest_sha256=args.expected_digest,
            protocol=protocol, corpus=corpus, oracle=corpus,
            current_arms=plan['arms'], arm_artifacts=artifacts)
        print(json.dumps({'integrity': 'matched', 'blockers': blockers}, indent=2))


if __name__ == '__main__':
    main()
