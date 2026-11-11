"""Three-arm offline fixture runner, not a model or production benchmark.

The V3 arms run in separate interpreters with the experimental projection flag
explicitly off/on. All providers and decisions are scripted. Fixtures and output
may contain synthetic task text. The runner does not enforce an OS/network sandbox.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .candidate_comparison import SingleLoopFixtureAdapter
from .isolated_v3_arm import IsolatedV3ReadArm
from .simplified_loop import Adapter, _run, load_frozen


def compare_three(corpus_path: str | Path, *, arms: Mapping[str, Adapter | None] | None = None) -> dict:
    """Run every arm in its own fresh fixture root and preserve all outcomes.

    An optional mapping is for synthetic test adapters only. `_run` enforces the
    explicit offline declaration and independently grades each arm. No missing
    case, oracle failure, or unsupported arm is silently excluded.
    """
    if arms is None:
        arms = {
            'current_v3': IsolatedV3ReadArm(projection_enabled=False),
            'targeted_v3_projection': IsolatedV3ReadArm(projection_enabled=True),
            'single_loop': SingleLoopFixtureAdapter(),
        }
    if not arms or any(not isinstance(name, str) or not name for name in arms):
        raise ValueError('arm_names_required')
    corpus, digest = load_frozen(corpus_path)
    rows = []
    for case in corpus['cases']:
        outcomes = {}
        for name, arm in arms.items():
            try:
                outcomes[name] = _run(case, arm)
            except Exception as exc:
                # Infrastructure errors outside the adapter's own guarded path
                # are inconclusive: actions and observations may be incomplete.
                # Do not expose exception messages or infer no effects occurred.
                outcomes[name] = {
                    'availability': 'error', 'passed': None,
                    'reason': 'runner_error:' + type(exc).__name__,
                    'metrics': None, 'oracle': None, 'events': None,
                }
        rows.append({'case_id': case['id'], 'arms': outcomes})
    return {
        'schema_version': 1,
        'simulation': True,
        'experiment_kind': 'offline_scripted_three_arm_plumbing',
        'corpus_sha256': digest,
        'split': corpus['split'],
        'arm_names': list(arms),
        'cases': rows,
        'limitations': [
            'Scripted providers and Jev do not establish autonomous behavior.',
            'Fixture timings are not user-facing model latency; usage is unknown unless reported.',
            'The V3 projection arm is an existing experimental flag, not a qualified architecture.',
            'Each arm receives a separate temporary fixture; this is not an OS/network sandbox.',
            'Missing or unsupported cases are not passes and no decision gate is evaluated.',
            'Receipts are written after the full run; interruption can lose all in-memory rows.',
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path, help='new synthetic JSON receipt; never overwritten')
    args = parser.parse_args(argv)
    result = compare_three(args.corpus)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return int(any(arm['passed'] is not True for row in result['cases'] for arm in row['arms'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
