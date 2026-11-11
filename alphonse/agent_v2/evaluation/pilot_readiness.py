"""Non-executing local prerequisites inspection for the synthetic pilot.

This is neither a launcher nor authorization. It executes no binaries, opens no
Docker sockets, and sends no model requests. Candidate file presence is not an
identity, version, availability or trust check. All launch blockers remain.
"""
from __future__ import annotations

import json
from pathlib import Path


def inspect_local(*, codex_path: Path = Path('/opt/homebrew/bin/codex'),
                  docker_path: Path = Path('/usr/local/bin/docker')) -> dict:
    """Inspect fixed candidate paths, without consulting PATH or executing them."""
    def present(path: Path) -> str:
        try:
            return 'present_unverified' if path.is_file() else 'not_found_at_candidate_path'
        except OSError:
            return 'unavailable_or_unverified'
    return {
        'kind': 'nonexecuting_local_prerequisite_inspection',
        'observed': {'codex_candidate': present(codex_path), 'docker_candidate': present(docker_path),
                     'docker_daemon': 'unverified'},
        'dispatch_authorized': False,
        'ready_to_run': False,
        'blockers': [
            'executable_identity_versions_and_integrity_unverified',
            'model_account_entitlement_and_exact_identity_unverified',
            'provider_usage_and_prices_unverified',
            'task_wide_nonbypassable_spend_control_unverified',
            'isolated_environment_and_network_policy_unverified',
            'fixture_attribution_and_independent_oracles_unverified',
            'frozen_protocol_inputs_and_runtime_provenance_unverified',
            'retention_and_provider_data_policy_unverified',
            'explicit_live_evaluation_approval_missing',
        ],
        'limitations': [
            'Presence at the candidate path does not establish a trusted executable or version.',
            'This checks only two fixed macOS paths and does not establish Docker daemon availability.',
            'This command does not execute programs, connect to a daemon or authorize an evaluation.',
        ],
    }


def main() -> int:
    print(json.dumps(inspect_local(), indent=2))
    return 1  # Mandatory independent checks remain unverified.


if __name__ == '__main__':
    raise SystemExit(main())
