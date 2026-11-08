"""Run bounded model-only synthetic browser-fill comparisons.

The model receives synthetic field names and target identifiers, never the
protected values or page contents. Browser operations run on a disposable
in-memory backend. No Jev API is called; tactical reviews are scripted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from alphonse.agent_v2.core.inference import InferenceRouter, ModelProfile, OpenAICodexProvider, OpenAICodexProviderConfig
from alphonse.agent_v2.core.intelligence.v3 import executor
from alphonse.agent_v2.database import default_database_path
from alphonse.agent_v2.evaluation import browser_handoff_comparison
from alphonse.agent_v2.evaluation.planning_multifield_live import _model_only_wrapper_source
from alphonse.agent_v2.evaluation.resourceful_live import read_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex-cli', type=Path, required=True)
    parser.add_argument('--settings-db', type=Path, default=default_database_path())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('both', 'single', 'sequence'), default='both')
    parser.add_argument('--qualify-phase', action='store_true',
                        help='Include an independent disposable fixture tool that can qualify the field phase.')
    args = parser.parse_args()
    llm_settings, _ = read_settings(args.settings_db)
    if llm_settings.provider_key != 'openai_codex' or not llm_settings.model_id:
        raise ValueError('configured_codex_provider_unavailable')
    report = {'version': 'browser-sequence-live-v1', 'model': llm_settings.model_id,
              'executor_sha256': hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest(),
              'harness_sha256': hashlib.sha256(Path(browser_handoff_comparison.__file__).read_bytes()).hexdigest(),
              'limitations': ['Synthetic target IDs are given in the task; this does not qualify real browser grounding.',
                  'The backend is disposable and in-memory, not Chrome.',
                  'System One reviews are scripted and no Jev API is called.',
                  'The single-action control forbids sequences in its prompt and rejects them if returned.',
                  'At most 12 compositions per arm; subscription cost is not measured.',
                  'One run per arm is exploratory, not a reliability estimate.'], 'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='browser-sequence-model-only-') as directory:
        wrapper = Path(directory) / 'codex-model-only'
        wrapper.write_text(_model_only_wrapper_source(args.codex_cli.resolve()))
        wrapper.chmod(0o700)
        llm = InferenceRouter(provider=OpenAICodexProvider(OpenAICodexProviderConfig(
            cli_bin=str(wrapper), timeout_seconds=60, ephemeral=True, require_explicit_model=True)),
            default_profile=ModelProfile(provider=llm_settings.provider_key, model=llm_settings.model_id,
                                         profile_id='synthetic-browser-sequence'))
        arms = (False, True) if args.arm == 'both' else (args.arm == 'sequence',)
        for sequence in arms:
            row = browser_handoff_comparison.run_case(control=False, sequence=sequence, inference=llm,
                                                       qualify=args.qualify_phase)
            report['results'].append(row)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    # The optional independent oracle qualifies only the synthetic field phase.
    # Blocked or errored phases must not exit as a pass.
    return 0 if len(report['results']) == len(arms) and all(
        row.get('phase_completion') == 'qualified' and row.get('effect_oracle_pass') for row in report['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
