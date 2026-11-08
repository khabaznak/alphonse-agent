"""Budgeted model-only comparison of independent synthetic reads."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile

from alphonse.agent_v2.core.inference import InferenceRouter, ModelProfile, OpenAICodexProvider, OpenAICodexProviderConfig
from alphonse.agent_v2.core.intelligence.v3 import executor
from alphonse.agent_v2.database import default_database_path
from alphonse.agent_v2.evaluation import planning_independent_reads
from alphonse.agent_v2.evaluation.planning_multifield_live import _model_only_wrapper_source
from alphonse.agent_v2.evaluation.resourceful_live import BudgetTransport, read_settings
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex-cli', type=Path, required=True)
    parser.add_argument('--settings-db', type=Path, default=default_database_path())
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    llm_settings, jev_settings = read_settings(args.settings_db)
    if not jev_settings.enabled or not jev_settings.api_key or llm_settings.provider_key != 'openai_codex':
        raise ValueError('configured_live_providers_unavailable')
    budget = BudgetTransport(args.ledger, limit='0.05')
    jev = JevCriterionDecisionProvider(replace(jev_settings, model='jev-1.14.0'), transport=budget)
    report = {'version': 'independent-reads-live-v1', 'model': llm_settings.model_id, 'jev_model': 'jev-1.14.0',
        'harness_sha256': hashlib.sha256(Path(planning_independent_reads.__file__).read_bytes()).hexdigest(),
        'executor_sha256': hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest(),
        'limitations': ['One trial per shape; this is not a statistical comparison.',
            'Synthetic in-memory reads do not qualify browser, mutation or real household workflows.',
            'ChatGPT subscription usage has no measured per-call dollar cost.',
            'Single-action control changes the composition prompt and rejects a sequence if returned.',
            'The Jev ledger is capped at US$0.05; reuse the same ledger for repeated runs.'], 'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='planning-reads-model-only-') as directory:
        wrapper = Path(directory) / 'codex-model-only'
        wrapper.write_text(_model_only_wrapper_source(args.codex_cli.resolve()))
        wrapper.chmod(0o700)
        llm = InferenceRouter(provider=OpenAICodexProvider(OpenAICodexProviderConfig(
            cli_bin=str(wrapper), timeout_seconds=60, ephemeral=True, require_explicit_model=True)),
            default_profile=ModelProfile(provider=llm_settings.provider_key, model=llm_settings.model_id,
                                         profile_id='synthetic-independent-reads'))
        for sequence in (False, True):
            budget.local.case_id = 'synthetic-sequence' if sequence else 'synthetic-single'
            before = budget.charged
            row = planning_independent_reads.run_case('normal', sequence, inference=llm, system_one=jev)
            row['typesafe_trial_charge_usd'] = str(budget.charged - before)
            row['typesafe_cumulative_charge_usd'] = str(budget.charged)
            report['results'].append(row)
            report['typesafe_cumulative_charge_usd'] = str(budget.charged)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    return 0 if all(row.get('oracle_pass') for row in report['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
