"""Budgeted live synthetic form trials using a tool-disabled Codex CLI.

Only the in-memory fixture tool is exposed to the V3 executor. The CLI wrapper
turns off shell, apps, browser and MCP tools; this is an evaluation mode, not a
security boundary for production code. Inspect provider pricing before use.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile

from alphonse.agent_v2.core.inference import InferenceRouter, ModelProfile, OpenAICodexProvider, OpenAICodexProviderConfig
from alphonse.agent_v2.database import default_database_path
from alphonse.agent_v2.core.intelligence.v3 import executor
from alphonse.agent_v2.evaluation import planning_form_fixture, planning_multifield
from alphonse.agent_v2.evaluation.resourceful_live import BudgetTransport, read_settings
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider


def _model_only_wrapper_source(binary: Path) -> str:
    # The binary path is encoded as a Python literal, never shell-expanded.
    return ("#!/usr/bin/env python3\nimport os,sys\n"
        + 'binary=' + repr(str(binary)) + '\n'
        + "features=['shell_tool','unified_exec','code_mode_host','apps','browser_use','computer_use','view_image','in_app_browser','skill_search']\n"
        + "flags=[item for name in features for item in ('--disable',name)]\n"
        + "flags += ['-c','mcp_servers={}','--enable','skip_host_skill_discovery','-c','project_doc_max_bytes=0','-c','approval_policy=\\\"never\\\"','-s','read-only']\n"
        + "if len(sys.argv)<2 or sys.argv[1]!='exec': raise SystemExit('exec only')\n"
        + "os.execv(binary,[binary,'exec',*flags,*sys.argv[2:]])\n")


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
    report = {'version': 'multifield-live-runner-v1', 'model': llm_settings.model_id,
              'jev_model': 'jev-1.14.0',
              'harness_sha256': hashlib.sha256(Path(planning_multifield.__file__).read_bytes()).hexdigest(),
              'fixture_sha256': hashlib.sha256(Path(planning_form_fixture.__file__).read_bytes()).hexdigest(),
              'executor_sha256': hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest(),
              'limitations': ['One trial per shape; not a statistical comparison.',
                  'In-memory fixture checks do not qualify real browser or external effects.',
                  'ChatGPT subscription usage has no measured per-call dollar cost.',
                  'The Jev ledger is capped at US$0.05; reuse the same ledger for repeated runs.'],
              'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='planning-model-only-') as directory:
        wrapper = Path(directory) / 'codex-model-only'
        # The binary path is encoded as a Python literal, never shell-expanded.
        wrapper.write_text(_model_only_wrapper_source(args.codex_cli.resolve()))
        wrapper.chmod(0o700)
        llm = InferenceRouter(provider=OpenAICodexProvider(OpenAICodexProviderConfig(
            cli_bin=str(wrapper), timeout_seconds=60, ephemeral=True, require_explicit_model=True)),
            default_profile=ModelProfile(provider=llm_settings.provider_key, model=llm_settings.model_id,
                                         profile_id='synthetic-multifield'))
        for batch in (False, True):
            budget.local.case_id = 'synthetic-batch' if batch else 'synthetic-per-action'
            row = planning_multifield.run_case('normal', batch, inference=llm, system_one=jev)
            row['typesafe_cumulative_charge_usd'] = str(budget.charged)
            report['results'].append(row)
            report['typesafe_cumulative_charge_usd'] = str(budget.charged)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    return 0 if all(row.get('independent_oracle_pass') for row in report['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
