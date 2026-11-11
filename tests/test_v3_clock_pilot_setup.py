"""Operator scope and binding tests: no live provider or Jev calls."""
import json

import pytest

from alphonse.agent_v2.evaluation import v3_clock_pilot_setup as setup
from alphonse.agent_v2.evaluation.jev_attempt_meter import MeteredJevTransport
from alphonse.agent_v2.evaluation.jev_pilot_http import JevPilotHTTPTransport


@pytest.mark.parametrize('arm', ['v3', 'targeted_v3'])
def test_preview_reports_scope_without_credentials_or_network(arm, monkeypatch, capsys):
    monkeypatch.setattr(setup.getpass, 'getpass', lambda _: pytest.fail('unexpected key prompt'))
    assert setup.main(['--arm', arm]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['preview_only'] is True and result['arm'] == arm
    assert result['projection_enabled'] is (arm == 'targeted_v3')
    assert result['model'] == 'gpt-6-luna' and result['jev']['model_requested'] == 'jev-latest'
    assert result['limits']['model_calls'] == 10
    assert result['limits']['input_reservation_per_call'] == 65536
    assert result['limits']['output_tokens_per_call'] == 512
    assert result['proposed_budget_usd'] == '0.25'
    assert result['jev']['estimate_fits_proposed_budget'] is True
    assert 'not a hard provider dollar cap' in ' '.join(result['limitations'])
    assert 'single candidate run' not in ' '.join(result['limitations'])
    assert 'one V3 variant is not a complete matched comparison' in ' '.join(result['limitations'])


def test_missing_scope_or_small_budget_refuses_before_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(setup.getpass, 'getpass', lambda _: pytest.fail('unexpected key prompt'))
    with pytest.raises(SystemExit):
        setup.main(['--arm', 'v3', '--execute'])
    args = ['--arm', 'v3', '--execute', '--scope-approved', '--accept-estimate-uncertainty',
            '--isolation-reviewed', '--journal', str(tmp_path / 'run.jsonl'),
            '--openai-ledger', str(tmp_path / 'openai.db'), '--jev-ledger', str(tmp_path / 'jev.db')]
    with pytest.raises(SystemExit):
        setup.main([*args, '--jev-budget-usd', '0.00001'])
    assert not (tmp_path / 'openai.db').exists() and not (tmp_path / 'jev.db').exists()


def test_direct_api_execution_is_disabled_before_key_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(setup.getpass, 'getpass', lambda _: pytest.fail('unexpected key prompt'))
    monkeypatch.setattr(setup, 'requests_sender', lambda *args: pytest.fail('unexpected provider call'))
    journal = tmp_path / 'run.jsonl'
    openai_ledger, jev_ledger = tmp_path / 'openai.db', tmp_path / 'jev.db'
    with pytest.raises(SystemExit):
        setup.main(['--arm', 'v3', '--execute', '--scope-approved', '--accept-estimate-uncertainty',
                    '--isolation-reviewed', '--journal', str(journal), '--openai-ledger', str(openai_ledger),
                    '--jev-ledger', str(jev_ledger)])
    assert not journal.exists() and not openai_ledger.exists() and not jev_ledger.exists()
