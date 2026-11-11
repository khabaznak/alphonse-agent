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


def test_explicit_opt_in_wires_fresh_meter_and_fake_runner_without_printing_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '0')
    fake_keys = iter(['synthetic-openai-key', 'synthetic-jev-key'])
    monkeypatch.setattr(setup.getpass, 'getpass', lambda _: next(fake_keys))
    calls = []
    def fake_attempt(output, **kwargs):
        calls.append(kwargs)
        assert kwargs['provider'].transport._model == setup.MODEL
        assert kwargs['system_one'].client.api_key == 'synthetic-jev-key'
        meter = kwargs['system_one'].client.transport
        assert type(meter) is MeteredJevTransport and type(meter.transport) is JevPilotHTTPTransport
        assert meter.ledger.snapshot() == [] and kwargs['projection_enabled'] is True
        assert os_value() == '1'
        return {'passed': True, 'status': 'completed', 'decision_qualified': False}
    def os_value():
        import os
        return os.environ.get('ALPHONSE_V3_CONTEXT_PROJECTIONS')
    monkeypatch.setattr(setup, 'run_v3_clock_responses_attempt', fake_attempt)
    args = ['--arm', 'targeted_v3', '--execute', '--scope-approved', '--accept-estimate-uncertainty',
            '--isolation-reviewed', '--journal', str(tmp_path / 'run.jsonl'),
            '--openai-ledger', str(tmp_path / 'openai.db'), '--jev-ledger', str(tmp_path / 'jev.db')]
    assert setup.main(args) == 0
    out = capsys.readouterr().out
    assert len(calls) == 1 and 'synthetic-openai-key' not in out and 'synthetic-jev-key' not in out
    assert (tmp_path / 'openai.db').exists() and (tmp_path / 'jev.db').exists()
