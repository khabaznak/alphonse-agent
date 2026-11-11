"""Single-clock operator setup tests; never call a provider."""
import json

import pytest

from alphonse.agent_v2.evaluation import clock_pilot_setup as setup


def test_preview_is_default_and_does_not_request_credentials(capsys, monkeypatch):
    monkeypatch.setattr(setup.getpass, 'getpass', lambda _: pytest.fail('credential prompt in preview'))
    assert setup.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['preview_only'] is True and report['attempts'] == 1
    assert report['model'] == 'gpt-6-luna'
    assert report['limits']['model_calls'] == 10
    assert report['limits']['input_reservation_per_call'] == 65536
    assert report['limits']['output_tokens_per_call'] == 512
    assert report['proposed_budget_usd'] == '0.25'
    assert report['reservation_estimate_usd'] == '0.09325568'
    assert report['estimated_reservations_fit_budget'] is True
    assert 'not a guaranteed billing cap' in ' '.join(report['limitations'])


def test_missing_scope_budget_or_existing_file_refuses_before_credential_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(setup.getpass, 'getpass', lambda _: pytest.fail('unexpected credential prompt'))
    with pytest.raises(SystemExit):
        setup.main(['--execute'])
    base = ['--execute', '--scope-approved', '--accept-estimate-uncertainty', '--isolation-reviewed',
            '--journal', str(tmp_path / 'run.jsonl'), '--ledger', str(tmp_path / 'ledger.db')]
    with pytest.raises(SystemExit):
        setup.main([*base, '--budget-usd', '0.00001'])
    (tmp_path / 'ledger.db').write_text('existing')
    with pytest.raises(SystemExit):
        setup.main(base)
    assert (tmp_path / 'ledger.db').read_text() == 'existing'


@pytest.mark.parametrize('bad', ['0', '-1', 'nan', 'Infinity', '0.0000000001', 'abc'])
def test_invalid_budget_rejected(bad):
    with pytest.raises(ValueError, match='invalid_budget'):
        setup.preview(budget_usd=bad)


def test_direct_api_execution_is_disabled_before_key_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(setup.getpass, 'getpass', lambda _: pytest.fail('unexpected credential prompt'))
    monkeypatch.setattr(setup, 'requests_sender', lambda *args: pytest.fail('unexpected provider call'))
    journal, ledger = tmp_path / 'run.jsonl', tmp_path / 'ledger.db'
    with pytest.raises(SystemExit):
        setup.main(['--execute', '--scope-approved', '--accept-estimate-uncertainty', '--isolation-reviewed',
                    '--journal', str(journal), '--ledger', str(ledger)])
    assert not journal.exists() and not ledger.exists()
