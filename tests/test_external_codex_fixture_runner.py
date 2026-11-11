"""Synthetic fake-executor integration; no Codex or external provider runs."""
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from alphonse.agent_v2.evaluation.external_codex_fixture_runner import run_fixture
from alphonse.agent_v2.evaluation.loopback_prescription_pilot import PrescriptionSources
from alphonse.agent_v2.evaluation.loopback_traffic_pilot import PortalConfig


class FakeExecutor:
    offline = True

    def __init__(self, response, read=True):
        self.response, self.read = response, read

    def __call__(self, *, case, fixture_url, request):
        assert request == 'Synthetic task for subject SYN-1.'
        if self.read:
            with urlopen(fixture_url, timeout=2) as response:
                response.read()
            if case == 'prescription':
                for name in ('conversation', 'prescription', 'treatment-status'):
                    with urlopen(fixture_url + name, timeout=2) as response:
                        response.read()
            elif case == 'traffic':
                request = Request(fixture_url + 'lookup', data=urlencode({'plate': 'SYN-1'}).encode())
                with urlopen(request, timeout=2) as response:
                    response.read()
        return {'jsonl': json.dumps({'type': 'turn.completed',
                'usage': {'input_tokens': 12, 'output_tokens': 4}}) + '\n',
                'final_text': json.dumps(self.response), 'exit_code': 0}


@pytest.mark.parametrize('case', ['clock', 'prescription', 'traffic'])
def test_common_sources_and_independent_oracles(case):
    if case == 'clock':
        fixture = {'instant_utc': '2026-04-12T14:00:00+00:00', 'timezone': 'America/Mexico_City'}
        expected = dict(fixture)
        response = {'local_time': '08:00', 'local_date': '2026-04-12', 'timezone': 'America/Mexico_City'}
    elif case == 'prescription':
        sources = PrescriptionSources({'started': '2026-04-10'}, {'name': 'Synthetic'}, {'state': 'active'})
        fixture = {'sources': sources}
        response = {'status': 'schedule', 'medications': [{'name': 'Synthetic', 'dose': '1 mg', 'time': '08:00'}]}
        expected = {'response': response, 'sources': {'index': {'sources': ['conversation', 'prescription', 'treatment-status']},
            'conversation': {'started': '2026-04-10'}, 'prescription': {'name': 'Synthetic'},
            'treatment-status': {'state': 'active'}}}
    else:
        fixture = {'config': PortalConfig('SYN-1', {'SYN-1': ('T-1',)}, require_capability=True)}
        expected = {'subject': 'SYN-1', 'ticket_ids': ('T-1',)}
        response = {'result': {'status': 'verified', 'subject': 'SYN-1', 'ticket_ids': ['T-1']}}
    output = run_fixture(case=case, request='Synthetic task for subject SYN-1.', fixture=fixture, expected=expected, executor=FakeExecutor(response))
    assert output['fixture_passed'] is True and output['decision_qualified'] is False
    assert output['trace']['reported_usage'] == {'input_tokens': 12, 'output_tokens': 4}
    assert output['trace']['usage_source'] == 'codex_cli_reported'
    assert '127.0.0.1' not in repr(output)
    guessed = run_fixture(case=case, request='Synthetic task for subject SYN-1.', fixture=fixture, expected=expected, executor=FakeExecutor(response, read=False))
    assert guessed['fixture_passed'] is False


def test_errors_unknown_and_live_executor_not_accepted():
    fixture = {'instant_utc': '2026-04-12T14:00:00+00:00', 'timezone': 'America/Mexico_City'}
    class Broken:
        offline = True
        def __call__(self, **_):
            raise RuntimeError('synthetic-secret')
    result = run_fixture(case='clock', request='Synthetic task', fixture=fixture, expected=dict(fixture), executor=Broken())
    assert not result['fixture_passed'] and result['trace']['usage_source'] == 'unknown'
    assert 'synthetic-secret' not in repr(result)
    with pytest.raises(ValueError, match='offline_executor_required'):
        run_fixture(case='clock', request='Synthetic task', fixture=fixture, expected=dict(fixture), executor=lambda **_: None)
