"""Clock Responses attempt wiring; all senders here are local test doubles."""
import json

import pytest

from alphonse.agent_v2.core.inference import InferenceResult
from alphonse.agent_v2.evaluation.clock_responses_attempt import _reason_category, run_clock_responses_attempt
from alphonse.agent_v2.evaluation.responses_pilot_transport import (
    PilotLedger, PilotLimits, ResponsesPilotTransport,
)

CONFIG = {'source_instant': '2026-04-12T14:00:00+00:00',
          'source_timezone': 'America/Mexico_City',
          'expected_instant': '2026-04-12T14:00:00+00:00',
          'expected_timezone': 'America/Mexico_City'}


def test_default_refuses_before_transport_or_journal(tmp_path):
    with pytest.raises(ValueError, match='dispatch_disabled'):
        run_clock_responses_attempt(tmp_path / 'run.jsonl', transport=object(), **CONFIG)
    assert not (tmp_path / 'run.jsonl').exists()


def test_only_safe_reason_categories_are_returned():
    assert _reason_category('candidate_deadline_exceeded') == 'deadline'
    assert _reason_category('pilot_v3_deadline_or_cancelled') == 'deadline_or_cancelled'
    assert _reason_category('candidate_model_unavailable') == 'model_unavailable'
    assert _reason_category('candidate_read_denied') == 'access_denied'
    assert _reason_category('RuntimeError:private-synthetic-secret') == 'other'
    assert _reason_category(None) is None


def test_real_transport_contract_with_local_sender_and_independent_clock(tmp_path):
    calls = []
    actions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}},
        {'kind': 'answer', 'text': json.dumps({'local_time': '08:00', 'local_date': '2026-04-12',
                                               'timezone': 'America/Mexico_City'})},
    ])
    def sender(url, body, key):
        calls.append(body)
        assert key == 'synthetic-key'
        return {'status': 'completed', 'model': 'synthetic-model',
                'usage': {'input_tokens': 100, 'output_tokens': 30},
                'output': [{'type': 'message', 'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': json.dumps(next(actions))}]}]}
    ledger = PilotLedger(str(tmp_path / 'ledger.db'), PilotLimits(
        max_calls=4, max_input_tokens=400000, max_output_tokens=3000,
        max_cost_units=450000, input_cost_units_per_token=1, output_cost_units_per_token=1))
    transport = ResponsesPilotTransport(model='synthetic-model', api_key='synthetic-key', ledger=ledger,
        count_complete_input=lambda body: len(json.dumps(body).encode()), sender=sender, enabled=True)
    result = run_clock_responses_attempt(tmp_path / 'run.jsonl', transport=transport,
                                         dispatch_enabled=True, **CONFIG)
    assert result['status'] == 'completed' and result['passed'] is True
    assert len(calls) == 2 and ledger.snapshot()['calls'] == 2
    assert result['decision_qualified'] is False and result['delivery'] == 'unassessed'
    receipt = (tmp_path / 'run.jsonl').read_text()
    assert 'synthetic-key' not in receipt and '127.0.0.1' not in receipt
    assert '"event": "attempt_finished"' in receipt
    with pytest.raises(FileExistsError):
        run_clock_responses_attempt(tmp_path / 'run.jsonl', transport=transport,
                                    dispatch_enabled=True, **CONFIG)
    assert len(calls) == 2


def test_transport_error_does_not_leak_exception_details(tmp_path):
    class FailingTransport:
        def generate_json(self, *args, **kwargs):
            raise RuntimeError('private-synthetic-secret')
    result = run_clock_responses_attempt(tmp_path / 'run.jsonl', transport=FailingTransport(),
                                         dispatch_enabled=True, **CONFIG)
    assert result['passed'] is False
    assert result['reason_category'] == 'model_unavailable'
    assert result['events'][0]['provider_usage'] is None
    assert 'private-synthetic-secret' not in (tmp_path / 'run.jsonl').read_text()


def test_known_usage_of_invalid_response_is_preserved_without_raw_text(tmp_path):
    class InvalidTransport:
        _model = 'private-synthetic-secret'
        def generate_json(self, *args, **kwargs):
            return InferenceResult(json_value=['private-synthetic-output'],
                                   usage={'input_tokens': 13, 'output_tokens': 4})
    path = tmp_path / 'invalid.jsonl'
    result = run_clock_responses_attempt(path, transport=InvalidTransport(),
                                         dispatch_enabled=True, **CONFIG)
    assert result['passed'] is False
    assert result['events'][0]['status'] == 'invalid'
    assert result['events'][0]['provider_usage'] == {'input_tokens': 13, 'output_tokens': 4}
    assert 'private-synthetic-secret' not in path.read_text()
    assert 'private-synthetic-output' not in path.read_text()


def test_incomplete_responses_api_result_keeps_known_usage_without_payload(tmp_path):
    def sender(*_):
        return {'status': 'incomplete', 'model': 'synthetic-model',
                'usage': {'input_tokens': 37, 'output_tokens': 12},
                'output': [{'private': 'private-synthetic-output'}]}
    ledger = PilotLedger(str(tmp_path / 'ledger.db'), PilotLimits(
        max_calls=2, max_input_tokens=200000, max_output_tokens=2000,
        max_cost_units=250000, input_cost_units_per_token=1, output_cost_units_per_token=1))
    transport = ResponsesPilotTransport(model='synthetic-model', api_key='synthetic-key', ledger=ledger,
        count_complete_input=lambda body: len(json.dumps(body).encode()), sender=sender, enabled=True)
    path = tmp_path / 'incomplete.jsonl'
    result = run_clock_responses_attempt(path, transport=transport, dispatch_enabled=True, **CONFIG)
    assert result['passed'] is False
    assert result['events'][0]['provider_usage'] == {'input_tokens': 37, 'output_tokens': 12}
    assert result['events'][0]['status'] == 'failed'
    assert 'private-synthetic-output' not in path.read_text()
