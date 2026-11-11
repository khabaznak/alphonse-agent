"""End-to-end synthetic wiring; no real model or external provider is called."""
import json

import pytest

from alphonse.agent_v2.evaluation.offline_responses_candidate import candidate_chooser
from alphonse.agent_v2.evaluation.responses_pilot_transport import PilotLedger, PilotLimits, requests_sender
from alphonse.agent_v2.evaluation.model_candidate_bridge import CandidateDecisionBridge
from alphonse.agent_v2.evaluation.prescription_http_processors import run_candidate_http
from alphonse.agent_v2.evaluation.loopback_prescription_pilot import (
    LoopbackPrescriptionPilot, PrescriptionSources, evaluate,
)

SOURCES = PrescriptionSources(
    conversation={'earlier_message': 'Started the prescribed course on 2026-04-10.'},
    prescription={'name': 'Asterin', 'dose': '10 mg', 'time': '08:00', 'days': 5},
    treatment_status={'state': 'active'},
)
EXPECTED = {
    'index': {'sources': ['conversation', 'prescription', 'treatment-status']},
    'conversation': {'earlier_message': 'Started the prescribed course on 2026-04-10.'},
    'prescription': {'name': 'Asterin', 'dose': '10 mg', 'time': '08:00', 'days': 5},
    'treatment-status': {'state': 'active'},
}
ANSWER = {'status': 'schedule', 'medications': [{'name': 'Asterin', 'dose': '10 mg', 'time': '08:00'}]}


class FakeResponsesSender:
    offline = True

    def __init__(self):
        self.calls = []
        self.decisions = iter([
            *({'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': name}}
              for name in ('index', 'conversation', 'prescription', 'treatment-status')),
            {'kind': 'answer', 'text': json.dumps(ANSWER)},
        ])

    def __call__(self, url, body, api_key):
        self.calls.append((url, body, api_key))
        return {'status': 'completed', 'model': 'synthetic-model',
                'usage': {'input_tokens': 100, 'output_tokens': 30},
                'output': [{'type': 'message', 'role': 'assistant', 'content': [
                    {'type': 'output_text', 'text': json.dumps(next(self.decisions))}]}]}


def ledger(tmp_path):
    return PilotLedger(str(tmp_path / 'pilot.db'), PilotLimits(
        max_calls=8, max_input_tokens=32768, max_output_tokens=4096,
        max_cost_units=100000, input_cost_units_per_token=1, output_cost_units_per_token=1))


def test_network_sender_and_undeclared_sender_are_rejected(tmp_path):
    for sender in (requests_sender, lambda *_: {}):
        with pytest.raises(ValueError, match='offline_sender_required'):
            candidate_chooser(model='synthetic-model', ledger=ledger(tmp_path),
                counter=lambda _: 100, sender=sender, emit=lambda _: None, max_input_tokens=4096)


def test_bridge_rejects_unknown_fixture_schema(tmp_path):
    transport = type('OfflineTransport', (), {'offline': True})()
    with pytest.raises(ValueError, match='invalid_fixture_argument'):
        CandidateDecisionBridge(transport, 4096, lambda _: None, fixture_argument='unsupported')


def test_offline_flag_is_only_a_caller_assertion(tmp_path):
    class FalselyDeclaredOffline:
        offline = True

        def __call__(self, *_args):
            raise AssertionError('Must not be invoked by this test')

    # The constructor cannot establish that this callable would not delegate to
    # a network sender. No request, network or otherwise, is made here.
    chooser = candidate_chooser(model='synthetic-model', ledger=ledger(tmp_path),
        counter=lambda _: 100, sender=FalselyDeclaredOffline(), emit=lambda _: None,
        max_input_tokens=4096)
    assert chooser.offline is True


def test_responses_wiring_drives_candidate_over_common_source(tmp_path):
    sender, events = FakeResponsesSender(), []
    budget = ledger(tmp_path)
    chooser = candidate_chooser(model='synthetic-model', ledger=budget,
        counter=lambda _: 100, sender=sender, emit=events.append, max_input_tokens=4096)
    with LoopbackPrescriptionPilot(SOURCES) as pilot:
        url = pilot.url
        run = run_candidate_http(base_url=url, goal='Find the synthetic morning prescription.',
                                 chooser=chooser, emit=events.append)
    assert run.status == 'completed'
    assert evaluate(pilot, response=json.loads(run.response), expected_response=ANSWER,
                    expected_sources=EXPECTED)['passed'] is True
    assert len(sender.calls) == budget.snapshot()['calls'] == 5
    assert all('"arguments":{"source":"..."}' in call[1]['input'] for call in sender.calls)
    assert all(event['provider_usage'] == {'input_tokens': 100, 'output_tokens': 30}
               for event in events if event['kind'] == 'system_two')
    assert not any(url in repr(event) for event in events)
