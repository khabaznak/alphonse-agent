"""Processors use a shared synthetic service, with scripted decisions only."""
import json
from urllib.error import URLError

import pytest

from alphonse.agent_v2.evaluation.loopback_prescription_pilot import LoopbackPrescriptionPilot, PrescriptionSources, evaluate
from alphonse.agent_v2.evaluation.prescription_http_processors import LocalSource, run_candidate_http

SOURCES = PrescriptionSources({'start_date': '2026-04-10'}, {'name': 'Asterin', 'dose': '10 mg'}, {'state': 'active'})
KEY = {'index': {'sources': ['conversation', 'prescription', 'treatment-status']},
       'conversation': {'start_date': '2026-04-10'}, 'prescription': {'name': 'Asterin', 'dose': '10 mg'},
       'treatment-status': {'state': 'active'}}
ANSWER = {'status': 'schedule', 'medications': [{'name': 'Asterin', 'dose': '10 mg'}]}
ORDER = tuple(KEY)


class OfflineChooser:
    offline = True
    def __init__(self, *, answer=ANSWER, read=True):
        decisions = ([{'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': p}} for p in ORDER]
                     if read else []) + [{'kind': 'answer', 'text': json.dumps(answer)}]
        self._decisions = iter(decisions)
    def __call__(self, projection, cap):
        result = dict(next(self._decisions))
        result['usage'] = {'input_tokens': len(projection.encode()), 'output_tokens': len(json.dumps(result).encode())}
        return result


def test_simple_loop_reads_http_sources_and_independent_oracle():
    events = []
    with LoopbackPrescriptionPilot(SOURCES) as pilot:
        result = run_candidate_http(base_url=pilot.url, goal='Read the synthetic records.',
                                    chooser=OfflineChooser(), emit=events.append)
    assert result.status == 'completed', result.reason
    oracle = evaluate(pilot, response=json.loads(result.response), expected_response=ANSWER, expected_sources=KEY)
    assert oracle['passed'] is True
    assert oracle['decision_qualified'] is False and oracle['delivery'] == 'unassessed'
    assert [e['source'] for e in events if e['kind'] == 'source_read'] == list(ORDER)


def test_guess_cannot_pass_evaluator_and_no_live_declaration():
    with LoopbackPrescriptionPilot(SOURCES) as pilot:
        result = run_candidate_http(base_url=pilot.url, goal='Read sources.', chooser=OfflineChooser(read=False), emit=lambda e: None)
    assert not evaluate(pilot, response=ANSWER, expected_response=ANSWER, expected_sources=KEY)['passed']
    assert result.status == 'blocked'
    with pytest.raises(ValueError, match='offline_chooser_required'):
        run_candidate_http(base_url='http://127.0.0.1:1/token/', goal='x', chooser=lambda *_: {}, emit=lambda e: None)


def test_only_literal_loopback_and_allowlisted_sources():
    for bad in ('https://127.0.0.1:4/token/', 'http://localhost:4/token/',
                'http://127.0.0.1:4/', 'http://127.0.0.1:4/token/?x=1',
                'http://127.0.0.1:4/../', 'http://127.0.0.1:4/' + 'a'*42 + '/x/'):
        with pytest.raises(ValueError, match='invalid_fixture_url'):
            LocalSource(bad)
    source = LocalSource('http://127.0.0.1:4/' + 'a'*43 + '/')
    for bad in ('../../secret', 'http://example.test/', [], 'audit'):
        with pytest.raises(PermissionError, match='fixture_source_denied'):
            source.read({'source': bad})


def test_clock_source_scope_denies_prescription_before_network():
    source = LocalSource('http://127.0.0.1:4/' + 'a' * 43 + '/', allowed_source_names=('index',))
    class MustNotConnect:
        def open(self, *_args, **_kwargs):
            raise AssertionError('denied source reached HTTP')
    source._opener = MustNotConnect()
    with pytest.raises(PermissionError, match='fixture_source_denied'):
        source.read({'source': 'prescription'})


def test_candidate_scope_denies_other_source_before_tool_dispatch():
    class ChoosePrescription:
        offline = True
        def __call__(self, *_args):
            return {'kind': 'tool', 'tool_id': 'native.fixture_read',
                    'arguments': {'source': 'prescription'},
                    'usage': {'input_tokens': 10, 'output_tokens': 10}}
    events = []
    result = run_candidate_http(base_url='http://127.0.0.1:4/' + 'a' * 43 + '/',
        goal='What time is it?', chooser=ChoosePrescription(), emit=events.append,
        allowed_source_names=('index',))
    assert result.status == 'blocked' and result.reason == 'candidate_read_denied'
    assert not events


def test_bridge_guidance_must_match_host_source_scope():
    from alphonse.agent_v2.evaluation.model_candidate_bridge import CandidateDecisionBridge
    transport = type('OfflineTransport', (), {'offline': True})()
    chooser = CandidateDecisionBridge(transport, 4096, lambda _: None,
        fixture_argument='source', allowed_source_names=('index',))
    with pytest.raises(ValueError, match='fixture_source_configuration_mismatch'):
        run_candidate_http(base_url='http://127.0.0.1:4/' + 'a' * 43 + '/',
            goal='x', chooser=chooser, emit=lambda _: None)


def test_transport_exception_does_not_disclose_bearer_or_exception_context():
    token = 'b' * 43
    source = LocalSource('http://127.0.0.1:4/' + token + '/')
    class FailedTransport:
        def open(self, *_args, **_kwargs):
            raise URLError('failure contains capability ' + token)
    source._opener = FailedTransport()
    with pytest.raises(ValueError, match='fixture_transport_failed') as error:
        source.read({'source': 'index'})
    assert token not in str(error.value)
    assert error.value.__cause__ is None and error.value.__context__ is None
