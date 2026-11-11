"""Synthetic-only candidate wiring; no external provider calls."""
import json

import pytest

from alphonse.agent_v2.core.inference import InferenceResult
from alphonse.agent_v2.evaluation.candidate_clock_responses import ClockResponsesChooser, run_clock_candidate
from alphonse.agent_v2.evaluation.responses_pilot_transport import PilotLedger, PilotLimits, ResponsesPilotTransport
from alphonse.agent_v2.evaluation.loopback_clock_pilot import LoopbackClockPilot, evaluate


class FakeSource:
    def __init__(self, *_args, **_kwargs):
        pass

    def read(self, args):
        assert args == {'source': 'index'}
        return {'source': 'index', 'content': {'instant_utc': '2026-04-12T14:00:00+00:00',
                'timezone': 'America/Mexico_City'}, 'candidate_complete': True}


class FakeTransport:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def generate_json(self, request, *, max_input_tokens, max_output_tokens):
        self.calls.append((request.prompt, max_input_tokens, max_output_tokens))
        return InferenceResult(json_value=next(self.decisions),
                               usage={'input_tokens': 100, 'output_tokens': 30})


def test_default_disabled_prevents_transport_dispatch(monkeypatch):
    monkeypatch.setattr('alphonse.agent_v2.evaluation.prescription_http_processors.LocalSource', FakeSource)
    provider = FakeTransport([])
    chooser = ClockResponsesChooser(provider, max_input_tokens=4096, emit=lambda _event: None)
    run = run_clock_candidate(base_url='synthetic', chooser=chooser, emit=lambda _event: None)
    assert run.status != 'completed'
    assert provider.calls == []


def test_candidate_reads_clock_and_uses_provider_reported_usage(monkeypatch):
    monkeypatch.setattr('alphonse.agent_v2.evaluation.prescription_http_processors.LocalSource', FakeSource)
    answer = {'local_time': '08:00', 'local_date': '2026-04-12', 'timezone': 'America/Mexico_City'}
    provider = FakeTransport([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'},
         'usage': {'input_tokens': 1, 'output_tokens': 1}},
        {'kind': 'answer', 'text': json.dumps(answer)},
    ])
    events = []
    chooser = ClockResponsesChooser(provider, max_input_tokens=4096, emit=events.append, enabled=True)
    run = run_clock_candidate(base_url='synthetic', chooser=chooser, emit=events.append)
    assert run.status == 'completed'
    assert json.loads(run.response) == answer
    assert len(provider.calls) == 2
    assert all(row['provider_usage'] == {'input_tokens': 100, 'output_tokens': 30}
               for row in events if row['kind'] == 'system_two')
    assert any(row['kind'] == 'source_read' for row in events)


def test_host_denies_other_source_before_source_read(monkeypatch):
    monkeypatch.setattr('alphonse.agent_v2.evaluation.prescription_http_processors.LocalSource', FakeSource)
    provider = FakeTransport([{'kind': 'tool', 'tool_id': 'native.fixture_read',
                               'arguments': {'source': 'prescription'}}])
    events = []
    chooser = ClockResponsesChooser(provider, max_input_tokens=4096, emit=events.append, enabled=True)
    run = run_clock_candidate(base_url='synthetic', chooser=chooser, emit=events.append)
    assert run.status == 'blocked'
    assert not any(row['kind'] == 'source_read' for row in events)


def test_transport_failure_does_not_leak_exception_or_mark_usage_known():
    class FailingTransport:
        def generate_json(self, *_args, **_kwargs):
            raise RuntimeError('synthetic-secret-in-original-error')
    events = []
    chooser = ClockResponsesChooser(FailingTransport(), max_input_tokens=4096, emit=events.append, enabled=True)
    try:
        chooser('synthetic prompt', 512)
    except ValueError as exc:
        assert str(exc) == 'clock_model_transport_failed'
        assert exc.__cause__ is None and exc.__context__ is None
    else:
        raise AssertionError('expected sanitized failure')
    assert 'synthetic-secret-in-original-error' not in repr(events)
    assert events[0]['provider_usage'] is None
    assert events[0]['usage_diagnostics']['output']['source'] == 'unknown'


def test_invalid_provider_result_records_attempt_without_inventing_usage():
    class InvalidTransport:
        def generate_json(self, *_args, **_kwargs):
            return object()
    events = []
    chooser = ClockResponsesChooser(InvalidTransport(), max_input_tokens=4096, emit=events.append, enabled=True)
    with pytest.raises(ValueError, match='invalid_clock_model_decision'):
        chooser('synthetic prompt', 512)
    assert events[0]['status'] == 'invalid'
    assert events[0]['provider_usage'] == {'input_tokens': None, 'output_tokens': None}


def test_invalid_output_preserves_reported_provider_usage():
    class InvalidTransport:
        def generate_json(self, *_args, **_kwargs):
            return InferenceResult(content='malformed', usage={'input_tokens': 123, 'output_tokens': 45})
    events = []
    chooser = ClockResponsesChooser(InvalidTransport(), max_input_tokens=4096, emit=events.append, enabled=True)
    with pytest.raises(ValueError, match='invalid_clock_model_decision'):
        chooser('synthetic prompt', 512)
    assert events[0]['status'] == 'invalid'
    assert events[0]['provider_usage'] == {'input_tokens': 123, 'output_tokens': 45}
    assert events[0]['usage_diagnostics']['input']['source'] == 'provider'


def test_cancellation_before_and_after_provider_call_is_not_completion(monkeypatch):
    monkeypatch.setattr('alphonse.agent_v2.evaluation.prescription_http_processors.LocalSource', FakeSource)
    before = lambda: True
    provider = FakeTransport([])
    chooser = ClockResponsesChooser(provider, max_input_tokens=4096, emit=lambda _: None,
                                    enabled=True, cancellation_checker=before)
    run = run_clock_candidate(base_url='synthetic', chooser=chooser, emit=lambda _: None,
                              cancellation_checker=before)
    assert run.status == 'cancelled' and provider.calls == []

    state = {'cancelled': False}
    def checker():
        return state['cancelled']
    class CancellingTransport:
        def generate_json(self, *_args, **_kwargs):
            state['cancelled'] = True
            return InferenceResult(json_value={'kind': 'answer', 'text': 'unexpected'},
                                   usage={'input_tokens': 100, 'output_tokens': 30})
    events = []
    chooser = ClockResponsesChooser(CancellingTransport(), max_input_tokens=4096, emit=events.append,
                                    enabled=True, cancellation_checker=checker)
    run = run_clock_candidate(base_url='synthetic', chooser=chooser, emit=events.append,
                              cancellation_checker=checker)
    assert run.status != 'completed'
    assert events[0]['provider_usage'] == {'input_tokens': 100, 'output_tokens': 30}
    assert events[0]['status'] == 'cancelled_after_dispatch'


def test_distinct_cancellation_checkers_are_rejected_before_execution():
    chooser = ClockResponsesChooser(FakeTransport([]), max_input_tokens=4096, emit=lambda _: None,
                                    cancellation_checker=lambda: False)
    with pytest.raises(ValueError, match='invalid_clock_runner_configuration'):
        run_clock_candidate(base_url='synthetic', chooser=chooser, emit=lambda _: None,
                            cancellation_checker=lambda: False)


def test_elapsed_task_budget_is_not_frozen_with_clock_fixture(monkeypatch):
    monkeypatch.setattr('alphonse.agent_v2.evaluation.prescription_http_processors.LocalSource', FakeSource)
    state = {'now': 0.0}
    monkeypatch.setattr('alphonse.agent_v2.evaluation.candidate_clock_responses.monotonic', lambda: state['now'])
    class SlowTransport:
        def generate_json(self, *_args, **_kwargs):
            state['now'] = 101.0
            return InferenceResult(json_value={'kind': 'tool', 'tool_id': 'native.fixture_read',
                                               'arguments': {'source': 'index'}},
                                   usage={'input_tokens': 100, 'output_tokens': 30})
    events = []
    chooser = ClockResponsesChooser(SlowTransport(), max_input_tokens=4096, emit=events.append, enabled=True)
    run = run_clock_candidate(base_url='synthetic', chooser=chooser, emit=events.append)
    assert run.status == 'blocked'
    assert not any(event.get('kind') == 'source_read' for event in events)


def test_answer_returned_after_deadline_is_not_accepted(monkeypatch):
    monkeypatch.setattr('alphonse.agent_v2.evaluation.prescription_http_processors.LocalSource', FakeSource)
    state = {'now': 0.0, 'calls': 0}
    monkeypatch.setattr('alphonse.agent_v2.evaluation.candidate_clock_responses.monotonic', lambda: state['now'])
    class SlowFinalTransport:
        def generate_json(self, *_args, **_kwargs):
            state['calls'] += 1
            if state['calls'] == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}}
            else:
                state['now'] = 101.0
                decision = {'kind': 'answer', 'text': '{"local_time":"08:00"}'}
            return InferenceResult(json_value=decision, usage={'input_tokens': 100, 'output_tokens': 30})
    events = []
    chooser = ClockResponsesChooser(SlowFinalTransport(), max_input_tokens=4096, emit=events.append, enabled=True)
    run = run_clock_candidate(base_url='synthetic', chooser=chooser, emit=events.append)
    assert any(event.get('kind') == 'source_read' for event in events)
    assert run.status != 'completed'


@pytest.mark.parametrize(('expire_on', 'expects_read'), [(2, False), (5, True)])
def test_deadline_expiry_during_last_freshness_check_fails_closed(monkeypatch, expire_on, expects_read):
    from alphonse.agent_v2.core.intelligence.v3 import single_loop_candidate
    monkeypatch.setattr('alphonse.agent_v2.evaluation.prescription_http_processors.LocalSource', FakeSource)
    state = {'now': 0.0, 'fresh_checks': 0}
    monkeypatch.setattr('alphonse.agent_v2.evaluation.candidate_clock_responses.monotonic', lambda: state['now'])
    original_fresh = single_loop_candidate._is_fresh
    def advancing_fresh(*args):
        ok = original_fresh(*args)
        state['fresh_checks'] += 1
        if state['fresh_checks'] == expire_on:
            state['now'] = 101.0
        return ok
    monkeypatch.setattr(single_loop_candidate, '_is_fresh', advancing_fresh)
    answer = {'local_time': '08:00', 'local_date': '2026-04-12', 'timezone': 'America/Mexico_City'}
    provider = FakeTransport([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}},
        {'kind': 'answer', 'text': json.dumps(answer)},
    ])
    events = []
    chooser = ClockResponsesChooser(provider, max_input_tokens=4096, emit=events.append, enabled=True)
    run = run_clock_candidate(base_url='synthetic', chooser=chooser, emit=events.append)
    assert run.status == 'blocked'
    assert run.reason == 'candidate_deadline_exceeded'
    assert any(event.get('kind') == 'source_read' for event in events) is expects_read


def test_candidate_through_responses_parser_and_persisted_reservations(monkeypatch, tmp_path):
    monkeypatch.setattr('alphonse.agent_v2.evaluation.prescription_http_processors.LocalSource', FakeSource)
    decisions = iter([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}},
        {'kind': 'answer', 'text': json.dumps({'local_time': '08:00', 'local_date': '2026-04-12',
                                              'timezone': 'America/Mexico_City'})},
    ])
    calls = []
    def fake_sender(url, body, key):
        calls.append((url, body['model']))
        return {'status': 'completed', 'model': 'synthetic-model',
                'usage': {'input_tokens': 100, 'output_tokens': 30},
                'output': [{'type': 'message', 'role': 'assistant', 'content': [
                    {'type': 'output_text', 'text': json.dumps(next(decisions))}]}]}
    ledger = PilotLedger(str(tmp_path / 'ledger.db'), PilotLimits(
        max_calls=6, max_input_tokens=24576, max_output_tokens=3072,
        max_cost_units=100000, input_cost_units_per_token=1, output_cost_units_per_token=1))
    transport = ResponsesPilotTransport(model='synthetic-model', api_key='synthetic-key', ledger=ledger,
        count_complete_input=lambda _body: 100, sender=fake_sender, enabled=True)
    chooser = ClockResponsesChooser(transport, max_input_tokens=4096, emit=lambda _: None, enabled=True)
    run = run_clock_candidate(base_url='synthetic', chooser=chooser, emit=lambda _: None)
    assert run.status == 'completed'
    assert len(calls) == ledger.snapshot()['calls'] == 2
    assert ledger.snapshot()['input_tokens'] == 8192  # full reservation, not the fake usage


def test_actual_synthetic_clock_http_and_independent_oracle():
    answer = {'local_time': '08:00', 'local_date': '2026-04-12', 'timezone': 'America/Mexico_City'}
    provider = FakeTransport([
        {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}},
        {'kind': 'answer', 'text': json.dumps(answer)},
    ])
    chooser = ClockResponsesChooser(provider, max_input_tokens=4096, emit=lambda _: None, enabled=True)
    with LoopbackClockPilot('2026-04-12T14:00:00+00:00', 'America/Mexico_City') as clock:
        result = run_clock_candidate(base_url=clock.url, chooser=chooser, emit=lambda _: None)
    assert result.status == 'completed'
    oracle = evaluate(clock, response=json.loads(result.response),
                      expected_instant_utc='2026-04-12T14:00:00+00:00',
                      expected_timezone='America/Mexico_City')
    assert oracle['passed'] is True
    assert oracle['decision_qualified'] is False
