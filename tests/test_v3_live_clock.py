"""Synthetic V3 pilot binding tests: local fakes only, no provider calls."""
import json

import pytest

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.evaluation.deterministic_adapters import (
    _Action, _DeterministicProvider, _ReplayJev, _RunState, _Scenario,
)
from alphonse.agent_v2.evaluation.loopback_clock_pilot import LoopbackClockPilot, evaluate
from alphonse.agent_v2.evaluation.jev_attempt_meter import JevAttemptLedger, MeteredJevTransport
from alphonse.agent_v2.evaluation.jev_pilot_http import JevPilotHTTPTransport
from alphonse.agent_v2.evaluation.responses_pilot_transport import PilotLedger, PilotLimits, ResponsesPilotTransport
from alphonse.agent_v2.evaluation.v3_clock_responses_attempt import run_v3_clock_responses_attempt
from alphonse.agent_v2.evaluation.v3_live_clock import V3PilotResponsesProvider, run_v3_live_clock
from alphonse.agent_v2.system_one import DEFAULT_SYSTEM_ONE_URL, JevCriterionDecisionProvider, SystemOneSettings


def build_provider(tmp_path, *, dispatch_enabled, model='synthetic', pilot_limits=False):
    response = json.dumps({'local_time': '08:00', 'local_date': '2026-04-12',
                           'timezone': 'America/Mexico_City'})
    scenario = _Scenario((_Action('native.fixture_read', 'project_record_search',
                                   {'source': 'index'}, role='read_clock', behavior='read'),),
                         'Answer from the synthetic clock.', response)
    replay = _DeterministicProvider(_RunState(scenario, tmp_path, 'hierarchical_v3'))
    calls = []
    def fake_sender(_url, body, _key):
        calls.append(body)
        envelope = json.loads(body['input'])
        purpose = InferencePurpose(envelope['purpose'])
        request = InferenceRequest(prompt=envelope['prompt'], purpose=purpose)
        if purpose == InferencePurpose.TASK_UNDERSTANDING:
            output = json.dumps({'requested_outcome': 'read synthetic clock', 'supplied_parameters': [],
                                 'constraints': [], 'unresolved_dependencies': [], 'superseded_constraints': []})
        elif 'text' in body:
            output = json.dumps(replay.generate_json(request).json_value)
        else:
            output = replay.generate_markdown(request).content
        return {'status': 'completed', 'model': model,
                'usage': {'input_tokens': 10, 'output_tokens': 10},
                'output': [{'type': 'message', 'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': output}]}]}
    ledger = PilotLedger(str(tmp_path / 'ledger.db'), PilotLimits(
        max_calls=10 if pilot_limits else 20, max_input_tokens=655360 if pilot_limits else 2000000,
        max_output_tokens=5120 if pilot_limits else 20000,
        max_cost_units=3000000, input_cost_units_per_token=1, output_cost_units_per_token=1))
    transport = ResponsesPilotTransport(model=model, api_key='synthetic', ledger=ledger,
        count_complete_input=lambda body: len(json.dumps(body).encode()), sender=fake_sender, enabled=True)
    provider = V3PilotResponsesProvider(transport, max_input_tokens=65536 if pilot_limits else 100000,
                                        max_output_tokens=512 if pilot_limits else 1000,
                                        dispatch_enabled=dispatch_enabled)
    return provider, ledger, calls


def fake_jev_with_real_binding(tmp_path, monkeypatch, *, retries=0):
    # Replace the HTTP class method and judgment methods only in this process.
    # Each synthetic judgment traverses the real Jev client and meter; no
    # TypeSafe HTTP request may occur. This is not model behavior evidence.
    monkeypatch.setattr(JevPilotHTTPTransport, '__call__', lambda self, url, key, payload, timeout:
                        {'model': 'synthetic-jev', 'answers': {'q': {'type': 'noul', 'noul': .5}},
                         'usage': {'input_tokens': 10, 'output_tokens': 2}})
    meter = MeteredJevTransport(JevPilotHTTPTransport(enabled=True),
        JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=100, max_payload_bytes=100000))
    jev = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_url=DEFAULT_SYSTEM_ONE_URL,
                                      api_key='synthetic-jev-key', model='synthetic-jev'), transport=meter)
    replay = _ReplayJev()
    for name in ('evaluate', 'select_plan_tools', 'curate_request_tools', 'triage_plan_messages',
                 'judge_act', 'evaluate_tactical_progress'):
        local_method = getattr(replay, name)
        def judgment(self, *args, _local_method=local_method, **kwargs):
            self.client.evaluate(state={'fixture': True},
                questions={'q': {'type': 'noul', 'instructions': 'synthetic'}}, retries=retries)
            return _local_method(*args, **kwargs)
        monkeypatch.setattr(JevCriterionDecisionProvider, name, judgment)
    def recall(self, *args, **kwargs):
        self.client.evaluate(state={'fixture': True},
            questions={'q': {'type': 'noul', 'instructions': 'synthetic'}}, retries=0)
        return None
    monkeypatch.setattr(JevCriterionDecisionProvider, 'judge_recall', recall)
    monkeypatch.setattr(JevCriterionDecisionProvider, 'requires_mission_outcome_review', False)
    return jev


def test_disabled_provider_and_runner_do_not_dispatch(tmp_path, monkeypatch):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    provider, ledger, calls = build_provider(tmp_path, dispatch_enabled=False)
    with pytest.raises(ValueError, match='dispatch_disabled'):
        provider.generate_json(InferenceRequest(prompt='synthetic', purpose=InferencePurpose.TACTICAL_ACTION))
    with pytest.raises(ValueError, match='dispatch_disabled'):
        run_v3_live_clock(base_url='', provider=provider, system_one=_ReplayJev(),
                          projection_enabled=False, emit=lambda _: None)
    assert calls == [] and ledger.snapshot()['calls'] == 0


@pytest.mark.parametrize('projection_enabled', [False, True])
def test_v3_reads_independently_verified_clock_through_real_transport_contract(tmp_path, monkeypatch, projection_enabled):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '1' if projection_enabled else '0')
    provider, ledger, calls = build_provider(tmp_path, dispatch_enabled=True)
    jev = fake_jev_with_real_binding(tmp_path, monkeypatch)
    prior_jev_cancellation = lambda: False
    jev.client.transport.cancellation_checker = prior_jev_cancellation
    events = []
    with LoopbackClockPilot('2026-04-12T14:00:00+00:00', 'America/Mexico_City') as clock:
        result = run_v3_live_clock(base_url=clock.url, provider=provider, system_one=jev,
                                   projection_enabled=projection_enabled, emit=events.append, dispatch_enabled=True)
    oracle = evaluate(clock, response=json.loads(result.response),
                      expected_instant_utc='2026-04-12T14:00:00+00:00',
                      expected_timezone='America/Mexico_City')
    assert result.status == 'completed' and oracle['passed'] is True
    assert len(calls) > 0 and ledger.snapshot()['calls'] == len(calls)
    assert len(jev.client.transport.ledger.snapshot()) > 0
    assert jev.client.transport.cancellation_checker is prior_jev_cancellation
    assert any(event.get('kind') == 'source_read' for event in events)


@pytest.mark.parametrize('projection_enabled', [False, True])
def test_v3_scripted_clock_fits_shared_predeclared_pilot_limits(tmp_path, monkeypatch, projection_enabled):
    # This regression is about viable test bounds, not actual model behavior.
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '1' if projection_enabled else '0')
    provider, ledger, calls = build_provider(tmp_path, dispatch_enabled=True, model='gpt-6-luna', pilot_limits=True)
    jev = fake_jev_with_real_binding(tmp_path, monkeypatch)
    with LoopbackClockPilot('2026-04-12T14:00:00+00:00', 'America/Mexico_City') as clock:
        result = run_v3_live_clock(base_url=clock.url, provider=provider, system_one=jev,
                                   projection_enabled=projection_enabled, emit=lambda _: None, dispatch_enabled=True)
    oracle = evaluate(clock, response=json.loads(result.response),
                      expected_instant_utc='2026-04-12T14:00:00+00:00', expected_timezone='America/Mexico_City')
    assert result.status == 'completed' and oracle['passed'] is True
    assert len(calls) <= 10 and ledger.snapshot()['calls'] == len(calls)


def test_projection_mismatch_is_rejected_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    provider, ledger, calls = build_provider(tmp_path, dispatch_enabled=True)
    with pytest.raises(ValueError, match='projection_process_mismatch'):
        run_v3_live_clock(base_url='', provider=provider, system_one=fake_jev_with_real_binding(tmp_path, monkeypatch),
                          projection_enabled=True, emit=lambda _: None, dispatch_enabled=True)
    assert calls == [] and ledger.snapshot()['calls'] == 0


def test_unmetered_jev_is_rejected_before_any_provider_dispatch(tmp_path, monkeypatch):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    provider, ledger, calls = build_provider(tmp_path, dispatch_enabled=True)
    with pytest.raises(ValueError, match='metered_jev_required'):
        run_v3_live_clock(base_url='', provider=provider, system_one=_ReplayJev(),
                          projection_enabled=False, emit=lambda _: None, dispatch_enabled=True)
    assert calls == [] and ledger.snapshot()['calls'] == 0


@pytest.mark.parametrize('transport_kind', ['direct', 'wrapped_arbitrary', 'disabled_http'])
def test_incomplete_jev_binding_is_rejected_before_dispatch(tmp_path, monkeypatch, transport_kind):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    provider, ledger, calls = build_provider(tmp_path, dispatch_enabled=True)
    underlying = (JevPilotHTTPTransport(enabled=False) if transport_kind == 'disabled_http'
                  else lambda *args: pytest.fail('unexpected transport call'))
    transport = (underlying if transport_kind == 'direct' else
                 MeteredJevTransport(underlying, JevAttemptLedger(str(tmp_path / 'jev.db'),
                                                                  max_attempts=2, max_payload_bytes=1000)))
    jev = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_url=DEFAULT_SYSTEM_ONE_URL,
                                      api_key='synthetic-jev-key', model='synthetic-jev'), transport=transport)
    with pytest.raises(ValueError, match='metered_jev_required'):
        run_v3_live_clock(base_url='', provider=provider, system_one=jev, projection_enabled=False,
                          emit=lambda _: None, dispatch_enabled=True)
    assert calls == [] and ledger.snapshot()['calls'] == 0


@pytest.mark.parametrize('tampering', ['instance_override', 'client_override', 'settings_mismatch', 'nonfresh_ledger'])
def test_jev_binding_tampering_fails_before_dispatch(tmp_path, monkeypatch, tampering):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    provider, ledger, calls = build_provider(tmp_path, dispatch_enabled=True)
    jev = fake_jev_with_real_binding(tmp_path, monkeypatch)
    if tampering == 'instance_override':
        jev._select_tools = lambda *args, **kwargs: None
    elif tampering == 'client_override':
        jev.client.evaluate = lambda **kwargs: {}
    elif tampering == 'settings_mismatch':
        jev.client.model = 'different-model'
    else:
        jev.client.transport.ledger.start(payload_bytes=10, question_count=1, estimated_input_tokens=2)
    with pytest.raises(ValueError, match='metered_jev_required|fresh_jev_ledger_required'):
        run_v3_live_clock(base_url='', provider=provider, system_one=jev, projection_enabled=False,
                          emit=lambda _: None, dispatch_enabled=True)
    assert calls == [] and ledger.snapshot()['calls'] == 0


def test_processor_without_observed_jev_attempt_cannot_qualify(tmp_path, monkeypatch):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    provider, _, _ = build_provider(tmp_path, dispatch_enabled=True)
    jev = fake_jev_with_real_binding(tmp_path, monkeypatch)
    replay = _ReplayJev()
    for name in ('evaluate', 'select_plan_tools', 'curate_request_tools', 'triage_plan_messages',
                 'judge_act', 'evaluate_tactical_progress'):
        local = getattr(replay, name)
        monkeypatch.setattr(JevCriterionDecisionProvider, name,
                            lambda self, *args, _local=local, **kwargs: _local(*args, **kwargs))
    monkeypatch.setattr(JevCriterionDecisionProvider, 'judge_recall', lambda *args, **kwargs: None)
    with LoopbackClockPilot('2026-04-12T14:00:00+00:00', 'America/Mexico_City') as clock:
        with pytest.raises(ValueError, match='no_jev_attempt_observed'):
            run_v3_live_clock(base_url=clock.url, provider=provider, system_one=jev,
                              projection_enabled=False, emit=lambda _: None, dispatch_enabled=True)


def test_deadline_crossed_before_fixture_read_prevents_read(tmp_path, monkeypatch):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    now = [0.0]
    monkeypatch.setattr('alphonse.agent_v2.evaluation.v3_live_clock.monotonic', lambda: now[0])
    provider, _, _ = build_provider(tmp_path, dispatch_enabled=True)
    jev = fake_jev_with_real_binding(tmp_path, monkeypatch)
    def emit(event):
        if event.get('kind') == 'tool':
            now[0] = 101.0
    with LoopbackClockPilot('2026-04-12T14:00:00+00:00', 'America/Mexico_City') as clock:
        result = run_v3_live_clock(base_url=clock.url, provider=provider, system_one=jev,
                                   projection_enabled=False, emit=emit, dispatch_enabled=True,
                                   timeout_seconds=100)
    assert result.status == 'cancelled'
    assert not any(event.get('event') == 'clock_read' for event in clock.audit())


def test_jev_retry_after_deadline_never_dispatches_second_http_request(tmp_path, monkeypatch):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    now, calls = [0.0], []
    monkeypatch.setattr('alphonse.agent_v2.evaluation.v3_live_clock.monotonic', lambda: now[0])
    monkeypatch.setattr('alphonse.agent_v2.system_one.sleep', lambda _delay: now.__setitem__(0, 101.0))
    provider, _, _ = build_provider(tmp_path, dispatch_enabled=True)
    jev = fake_jev_with_real_binding(tmp_path, monkeypatch, retries=2)
    def unavailable(self, url, key, payload, timeout):
        calls.append(True)
        raise RuntimeError('system_one_temporarily_unavailable:503')
    monkeypatch.setattr(JevPilotHTTPTransport, '__call__', unavailable)
    with LoopbackClockPilot('2026-04-12T14:00:00+00:00', 'America/Mexico_City') as clock:
        result = run_v3_live_clock(base_url=clock.url, provider=provider, system_one=jev,
                                   projection_enabled=False, emit=lambda _: None, dispatch_enabled=True,
                                   timeout_seconds=100)
    assert result.status == 'cancelled'
    assert calls == [True]
    assert len(jev.client.transport.ledger.snapshot()) == 1
    assert not any(event.get('event') == 'clock_read' for event in clock.audit())


@pytest.mark.parametrize('projection_enabled', [False, True])
def test_v3_attempt_journals_actual_processor_with_separately_metered_fake_jev(tmp_path, monkeypatch, projection_enabled):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '1' if projection_enabled else '0')
    provider, _, calls = build_provider(tmp_path, dispatch_enabled=True, model='gpt-6-luna')
    jev = fake_jev_with_real_binding(tmp_path, monkeypatch)
    path = tmp_path / 'v3.jsonl'
    result = run_v3_clock_responses_attempt(path, source_instant='2026-04-12T14:00:00+00:00',
        source_timezone='America/Mexico_City', expected_instant='2026-04-12T14:00:00+00:00',
        expected_timezone='America/Mexico_City', provider=provider, system_one=jev,
        projection_enabled=projection_enabled, dispatch_enabled=True)
    assert result['status'] == 'completed' and result['passed'] is True
    assert result['decision_qualified'] is False and result['delivery'] == 'unassessed'
    assert len(result['jev_attempts']) > 0 and len(calls) > 0
    text = path.read_text()
    assert 'synthetic-jev-key' not in text and '127.0.0.1' not in text
    assert '"event": "attempt_finished"' in text


def test_v3_attempt_disabled_and_unmetered_are_never_passes(tmp_path, monkeypatch):
    monkeypatch.delenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', raising=False)
    provider, ledger, calls = build_provider(tmp_path, dispatch_enabled=True, model='gpt-6-luna')
    config = dict(source_instant='2026-04-12T14:00:00+00:00', source_timezone='America/Mexico_City',
                  expected_instant='2026-04-12T14:00:00+00:00', expected_timezone='America/Mexico_City',
                  provider=provider, system_one=_ReplayJev(), projection_enabled=False)
    with pytest.raises(ValueError, match='dispatch_disabled'):
        run_v3_clock_responses_attempt(tmp_path / 'disabled.jsonl', **config)
    assert not (tmp_path / 'disabled.jsonl').exists()
    result = run_v3_clock_responses_attempt(tmp_path / 'unmetered.jsonl', dispatch_enabled=True, **config)
    assert result['status'] == 'error' and result['passed'] is None and result['jev_attempts'] is None
    assert calls == [] and ledger.snapshot()['calls'] == 0
