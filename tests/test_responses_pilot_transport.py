"""Synthetic HTTP doubles only; no provider or household calls."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.evaluation.responses_pilot_transport import PilotLedger, PilotLimits, ResponsesPilotTransport, requests_sender


@pytest.fixture
def ledger(tmp_path):
    return PilotLedger(str(tmp_path / 'pilot.db'), PilotLimits(2, 200, 40, 1000, 2, 3))


def response(text='{"kind":"answer"}', *, usage=None, model='synthetic-model'):
    return {'status': 'completed', 'model': model,
            'usage': {'input_tokens': 5, 'output_tokens': 3} if usage is None else usage,
            'output': [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': text}]}]}


def request(**kwargs):
    return InferenceRequest(prompt='Return a JSON object.', purpose=InferencePurpose.TACTICAL_ACTION, **kwargs)


def transport(ledger, sender, *, enabled=False, counter=lambda body: 10):
    return ResponsesPilotTransport(model='synthetic-model', api_key='fake-key', ledger=ledger,
                                   count_complete_input=counter, sender=sender, enabled=enabled)


def test_disabled_by_default_without_dispatch_or_charge(ledger):
    calls = []
    subject = transport(ledger, lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='disabled'):
        subject.generate_json(request(), max_input_tokens=50, max_output_tokens=10)
    assert calls == [] and ledger.snapshot()['calls'] == 0


def test_reserve_before_mock_send_and_record_only_provider_usage(ledger):
    calls = []
    def send(url, body, key):
        assert ledger.snapshot() == {'calls': 1, 'input_tokens': 50, 'output_tokens': 10, 'cost_units': 130}
        calls.append((url, body, key))
        return response('{"kind":"answer","usage":{"input_tokens":0}}')
    result = transport(ledger, send, enabled=True).generate_json(request(), max_input_tokens=50, max_output_tokens=10)
    assert result.usage == {'input_tokens': 5, 'output_tokens': 3}
    assert calls[0][1]['max_output_tokens'] == 10 and calls[0][1]['store'] is False


def test_pre_dispatch_budget_count_and_cancellation_fail_closed(ledger):
    calls = []
    bad_counter = transport(ledger, lambda *args: calls.append(args), enabled=True, counter=lambda _: 51)
    with pytest.raises(ValueError, match='input_limit'):
        bad_counter.generate_json(request(), max_input_tokens=50, max_output_tokens=10)
    subject = transport(ledger, lambda *args: calls.append(args), enabled=True)
    with pytest.raises(ValueError, match='cancelled_before'):
        subject.generate_json(request(cancel_checker=lambda: True), max_input_tokens=50, max_output_tokens=10)
    assert ledger.snapshot()['calls'] == 0 and not calls


def test_unknown_usage_failure_and_timeout_reservations_never_refunded(ledger):
    subject = transport(ledger, lambda *_: response(usage={}), enabled=True)
    with pytest.raises(ValueError, match='usage_invalid'):
        subject.generate_json(request(), max_input_tokens=50, max_output_tokens=10)
    def failed(*_):
        raise TimeoutError('unknown remote effect')
    subject = transport(ledger, failed, enabled=True)
    with pytest.raises(ValueError, match='sender_failed') as caught:
        subject.generate_json(request(), max_input_tokens=50, max_output_tokens=10)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    with pytest.raises(ValueError, match='budget_exhausted'):
        subject.generate_json(request(), max_input_tokens=50, max_output_tokens=10)
    assert ledger.snapshot()['calls'] == 2


def test_provider_model_and_usage_overrun_are_rejected_after_dispatch(ledger):
    subject = transport(ledger, lambda *_: response(model='other'), enabled=True)
    with pytest.raises(ValueError, match='model_mismatch'):
        subject.generate_json(request(), max_input_tokens=50, max_output_tokens=10)
    subject = transport(ledger, lambda *_: response(usage={'input_tokens': 51, 'output_tokens': 1}), enabled=True)
    with pytest.raises(ValueError, match='usage_invalid'):
        subject.generate_json(request(), max_input_tokens=50, max_output_tokens=10)


def test_concurrent_reservations_cannot_overrun_budget(tmp_path):
    shared = PilotLedger(str(tmp_path / 'pilot.db'), PilotLimits(1, 100, 20, 1000, 2, 3))
    def reserve():
        try:
            shared.reserve(50, 10)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: reserve(), range(4)))
    assert sum(results) == 1 and shared.snapshot()['calls'] == 1


def test_restart_preserves_reservations_and_rejects_changed_rates(tmp_path):
    path = str(tmp_path / 'pilot.db')
    config = PilotLimits(1, 100, 20, 1000, 2, 3)
    PilotLedger(path, config).reserve(50, 10)
    reopened = PilotLedger(path, config)
    with pytest.raises(ValueError, match='budget_exhausted'):
        reopened.reserve(50, 10)
    with pytest.raises(ValueError, match='limits_changed'):
        PilotLedger(path, PilotLimits(2, 100, 20, 1000, 2, 3))


def test_cancelled_after_reservation_does_not_dispatch_or_refund(ledger):
    calls = []
    checks = iter([False, True])
    subject = transport(ledger, lambda *args: calls.append(args), enabled=True)
    with pytest.raises(ValueError, match='cancelled_after_reservation'):
        subject.generate_json(request(cancel_checker=lambda: next(checks)), max_input_tokens=50, max_output_tokens=10)
    assert calls == [] and ledger.snapshot()['calls'] == 1


def test_unsupported_tools_and_schema_fail_before_dispatch(ledger):
    calls = []
    subject = transport(ledger, lambda *args: calls.append(args), enabled=True)
    with pytest.raises(ValueError, match='capability_unsupported'):
        subject.generate_json(request(output_schema={'type': 'object'}), max_input_tokens=50, max_output_tokens=10)
    assert calls == [] and ledger.snapshot()['calls'] == 0


def test_returned_refusal_or_invalid_json_keeps_reservation(ledger):
    subject = transport(ledger, lambda *_: response('not JSON'), enabled=True)
    with pytest.raises(ValueError, match='invalid_json'):
        subject.generate_json(request(), max_input_tokens=50, max_output_tokens=10)
    incomplete = response()
    incomplete['status'] = 'incomplete'
    subject = transport(ledger, lambda *_: incomplete, enabled=True)
    with pytest.raises(ValueError, match='not_completed'):
        subject.generate_json(request(), max_input_tokens=50, max_output_tokens=10)
    assert ledger.snapshot()['calls'] == 2


def test_http_sender_posts_once_without_environment_proxy_or_redirect(monkeypatch):
    import requests
    calls = []
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, chunk_size):
            yield b'{"status":"completed"}'
    class Session:
        trust_env = True
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def mount(self, *args): pass
        def post(self, *args, **kwargs):
            assert self.trust_env is False
            calls.append((args, kwargs))
            return Response()
    monkeypatch.setattr(requests, 'Session', Session)
    result = requests_sender('https://api.openai.com/v1/responses', {'model': 'fake'}, 'synthetic-key')
    assert result == {'status': 'completed'} and len(calls) == 1
    assert calls[0][1]['allow_redirects'] is False and calls[0][1]['stream'] is True
    with pytest.raises(ValueError, match='endpoint_denied'):
        requests_sender('https://example.com', {}, 'synthetic-key')


def test_http_sender_does_not_chain_sensitive_request_or_response(monkeypatch):
    import requests
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def mount(self, *args): pass
        def post(self, *args, **kwargs):
            req = requests.Request('POST', args[0], headers=kwargs['headers']).prepare()
            raise requests.RequestException('synthetic-key', request=req)
    monkeypatch.setattr(requests, 'Session', Session)
    with pytest.raises(ValueError, match='pilot_provider_request_failed') as caught:
        requests_sender('https://api.openai.com/v1/responses', {}, 'synthetic-key')
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert 'synthetic-key' not in str(caught.value)


def test_malformed_provider_response_does_not_chain_response_text(monkeypatch):
    import requests
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, chunk_size): yield b'synthetic-sensitive-invalid-json'
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def mount(self, *args): pass
        def post(self, *args, **kwargs): return Response()
    monkeypatch.setattr(requests, 'Session', Session)
    with pytest.raises(ValueError, match='pilot_provider_request_failed') as caught:
        requests_sender('https://api.openai.com/v1/responses', {}, 'synthetic-key')
    assert caught.value.__cause__ is None and caught.value.__context__ is None
