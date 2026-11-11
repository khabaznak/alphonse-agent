"""Synthetic HTTP doubles only: no live TypeSafe request is made."""
import json

import pytest
import requests

from alphonse.agent_v2.evaluation.jev_attempt_meter import JevAttemptLedger, MeteredJevTransport
from alphonse.agent_v2.evaluation.jev_pilot_http import JevPilotHTTPTransport
from alphonse.agent_v2.system_one import DEFAULT_SYSTEM_ONE_URL, SystemOneRequestError, TypeSafeSystemOneClient
from alphonse.agent_v2.evaluation.jev_pilot_payload import serialize_payload


class Response:
    def __init__(self, status, data=b'{}'):
        self.status_code, self.data = status, data
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def iter_content(self, *, chunk_size):
        yield self.data


class Session:
    def __init__(self, response, calls):
        self.response, self.calls, self.trust_env, self.adapter = response, calls, True, None
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def mount(self, prefix, adapter):
        self.adapter = adapter
        assert prefix == 'https://'
    def post(self, url, **kwargs):
        self.calls.append((self.trust_env, self.adapter.max_retries.total, url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_disabled_and_endpoint_mismatch_make_no_request(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, 'Session', lambda: Session(Response(200), calls))
    with pytest.raises(ValueError, match='http_disabled'):
        JevPilotHTTPTransport()(DEFAULT_SYSTEM_ONE_URL, 'key', {}, 1)
    with pytest.raises(ValueError, match='endpoint_denied'):
        JevPilotHTTPTransport(enabled=True)('https://other.invalid', 'key', {}, 1)
    assert not calls


def test_transport_is_single_request_with_no_redirect_proxy_or_retries(monkeypatch):
    calls = []
    value = {'answers': {'q': {'type': 'noul', 'noul': .5}}, 'usage': {'input_tokens': 3, 'output_tokens': 1}}
    monkeypatch.setattr(requests, 'Session', lambda: Session(Response(200, json.dumps(value).encode()), calls))
    result = JevPilotHTTPTransport(enabled=True)(DEFAULT_SYSTEM_ONE_URL, 'fixture-key', {'questions': {}}, 3)
    assert result == value and len(calls) == 1
    trust_env, retries, url, kwargs = calls[0]
    assert not trust_env and retries == 0 and url == DEFAULT_SYSTEM_ONE_URL
    assert kwargs['allow_redirects'] is False and kwargs['stream'] is True
    assert kwargs['timeout'] == (3, 3)
    assert kwargs['headers']['Authorization'] == 'Bearer fixture-key'
    assert kwargs['headers']['Content-Type'] == 'application/json'
    assert kwargs['data'] == serialize_payload({'questions': {}})


def test_connection_failure_sanitizes_credentials_and_chains(monkeypatch):
    calls = []
    secret = 'synthetic-secret-token'
    monkeypatch.setattr(requests, 'Session', lambda: Session(requests.ConnectionError(secret), calls))
    with pytest.raises(RuntimeError, match='^system_one_connection_failed$') as caught:
        JevPilotHTTPTransport(enabled=True)(DEFAULT_SYSTEM_ONE_URL, secret, {}, 3)
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert secret not in str(caught.value)


@pytest.mark.parametrize('body', [b'not json PRIVATE', b'[]', b'x' * 11])
def test_bad_or_excessive_response_is_not_retryable(monkeypatch, body):
    calls = []
    monkeypatch.setattr(requests, 'Session', lambda: Session(Response(200, body), calls))
    with pytest.raises(ValueError, match='^system_one_response_invalid$'):
        JevPilotHTTPTransport(enabled=True, max_response_bytes=10)(DEFAULT_SYSTEM_ONE_URL, 'key', {}, 3)
    assert len(calls) == 1


def test_real_client_retries_are_metered_at_http_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr('alphonse.agent_v2.system_one.sleep', lambda _: None)
    calls = []
    good = {'answers': {'q': {'type': 'noul', 'noul': .5}}, 'usage': {'input_tokens': 5, 'output_tokens': 2}}
    responses = iter([Response(503), Response(200, json.dumps(good).encode())])
    monkeypatch.setattr(requests, 'Session', lambda: Session(next(responses), calls))
    ledger = JevAttemptLedger(str(tmp_path / 'attempts.db'), max_attempts=2, max_payload_bytes=5000)
    transport = MeteredJevTransport(JevPilotHTTPTransport(enabled=True), ledger)
    client = TypeSafeSystemOneClient(api_url=DEFAULT_SYSTEM_ONE_URL, api_key='fixture-key', model='fixture-model', transport=transport)
    assert client.evaluate(state='synthetic', questions={'q': {'type': 'noul', 'instructions': 'is this a test?'}}) == good
    assert len(calls) == 2
    assert [x['state'] for x in ledger.snapshot()] == ['failed', 'returned']
    assert [x['input_tokens'] for x in ledger.snapshot()] == [None, 5]


def test_non_ascii_meter_matches_exact_wire_bytes_and_blocks_over_limit(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(requests, 'Session', lambda: Session(Response(200, b'{"answers": {}}'), calls))
    payload = {'model': 'jev-test', 'state': 'mañana 東京', 'questions': {'q': {'type': 'noul'}}}
    wire = serialize_payload(payload)
    ledger = JevAttemptLedger(str(tmp_path / 'attempts.db'), max_attempts=2, max_payload_bytes=len(wire) - 1)
    transport = MeteredJevTransport(JevPilotHTTPTransport(enabled=True), ledger)
    with pytest.raises(ValueError, match='attempt_limit_exceeded'):
        transport(DEFAULT_SYSTEM_ONE_URL, 'key', payload, 2)
    assert not calls and ledger.snapshot() == []
    ledger2 = JevAttemptLedger(str(tmp_path / 'attempts2.db'), max_attempts=2, max_payload_bytes=len(wire))
    MeteredJevTransport(JevPilotHTTPTransport(enabled=True), ledger2)(DEFAULT_SYSTEM_ONE_URL, 'key', payload, 2)
    assert calls[0][3]['data'] == wire
    assert ledger2.snapshot()[0]['payload_bytes'] == len(wire)


@pytest.mark.parametrize(('status', 'body', 'expected_code'), [
    (400, b'{"error": {"code": "max_tokens_exceeded", "private": "secret"}}', 'max_tokens_exceeded'),
    (422, b'{"error": {"code": "invalid_json", "private": "secret"}}', 'invalid_json'),
    (413, b'not JSON secret', ''),
])
def test_meter_preserves_sanitized_nonretryable_request_error(monkeypatch, tmp_path, status, body, expected_code):
    calls = []
    monkeypatch.setattr(requests, 'Session', lambda: Session(Response(status, body), calls))
    ledger = JevAttemptLedger(str(tmp_path / 'attempts.db'), max_attempts=3, max_payload_bytes=5000)
    transport = MeteredJevTransport(JevPilotHTTPTransport(enabled=True), ledger)
    client = TypeSafeSystemOneClient(api_url=DEFAULT_SYSTEM_ONE_URL, api_key='private-key', model='fixture-model', transport=transport)
    with pytest.raises(SystemOneRequestError) as caught:
        client.evaluate(state='synthetic', questions={'q': {'type': 'noul', 'instructions': 'test'}})
    assert caught.value.http_status == status and caught.value.provider_code == expected_code
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert 'secret' not in str(caught.value) and 'private-key' not in json.dumps(ledger.snapshot())
    assert len(calls) == 1


def test_meter_preserves_sanitized_invalid_api_key_error(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(requests, 'Session', lambda: Session(Response(401), calls))
    ledger = JevAttemptLedger(str(tmp_path / 'attempts.db'), max_attempts=2, max_payload_bytes=5000)
    transport = MeteredJevTransport(JevPilotHTTPTransport(enabled=True), ledger)
    with pytest.raises(ValueError, match='^system_one_api_key_invalid$') as caught:
        transport(DEFAULT_SYSTEM_ONE_URL, 'private-key', {'questions': {'q': {}}}, 2)
    assert caught.value.__context__ is None and len(calls) == 1
