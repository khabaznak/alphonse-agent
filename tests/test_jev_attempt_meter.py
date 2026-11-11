"""Offline tests of request-attempt accounting; no live TypeSafe calls."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from alphonse.agent_v2.evaluation.jev_attempt_meter import JevAttemptLedger, MeteredJevTransport
from alphonse.agent_v2.system_one import DEFAULT_SYSTEM_ONE_URL, TypeSafeSystemOneClient


def test_actual_client_retries_are_individual_durable_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr('alphonse.agent_v2.system_one.sleep', lambda _: None)
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=4, max_payload_bytes=4000)
    calls = []
    def underlying(url, key, payload, timeout):
        calls.append(payload)
        if len(calls) < 3:
            raise RuntimeError('system_one_connection_failed')
        return {'model': 'jev-versioned-synthetic', 'answers': {'q': {'type': 'noul', 'noul': .5}},
                'usage': {'input_tokens': 15, 'output_tokens': 4}}
    client = TypeSafeSystemOneClient(api_url=DEFAULT_SYSTEM_ONE_URL, api_key='synthetic-key', model='synthetic-model',
                                     transport=MeteredJevTransport(underlying, ledger))
    response = client.evaluate(state={'note': 'fixture-private-state'}, questions={'q': {'type': 'noul', 'instructions': 'test'}})
    assert response['usage']['input_tokens'] == 15
    rows = ledger.snapshot()
    assert [row['state'] for row in rows] == ['failed', 'failed', 'returned']
    assert [row['input_tokens'] for row in rows] == [None, None, 15]
    assert rows[0]['output_tokens'] is None
    assert rows[2]['output_tokens'] == 4
    assert rows[0]['response_model'] is None and rows[2]['response_model'] == 'jev-versioned-synthetic'
    assert all(row['input_estimated_tokens'] > 0 for row in rows)
    assert 'synthetic-key' not in json.dumps(rows)
    assert 'synthetic-model' not in json.dumps(rows)
    assert 'fixture-private-state' not in json.dumps(rows)


def test_limit_includes_uncertain_attempts_and_survives_reopen(tmp_path):
    path = str(tmp_path / 'jev.db')
    ledger = JevAttemptLedger(path, max_attempts=1, max_payload_bytes=4000)
    attempt = ledger.start(payload_bytes=2, question_count=1, estimated_input_tokens=1)
    assert attempt == 1
    reopened = JevAttemptLedger(path, max_attempts=1, max_payload_bytes=4000)
    assert reopened.snapshot()[0]['state'] == 'started'
    called = []
    transport = MeteredJevTransport(lambda *a: called.append(True), reopened)
    with pytest.raises(ValueError, match='jev_pilot_attempt_limit_exceeded'):
        transport(DEFAULT_SYSTEM_ONE_URL, 'key', {'questions': {'q': {}}}, 1)
    assert called == []
    with pytest.raises(ValueError, match='jev_limits_changed'):
        JevAttemptLedger(path, max_attempts=2, max_payload_bytes=4000)


def test_usage_partial_invalid_and_failures_are_never_zero(tmp_path):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=3, max_payload_bytes=4000)
    cases = iter([{'answers': {}, 'usage': {'input_tokens': 7, 'output_tokens': True}}, {'answers': {}}, None])
    transport = MeteredJevTransport(lambda *a: next(cases), ledger)
    payload = {'questions': {'q': {}}}
    transport(DEFAULT_SYSTEM_ONE_URL, 'key', payload, 1)
    transport(DEFAULT_SYSTEM_ONE_URL, 'key', payload, 1)
    with pytest.raises(ValueError, match='system_one_response_invalid'):
        transport(DEFAULT_SYSTEM_ONE_URL, 'key', payload, 1)
    rows = ledger.snapshot()
    assert [(row['input_tokens'], row['output_tokens']) for row in rows] == [(7, None), (None, None), (None, None)]
    assert [row['state'] for row in rows] == ['returned', 'returned', 'invalid']


def test_returned_model_is_never_inferred_and_rejects_unsanitized_identifiers(tmp_path):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=2, max_payload_bytes=4000)
    cases = iter([{'answers': {}}, {'model': 'secret with whitespace', 'answers': {}}])
    transport = MeteredJevTransport(lambda *a: next(cases), ledger)
    for _ in range(2):
        transport(DEFAULT_SYSTEM_ONE_URL, 'key', {'questions': {'q': {}}}, 1)
    assert [row['response_model'] for row in ledger.snapshot()] == [None, None]


def test_returned_model_cannot_retain_echoed_api_key(tmp_path):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=1, max_payload_bytes=4000)
    transport = MeteredJevTransport(lambda *a: {'model': 'fixture-api-key', 'answers': {}}, ledger)
    transport(DEFAULT_SYSTEM_ONE_URL, 'fixture-api-key', {'questions': {'q': {}}}, 1)
    assert ledger.snapshot()[0]['response_model'] is None
    assert 'fixture-api-key' not in json.dumps(ledger.snapshot())


def test_payload_and_endpoint_limits_prevent_dispatch(tmp_path):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=1, max_payload_bytes=5)
    calls = []
    transport = MeteredJevTransport(lambda *a: calls.append(True), ledger)
    with pytest.raises(ValueError, match='endpoint_denied'):
        transport('https://attacker.invalid', 'key', {'questions': {'q': {}}}, 1)
    with pytest.raises(ValueError, match='attempt_limit_exceeded'):
        transport(DEFAULT_SYSTEM_ONE_URL, 'key', {'questions': {'q': {}}}, 1)
    assert calls == [] and ledger.snapshot() == []


def test_meter_passes_detached_snapshot_of_the_measured_request(tmp_path):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=1, max_payload_bytes=4000)
    original = {'state': {'text': 'original'}, 'questions': {'q': {}}}
    observed = []
    def transport(_url, _key, payload, _timeout):
        original['state']['text'] = 'changed after reservation'
        observed.append(payload['state']['text'])
        return {'answers': {}}
    MeteredJevTransport(transport, ledger)(DEFAULT_SYSTEM_ONE_URL, 'key', original, 1)
    assert observed == ['original']


def test_concurrent_attempt_limit_is_atomic(tmp_path):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=3, max_payload_bytes=4000)
    def start(_):
        try:
            ledger.start(payload_bytes=5, question_count=1, estimated_input_tokens=2)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(start, range(12))) == 3
    assert len(ledger.snapshot()) == 3


def test_transport_errors_do_not_leak_secrets_and_retry_code_preserved(tmp_path):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=2, max_payload_bytes=4000)
    def fail(*_):
        try:
            raise RuntimeError('secret key and source')
        except RuntimeError:
            raise RuntimeError('system_one_connection_failed')
    transport = MeteredJevTransport(fail, ledger)
    with pytest.raises(RuntimeError, match='^system_one_connection_failed$') as caught:
        transport(DEFAULT_SYSTEM_ONE_URL, 'key', {'questions': {'q': {}}}, 1)
    assert caught.value.__context__ is None
    assert 'secret' not in json.dumps(ledger.snapshot())


@pytest.mark.parametrize('checker', [lambda: True, lambda: None, lambda: (_ for _ in ()).throw(RuntimeError('private'))])
def test_cancellation_or_checker_failure_stops_before_attempt(tmp_path, checker):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=2, max_payload_bytes=4000)
    calls = []
    transport = MeteredJevTransport(lambda *a: calls.append(True), ledger, cancellation_checker=checker)
    with pytest.raises(ValueError, match='^jev_pilot_cancelled_before_dispatch$'):
        transport(DEFAULT_SYSTEM_ONE_URL, 'key', {'questions': {'q': {}}}, 1)
    assert calls == [] and ledger.snapshot() == []


def test_cancellation_after_reservation_is_recorded_without_dispatch(tmp_path):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=2, max_payload_bytes=4000)
    checks = iter([False, True])
    calls = []
    transport = MeteredJevTransport(lambda *a: calls.append(True), ledger, cancellation_checker=lambda: next(checks))
    with pytest.raises(ValueError, match='^jev_pilot_cancelled_before_dispatch$'):
        transport(DEFAULT_SYSTEM_ONE_URL, 'key', {'questions': {'q': {}}}, 1)
    assert calls == [] and ledger.snapshot()[0]['state'] == 'cancelled_before_dispatch'


def test_cancellation_after_dispatch_keeps_usage_but_returns_no_answer(tmp_path):
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=2, max_payload_bytes=4000)
    checks = iter([False, False, True])
    transport = MeteredJevTransport(lambda *a: {'answers': {'q': {}}, 'usage': {'input_tokens': 9, 'output_tokens': 2}},
                                    ledger, cancellation_checker=lambda: next(checks))
    with pytest.raises(ValueError, match='^jev_pilot_cancelled_after_dispatch$'):
        transport(DEFAULT_SYSTEM_ONE_URL, 'key', {'questions': {'q': {}}}, 1)
    row = ledger.snapshot()[0]
    assert row['state'] == 'returned_after_cancellation'
    assert (row['input_tokens'], row['output_tokens']) == (9, 2)


def test_client_retry_checks_deadline_before_second_dispatch(tmp_path, monkeypatch):
    expired = {'value': False}
    monkeypatch.setattr('alphonse.agent_v2.system_one.sleep', lambda _: expired.update(value=True))
    ledger = JevAttemptLedger(str(tmp_path / 'jev.db'), max_attempts=4, max_payload_bytes=4000)
    calls = []
    def first_failure(*args):
        calls.append(True)
        raise RuntimeError('system_one_http_error:503')
    transport = MeteredJevTransport(first_failure, ledger, cancellation_checker=lambda: expired['value'])
    client = TypeSafeSystemOneClient(api_url=DEFAULT_SYSTEM_ONE_URL, api_key='key', model='fixture', transport=transport)
    with pytest.raises(ValueError, match='^jev_pilot_cancelled_before_dispatch$'):
        client.evaluate(state='synthetic', questions={'q': {'type': 'noul', 'instructions': 'test'}})
    assert calls == [True]
    assert [row['state'] for row in ledger.snapshot()] == ['failed']
