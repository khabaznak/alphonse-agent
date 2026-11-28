from datetime import datetime, timedelta, timezone
import json

import pytest
import requests

from alphonse.agent_v2.integrations.google_mcp import GoogleMcpClient, GoogleMcpError


class Response:
    def __init__(self, content, *, status=200, content_type='application/json'):
        self.content = content
        self.status_code = status
        self.headers = {'Content-Type': content_type}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), 17):
            yield self.content[offset:offset + 17]


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        assert self.trust_env is False
        self.calls.append((url, kwargs))
        return self.response


def call(monkeypatch, response, *, budget=1024, guard=None, deadline=None):
    session = Session(response)
    monkeypatch.setattr(requests, 'Session', lambda: session)
    payload = GoogleMcpClient().call('calendar', 'list_events', {'calendarId': 'exact'},
        token='disposable', deadline=deadline or datetime.now(timezone.utc) + timedelta(seconds=5),
        max_bytes=budget, guard=guard)
    return payload, session.calls


def envelope(result):
    return json.dumps({'jsonrpc': '2.0', 'id': 1, 'result': result}).encode()


@pytest.mark.parametrize('stream', [False, True])
def test_fixed_origin_bearer_headers_and_bounded_stream(monkeypatch, stream):
    wire = envelope({'structuredContent': {'events': []}})
    if stream:
        wire = b': ping\r\n\r\nevent: message\r\ndata: ' + wire + b'\r\n\r\n'
    events = []
    (value, size), calls = call(monkeypatch, Response(wire,
        content_type='text/event-stream; charset=UTF-8' if stream else 'application/json'),
        guard=lambda: events.append('authorized'))
    assert value == {'events': []} and size == len(wire)
    assert events == ['authorized', 'authorized']
    assert len(calls) == 1
    url, request = calls[0]
    assert url == 'https://calendarmcp.googleapis.com/mcp/v1'
    assert request['allow_redirects'] is False and request['stream'] is True
    assert request['headers']['Authorization'] == 'Bearer disposable'
    assert request['json']['params']['_meta']['io.modelcontextprotocol/clientCapabilities'] == {}


@pytest.mark.parametrize(('response', 'budget', 'status', 'reason'), [
    (Response(b'', status=302), 1024, 502, 'redirect_denied'),
    (Response(b'{}'), 1, 413, 'bytes_budget'),
    (Response(envelope({'resultType': 'input_required', 'requestState': 'private'})),
     1024, 409, 'approval_required'),
    (Response(envelope({'resultType': 'complete', 'isError': True,
                       'content': [{'text': 'Requested entity was not found.'}]})), 1024, 404, 'not_found'),
    (Response(b'{"jsonrpc":"2.0","id":7,"result":{"structuredContent":{}}}'),
     1024, 502, 'response_invalid'),
])
def test_transport_returns_bounded_typed_errors(monkeypatch, response, budget, status, reason):
    with pytest.raises(GoogleMcpError) as stopped:
        call(monkeypatch, response, budget=budget)
    assert stopped.value.status == status and stopped.value.reason == reason
    assert 'private' not in str(stopped.value)
