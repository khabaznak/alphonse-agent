import json
import socket
from types import SimpleNamespace

from alphonse.agent_v2.ipc import V2DaemonServer


def request_with_dispatch(dispatch):
    server = V2DaemonServer(SimpleNamespace())
    server._dispatch = dispatch
    client, connection = socket.socketpair()
    try:
        client.sendall(b'{"version":1,"request_id":"fixture","method":"test","params":{}}\n')
        server._handle(connection)
        return json.loads(client.recv(4096))
    finally:
        client.close()


def test_dispatch_error_responds_even_if_stderr_is_unavailable(monkeypatch):
    class BrokenStderr:
        def write(self, _):
            raise BrokenPipeError('closed')
        def flush(self):
            pass
    monkeypatch.setattr('sys.stderr', BrokenStderr())
    def fail(_):
        raise PermissionError('question_respondent_required')
    result = request_with_dispatch(fail)
    assert result['ok'] is False
    assert result['error'] == 'PermissionError: question_respondent_required'


def test_non_json_result_returns_sanitized_error():
    result = request_with_dispatch(lambda _: {'private': object()})
    assert result['ok'] is False
    assert result['error'] == 'daemon_response_unavailable'
    assert result['request_id'] == 'fixture'
