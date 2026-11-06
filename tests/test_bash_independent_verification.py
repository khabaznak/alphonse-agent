"""Disposable independent read-back checks for the native Bash boundary."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
from alphonse.agent_v2.core.tools.registry.native import bash_verification


def context(root, *, effects=('project_mutation',), allowed=('notes.txt',), goal='Create notes', external=False):
    task = TaskState(task_id='task', user='owner', goal=goal, hierarchical_state={
        'active_subgoal_id': 'write', 'phase': {'subgoals': [{'subgoal_id': 'write', 'allowed_side_effects': list(effects)}],
        'mutation_scope': {'allowed_paths': list(allowed), 'allow_external_effects': external}},
    })
    resolver = SimpleNamespace(ordinary_workspace=lambda actor, **_kwargs: root if actor == 'owner' else None)
    return ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), identity_resolver=resolver)


def test_scoped_workspace_is_read_independently_before_and_after(tmp_path):
    spec = {'source': 'workspace_text', 'path': 'notes.txt', 'expected': 'verified', 'require_change': True}
    result = execute_bash({'command': "printf verified > notes.txt", 'verification': spec}, context=context(tmp_path))
    assert result['verification']['status'] == 'verified'
    assert result['verification']['changed'] is True
    assert result['verification']['observed'] == 'verified'
    assert result['verification']['resource'] == {'path': 'notes.txt'}
    assert (tmp_path / 'notes.txt').read_text() == 'verified'
    # A claimed success with no actual change is not an effect receipt.
    replay = execute_bash({'command': "printf '{\"verified\":true}'", 'verification': spec}, context=context(tmp_path))
    assert replay['exit_code'] == 0 and replay['verification_failed'] is True
    assert replay['verification']['status'] == 'unavailable'


@pytest.mark.parametrize('path', ['other.txt', '../outside.txt', '/tmp/outside.txt'])
def test_wrong_or_out_of_scope_workspace_resource_never_dispatches(tmp_path, path):
    spec = {'source': 'workspace_text', 'path': path, 'expected': 'verified', 'require_change': True}
    with pytest.raises(PermissionError):
        execute_bash({'command': 'touch dispatched', 'verification': spec}, context=context(tmp_path))
    assert not (tmp_path / 'dispatched').exists()


def test_symlink_escape_never_dispatches(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.write_text('old')
    (tmp_path / 'notes.txt').symlink_to(outside)
    spec = {'source': 'workspace_text', 'path': 'notes.txt', 'expected': 'verified', 'require_change': True}
    with pytest.raises(PermissionError):
        execute_bash({'command': 'touch dispatched', 'verification': spec}, context=context(tmp_path))
    assert not (tmp_path / 'dispatched').exists() and outside.read_text() == 'old'


def test_public_confirmation_requires_user_origin_and_action_nonce(tmp_path, monkeypatch):
    token_path = tmp_path / 'nonce.txt'
    def read(_url):
        return {'_verification_protocol': 'nonce-v1', 'name': 'Alex' if token_path.exists() else None,
                'nonce': token_path.read_text() if token_path.exists() else None,
                'event_id': 'new-event' if token_path.exists() else None}
    monkeypatch.setattr(bash_verification, '_http_json', read)
    spec = {'source': 'public_json', 'url': 'https://example.test/receipt', 'pointer': '/name',
            'nonce_pointer': '/nonce', 'event_pointer': '/event_id', 'expected': 'Alex', 'require_change': True}
    ctx = context(tmp_path, effects=('external_irreversible',), goal='Register Alex at https://example.test', external=True)
    result = execute_bash({'command': 'printf %s "$ALPHONSE_VERIFICATION_NONCE" > nonce.txt', 'verification': spec}, context=ctx)
    assert result['verification']['status'] == 'verified' and result['verification']['nonce_checked'] is True
    assert result['verification']['resource']['url'] == 'https://example.test/receipt'
    token_path.unlink()
    monkeypatch.setattr(bash_verification, '_http_json', lambda _url: {'_verification_protocol': 'nonce-v1', 'name': 'Alex', 'nonce': 'stale', 'event_id': 'old-event'})
    result = execute_bash({'command': 'true', 'verification': spec}, context=ctx)
    assert result['verification_failed'] is True
    spec['url'] = 'https://other.test/receipt'
    with pytest.raises(PermissionError):
        execute_bash({'command': 'touch dispatched', 'verification': spec}, context=ctx)
    assert not (tmp_path / 'dispatched').exists()


def test_nonpublic_dns_is_rejected_before_connect(monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 80))])
    with pytest.raises(ValueError, match='not_public'):
        bash_verification._http_json('http://example.test/receipt')


def test_act_rejects_stale_or_wrong_identity_receipts():
    from alphonse.agent_v2.core.intelligence.v3.act_evidence import observation
    proof = {'status': 'verified', 'source': 'bash_independent_readback',
             'target': {'kind': 'workspace_text', 'id': 'a' * 64}, 'resource': {'path': 'notes.txt'}, 'observed': 'value',
             'changed': True, 'actor_user_id': 'owner', 'task_id': 'task',
             'observed_at': datetime.now(timezone.utc).isoformat()}
    entry = {'evidence_ref': 'tactical-action:write', 'status': 'success', 'tool_id': 'native.bash',
             'invocation': {'call_id': 'write', 'task_understanding': {'input_digest': 'current'}},
             'arguments': {'command': 'opaque', 'verification': {'source': 'workspace_text', 'path': 'notes.txt', 'expected': 'value', 'require_change': True}},
             'result': {'exit_code': 0, 'stdout': '', 'stderr': '', 'verification': proof}}
    def row():
        return observation(entry, ['criterion'], task_id='task', actor_user_id='owner', input_digest='current')
    assert row()['effect_status'] == 'verified_persisted_effect'
    for field, bad in [('actor_user_id', 'other'), ('task_id', 'other'), ('observed_at', (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat())]:
        good = proof[field]
        proof[field] = bad
        assert row()['effect_status'] != 'verified_persisted_effect'
        proof[field] = good
    proof['resource'] = {'path': 'unrelated.txt'}
    assert row()['effect_status'] != 'verified_persisted_effect'
    proof['resource'] = {'path': 'notes.txt'}
    entry['invocation']['task_understanding']['input_digest'] = 'stale'
    assert row()['effect_status'] != 'verified_persisted_effect'


def test_unavailable_post_write_readback_is_failed_not_retried(tmp_path, monkeypatch):
    seen = []
    def read(_url):
        seen.append(True)
        if len(seen) == 1:
            return {'_verification_protocol': 'nonce-v1', 'name': None, 'nonce': None, 'event_id': None}
        raise TimeoutError('fixture unavailable')
    monkeypatch.setattr(bash_verification, '_http_json', read)
    spec = {'source': 'public_json', 'url': 'https://example.test/receipt', 'pointer': '/name',
            'nonce_pointer': '/nonce', 'event_pointer': '/event_id', 'expected': 'Alex', 'require_change': True}
    ctx = context(tmp_path, effects=('external_irreversible',), goal='Register Alex at https://example.test', external=True)
    result = execute_bash({'command': 'printf once >> attempts.txt', 'verification': spec}, context=ctx)
    assert result['verification_failed'] is True
    assert result['verification']['status'] == 'unavailable'
    assert (tmp_path / 'attempts.txt').read_text() == 'once'
    from alphonse.agent_v2.core.tools.invocation import _bash_result_error
    assert 'reconciled before retrying' in _bash_result_error('native.bash', result)


@pytest.mark.parametrize('failure', ['{\"status\":\"failed\"}', '{\"verified\":false}', 'HTTP Error 403: denied'])
def test_explicit_action_failure_cannot_be_overridden_by_readback(tmp_path, failure):
    import shlex
    spec = {'source': 'workspace_text', 'path': 'notes.txt', 'expected': 'verified', 'require_change': True}
    command = 'printf verified > notes.txt; printf %s ' + shlex.quote(failure)
    result = execute_bash({'command': command, 'verification': spec}, context=context(tmp_path))
    assert (tmp_path / 'notes.txt').read_text() == 'verified'
    assert result['verification_failed'] is True
    assert result['verification']['status'] == 'unavailable'


def test_unadvertised_remote_verification_never_dispatches(tmp_path, monkeypatch):
    monkeypatch.setattr(bash_verification, '_http_json', lambda _url: {'name': None, 'nonce': None})
    spec = {'source': 'public_json', 'url': 'https://example.test/receipt', 'pointer': '/name',
            'nonce_pointer': '/nonce', 'event_pointer': '/event_id', 'expected': 'Alex', 'require_change': True}
    ctx = context(tmp_path, effects=('external_irreversible',), goal='Register Alex at https://example.test', external=True)
    with pytest.raises(ValueError, match='protocol_unavailable'):
        execute_bash({'command': 'touch dispatched', 'verification': spec}, context=ctx)
    assert not (tmp_path / 'dispatched').exists()


def test_public_receipt_requires_a_new_event_even_if_name_changes(tmp_path, monkeypatch):
    token_path = tmp_path / 'nonce.txt'
    def read(_url):
        return {'_verification_protocol': 'nonce-v1', 'name': 'Alex' if token_path.exists() else None,
                'nonce': token_path.read_text() if token_path.exists() else None, 'event_id': 'old-event'}
    monkeypatch.setattr(bash_verification, '_http_json', read)
    spec = {'source': 'public_json', 'url': 'https://example.test/receipt', 'pointer': '/name',
            'nonce_pointer': '/nonce', 'event_pointer': '/event_id', 'expected': 'Alex', 'require_change': True}
    ctx = context(tmp_path, effects=('external_irreversible',), goal='Register Alex at https://example.test', external=True)
    result = execute_bash({'command': 'printf %s "$ALPHONSE_VERIFICATION_NONCE" > nonce.txt', 'verification': spec}, context=ctx)
    assert result['verification_failed'] is True


def test_public_source_deadline_is_bounded(monkeypatch):
    import threading
    release = threading.Event()
    monkeypatch.setattr(bash_verification, '_TOTAL_HTTP_SECONDS', .02)
    monkeypatch.setattr(bash_verification, '_read_public_json', lambda _url: release.wait(1))
    try:
        with pytest.raises(ValueError, match='source_timeout'):
            bash_verification._http_json('https://example.test/receipt')
    finally:
        release.set()


def test_truncated_action_output_cannot_hide_a_failure(tmp_path):
    spec = {'source': 'workspace_text', 'path': 'notes.txt', 'expected': 'verified', 'require_change': True}
    command = "printf verified > notes.txt; python -c 'print(\"x\" * 20000)'"
    result = execute_bash({'command': command, 'verification': spec}, context=context(tmp_path))
    assert result['stdout'].endswith('... [truncated]')
    assert result['verification_failed'] is True
    assert result['verification']['status'] == 'unavailable'


def test_pinned_http_reader_uses_checked_address_and_rejects_redirects_and_large_bodies(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    hosts = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hosts.append(self.headers['Host'])
            if self.path == '/redirect':
                self.send_response(302)
                self.send_header('Location', 'http://127.0.0.1/private')
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'x' * (bash_verification._LIMIT + 1) if self.path == '/large' else b'{"value": 23}')
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(bash_verification, '_resolve_public', lambda host, port: '127.0.0.1')
    url = f'http://fixture.example:{server.server_port}'
    try:
        assert bash_verification._http_json(url + '/ok') == {'value': 23}
        with pytest.raises(ValueError, match='source_unavailable'):
            bash_verification._http_json(url + '/redirect')
        with pytest.raises(ValueError, match='source_too_large'):
            bash_verification._http_json(url + '/large')
        assert hosts == [f'fixture.example:{server.server_port}'] * 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_public_confirmation_rejects_plain_http_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(bash_verification, '_http_json', lambda _url: {})
    spec = {'source': 'public_json', 'url': 'http://example.test/receipt', 'pointer': '/name',
            'nonce_pointer': '/nonce', 'event_pointer': '/event_id', 'expected': 'Alex', 'require_change': True}
    ctx = context(tmp_path, effects=('external_irreversible',), goal='Register Alex at http://example.test', external=True)
    with pytest.raises(ValueError, match='https_required'):
        execute_bash({'command': 'touch dispatched', 'verification': spec}, context=ctx)
    assert not (tmp_path / 'dispatched').exists()


def test_public_nonce_cannot_double_as_event_identity(tmp_path, monkeypatch):
    token = tmp_path / 'nonce.txt'
    def read(_url):
        value = token.read_text() if token.exists() else None
        return {'_verification_protocol': 'nonce-v1', 'name': 'Alex' if value else None,
                'nonce': value, 'event_id': value}
    monkeypatch.setattr(bash_verification, '_http_json', read)
    spec = {'source': 'public_json', 'url': 'https://example.test/receipt', 'pointer': '/name',
            'nonce_pointer': '/nonce', 'event_pointer': '/event_id', 'expected': 'Alex', 'require_change': True}
    ctx = context(tmp_path, effects=('external_irreversible',), goal='Register Alex at https://example.test', external=True)
    result = execute_bash({'command': 'printf %s "$ALPHONSE_VERIFICATION_NONCE" > nonce.txt', 'verification': spec}, context=ctx)
    assert result['verification_failed'] is True
    spec['event_pointer'] = '/nonce'
    with pytest.raises(ValueError, match='pointers_invalid'):
        execute_bash({'command': 'true', 'verification': spec}, context=ctx)


def test_ordinary_output_containing_a_denial_word_is_not_itself_a_failure(tmp_path):
    spec = {'source': 'workspace_text', 'path': 'notes.txt', 'expected': 'verified', 'require_change': True}
    result = execute_bash({'command': 'printf verified > notes.txt; printf forbidden', 'verification': spec}, context=context(tmp_path))
    assert result['verification']['status'] == 'verified'


def test_workspace_access_revoked_after_write_fails_closed(tmp_path):
    ctx = context(tmp_path)
    rechecks = []
    def resolve(actor, *, create=True):
        assert actor == 'owner'
        if not create:
            rechecks.append(True)
            if len(rechecks) > 1:
                raise PermissionError('owner access revoked')
        return tmp_path
    ctx.identity_resolver.ordinary_workspace = resolve
    spec = {'source': 'workspace_text', 'path': 'notes.txt', 'expected': 'verified', 'require_change': True}
    result = execute_bash({'command': 'printf verified > notes.txt', 'verification': spec}, context=ctx)
    assert (tmp_path / 'notes.txt').read_text() == 'verified'
    assert result['verification_failed'] is True
    assert result['verification']['status'] == 'unavailable'


def test_workspace_verification_requires_a_change_before_dispatch(tmp_path):
    spec = {'source': 'workspace_text', 'path': 'notes.txt', 'expected': 'verified', 'require_change': False}
    with pytest.raises(PermissionError):
        execute_bash({'command': 'touch dispatched', 'verification': spec}, context=context(tmp_path))
    assert not (tmp_path / 'dispatched').exists()
