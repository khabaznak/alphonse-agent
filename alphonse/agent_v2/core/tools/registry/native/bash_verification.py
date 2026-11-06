"""Independent, bounded read-back for explicitly scoped Bash effects.

The verifier is harness code; program output never supplies the observation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import threading
import queue
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

_LIMIT = 1024 * 1024
_NETWORK_TIMEOUT_SECONDS = 5.0
_TOTAL_HTTP_SECONDS = 15.0
_NETWORK_SLOTS = threading.BoundedSemaphore(2)
_DNS_SLOTS = threading.BoundedSemaphore(2)
_MISSING = object()


def _phase(context):
    state = context.task.hierarchical_state or {}
    phase = state.get('phase') or {}
    goal = next((item for item in phase.get('subgoals', []) if item.get('subgoal_id') == state.get('active_subgoal_id')), {})
    return phase, set(goal.get('allowed_side_effects') or [])


def _origin(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('verification_url_invalid')
    return parsed.scheme, parsed.hostname.casefold(), parsed.port or (443 if parsed.scheme == 'https' else 80)


def _require_secure_public(url):
    if _origin(url)[0] != 'https':
        raise ValueError('verification_https_required')


def _resolve_public(hostname, port):
    import ipaddress
    import socket

    if not _DNS_SLOTS.acquire(blocking=False):
        raise ValueError('verification_dns_busy')
    result = queue.Queue(maxsize=1)
    def resolve():
        try:
            result.put((True, socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)))
        except Exception as exc:
            result.put((False, exc))
        finally:
            _DNS_SLOTS.release()
    threading.Thread(target=resolve, daemon=True).start()
    try:
        ok, addresses = result.get(timeout=_NETWORK_TIMEOUT_SECONDS)
    except queue.Empty:
        raise ValueError('verification_dns_timeout') from None
    if not ok:
        raise ValueError('verification_dns_unavailable') from None
    if not addresses or any(not ipaddress.ip_address(row[4][0].split('%')[0]).is_global for row in addresses):
        raise ValueError('verification_url_not_public')
    return addresses[0][4][0].split('%')[0]


def _read_public_json(url):
    # Resolve once, reject every non-public address, and connect to the checked
    # IP directly. TLS still verifies the original hostname. No redirects.
    import http.client
    import socket
    import ssl
    from urllib.parse import urlunsplit

    parsed = urlsplit(url)
    scheme, hostname, port = _origin(url)
    if parsed.fragment or len(url) > 4096:
        raise ValueError('verification_url_invalid')
    selected_ip = _resolve_public(hostname, port)

    class PinnedConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.create_connection((selected_ip, port), self.timeout)
            if scheme == 'https':
                self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=hostname)

    connection = PinnedConnection(hostname, port, timeout=_NETWORK_TIMEOUT_SECONDS)
    try:
        path = urlunsplit(('', '', parsed.path or '/', parsed.query, ''))
        connection.request('GET', path, headers={'Accept': 'application/json'})
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            raise ValueError('verification_source_unavailable')
        deadline = time.monotonic() + _NETWORK_TIMEOUT_SECONDS
        chunks, size = [], 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError('verification_source_timeout')
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            chunk = response.read1(min(16384, _LIMIT + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > _LIMIT:
                raise ValueError('verification_source_too_large')
            chunks.append(chunk)
        return json.loads(b''.join(chunks))
    finally:
        connection.close()


def _http_json(url):
    # The deadline also bounds slow DNS/TLS/header streams. A fixed number of
    # daemon workers prevents slow peers from creating unbounded background work.
    if not _NETWORK_SLOTS.acquire(blocking=False):
        raise ValueError('verification_network_busy')
    result = queue.Queue(maxsize=1)
    def read():
        try:
            result.put((True, _read_public_json(url)))
        except Exception as exc:
            result.put((False, exc))
        finally:
            _NETWORK_SLOTS.release()
    threading.Thread(target=read, daemon=True).start()
    try:
        ok, value = result.get(timeout=_TOTAL_HTTP_SECONDS)
    except queue.Empty:
        raise ValueError('verification_source_timeout') from None
    if not ok:
        if isinstance(value, ValueError):
            raise ValueError(str(value)) from None
        raise ValueError('verification_source_unavailable') from None
    return value


def _pointer(value, pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/') or len(pointer) > 1024:
        raise ValueError('verification_pointer_invalid')
    for token in pointer[1:].split('/'):
        if re.search(r'~(?![01])', token):
            raise ValueError('verification_pointer_invalid')
        token = token.replace('~1', '/').replace('~0', '~')
        if not isinstance(value, dict) or token not in value:
            return _MISSING
        value = value[token]
    return value


def _read_file(path, *, missing_ok):
    try:
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError('verification_source_changed')
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    except FileNotFoundError:
        if missing_ok:
            return _MISSING
        raise ValueError('verification_source_unavailable') from None
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        current = os.stat(path, follow_symlinks=False)
        if (not stat.S_ISREG(opened.st_mode) or path.resolve(strict=True) != path
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
            raise ValueError('verification_source_changed')
        data = stream.read(_LIMIT + 1)
    if len(data) > _LIMIT:
        raise ValueError('verification_source_too_large')
    return data.decode('utf-8')


def prepare_verification(spec, arguments, context, prepared, cwd):
    if context is None or not isinstance(spec, dict):
        raise ValueError('verification_context_required')
    phase, effects = _phase(context)
    source = spec.get('source')
    workspace_root = None
    allowed_keys = {'source', 'binding_name', 'path', 'url', 'pointer', 'nonce_pointer', 'event_pointer', 'expected', 'require_change'}
    if set(spec) - allowed_keys or ('require_change' in spec and type(spec['require_change']) is not bool):
        raise ValueError('verification_spec_invalid')
    if ('expected' not in spec or type(spec['expected']) not in (str, int, float, bool, type(None))
            or (isinstance(spec['expected'], str) and len(spec['expected']) > 4096)
            or (type(spec['expected']) is float and not math.isfinite(spec['expected']))):
        raise ValueError('verification_expected_scalar_required')
    if source == 'skill_data_json':
        name = spec.get('binding_name')
        selected = prepared.values.get(name) if isinstance(name, str) else None
        if (not selected or selected.get('kind') != 'data'
                or (selected.get('configuration') or {}).get('alphonse_verification_authority') != 'resource_state'
                or name not in arguments.get('binding_names', []) or not arguments.get('skill_id') or not effects.intersection({'project_mutation', 'external_reversible', 'external_irreversible'})
                or not phase.get('mutation_scope', {}).get('allow_external_effects')):
            raise PermissionError('verification_binding_or_scope_denied')
        path = Path(selected['path'])
        if not path.is_file():
            raise ValueError('verification_source_unavailable')
        identity = ['skill_data_json', arguments['skill_id'], name, str(path), spec.get('pointer')]
        resource = {'skill_id': arguments['skill_id'], 'binding_name': name, 'pointer': spec.get('pointer')}
    elif source == 'workspace_text':
        relative = spec.get('path')
        path_part = Path(relative) if isinstance(relative, str) else Path('/')
        allowed = phase.get('mutation_scope', {}).get('allowed_paths', [])
        if (path_part.is_absolute() or '..' in path_part.parts or not relative or relative not in allowed
                or 'project_mutation' not in effects or arguments.get('skill_id')
                or spec.get('require_change') is not True):
            raise PermissionError('verification_path_or_scope_denied')
        from alphonse.agent_v2.core.tools.registry.native.bash import _resolve_cwd
        root = Path(cwd).resolve(strict=True)
        if root != Path(_resolve_cwd(None, context=context)).resolve(strict=True):
            raise PermissionError('verification_path_or_scope_denied')
        workspace_root = str(root)
        path = root / path_part
        if not path.resolve(strict=False).is_relative_to(root):
            raise PermissionError('verification_path_or_scope_denied')
        identity = ['workspace_text', str(root), relative]
        resource = {'path': relative}
    elif source == 'public_json':
        url = spec.get('url')
        if isinstance(url, str):
            _require_secure_public(url)
            parsed = urlsplit(url)
            if parsed.query or parsed.fragment:
                raise ValueError('verification_url_query_denied')
        pointers = (spec.get('pointer'), spec.get('nonce_pointer'), spec.get('event_pointer'))
        if not all(isinstance(item, str) and item.startswith('/') for item in pointers) or len(set(pointers)) != 3:
            raise ValueError('verification_pointers_invalid')
        request_urls = re.findall(r'https?://[^\s<>"\']+', str(context.task.goal or ''))
        if (not isinstance(url, str) or not any(_origin(url) == _origin(item.rstrip('.,;')) for item in request_urls)
                or not effects.intersection({'external_reversible', 'external_irreversible'})
                or not phase.get('mutation_scope', {}).get('allow_external_effects')
                or spec.get('require_change') is not True or not spec.get('nonce_pointer') or not spec.get('event_pointer')):
            raise PermissionError('verification_url_or_scope_denied')
        identity = ['public_json', url, spec.get('pointer'), spec.get('nonce_pointer'), spec.get('event_pointer')]
        resource = {'url': url, 'pointer': spec.get('pointer'), 'event_pointer': spec.get('event_pointer')}
        path = None
    else:
        raise ValueError('verification_source_invalid')
    fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
    nonce = secrets.token_urlsafe(32) if source == 'public_json' else ''
    return {'spec': spec, 'path': path, 'workspace_root': workspace_root, 'fingerprint': fingerprint, 'nonce': nonce, 'resource': resource,
            'skill_id': arguments.get('skill_id'), 'binding_name': spec.get('binding_name')}


def _current_workspace(task, context):
    from alphonse.agent_v2.core.tools.registry.native.bash import _project_root_from_context
    if str(task.project_id or '').strip():
        root = _project_root_from_context(context)
        if not root:
            raise PermissionError('verification_workspace_unavailable')
        return Path(root).resolve(strict=True)
    resolver = getattr(context.identity_resolver, 'ordinary_workspace', None)
    if not callable(resolver):
        raise PermissionError('verification_workspace_unavailable')
    return Path(resolver(str(task.user), create=False)).resolve(strict=True)


def workspace_still_authorized(*, task, arguments, context, target):
    try:
        spec = arguments.get('verification') or {}
        if context is None or arguments.get('skill_id') or not spec.get('path'):
            return False
        root = _current_workspace(task, context.tool_execution_context(task))
        identity = ['workspace_text', str(root), spec['path']]
        fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        return isinstance(target, dict) and target.get('kind') == 'workspace_text' and secrets.compare_digest(str(target.get('id') or ''), fingerprint)
    except Exception:
        return False


def binding_still_authorized(*, task, arguments, context, target):
    """Revalidate an actor's binding before old read-back is shown to Act."""
    spec = arguments.get('verification') or {}
    skill_id, name = arguments.get('skill_id'), spec.get('binding_name')
    if (context is None or context.skill_bindings is None or not skill_id or not name
            or name not in arguments.get('binding_names', [])
            or skill_id not in (task.metadata.get('v3_skill_selection') or {}).get('selected_skill_ids', [])):
        return False
    try:
        current = context.skill_bindings.prepare(actor_id=str(task.user), skill_id=skill_id, names=[name]).values[name]
        if (current.get('kind') != 'data'
                or (current.get('configuration') or {}).get('alphonse_verification_authority') != 'resource_state'):
            return False
        identity = ['skill_data_json', skill_id, name, str(Path(current['path'])), spec.get('pointer')]
        fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        return isinstance(target, dict) and target.get('kind') == 'skill_data_json' and secrets.compare_digest(str(target.get('id') or ''), fingerprint)
    except Exception:
        return False


def observe(prepared_verification, *, before, context):
    spec = prepared_verification['spec']
    source = spec['source']
    if source == 'workspace_text' and str(_current_workspace(context.task, context)) != prepared_verification['workspace_root']:
        raise PermissionError('verification_workspace_changed')
    if source == 'skill_data_json':
        # Recheck binding ACL and stable resource identity at each read.
        current = context.skill_bindings.prepare(actor_id=str(context.task.user), skill_id=prepared_verification['skill_id'],
                                                  names=[prepared_verification['binding_name']])
        binding = current.values[prepared_verification['binding_name']]
        if (binding.get('kind') != 'data' or binding.get('path') != str(prepared_verification['path'])
                or (binding.get('configuration') or {}).get('alphonse_verification_authority') != 'resource_state'):
            raise PermissionError('verification_binding_changed')
    if source == 'public_json':
        data = _http_json(spec['url'])
        if not isinstance(data, dict) or data.get('_verification_protocol') != 'nonce-v1':
            raise ValueError('verification_protocol_unavailable')
        value = _pointer(data, spec.get('pointer'))
        nonce_value = _pointer(data, spec.get('nonce_pointer'))
        event_value = _pointer(data, spec.get('event_pointer'))
    else:
        raw = _read_file(prepared_verification['path'], missing_ok=before and source == 'workspace_text')
        if raw is _MISSING:
            return {'value': _MISSING, 'nonce': _MISSING, 'event': _MISSING}
        if source == 'workspace_text':
            value, nonce_value, event_value = raw, _MISSING, _MISSING
        else:
            value, nonce_value, event_value = _pointer(json.loads(raw), spec.get('pointer')), _MISSING, _MISSING
    return {'value': value, 'nonce': nonce_value, 'event': event_value}


def output_reports_failure(stdout, stderr):
    """Explicit failures remain failures even if a process exits zero."""
    text = f'{stdout or ""}\n{stderr or ""}'
    if any(isinstance(value, str) and value.endswith('... [truncated]') for value in (stdout, stderr)):
        return True
    if (re.search(r'\b(?:permission denied|access denied|forbidden|unauthorized)\b', str(stderr or ''), re.IGNORECASE)
            or re.search(r'\bHTTP(?:\s*Error|/\d+(?:\.\d+)?)?\s*[: ]\s*[45]\d\d\b', text, re.IGNORECASE)):
        return True
    try:
        payload = json.loads(stdout)
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    status = payload.get('status')
    return ((isinstance(status, str) and status in {'failed', 'error', 'denied'})
            or payload.get('success') is False or payload.get('ok') is False
            or payload.get('verified') is False or payload.get('acknowledged') is False
            or bool(payload.get('error')))


def finish_verification(prepared_verification, before, after, context):
    spec = prepared_verification['spec']
    expected, actual = spec['expected'], after['value']
    same = actual is not _MISSING and type(actual) is type(expected) and actual == expected
    if type(actual) in (int, float) and type(expected) in (int, float):
        same = actual == expected
    event = after['event']
    if spec['source'] == 'public_json':
        changed = (isinstance(event, str) and 0 < len(event) <= 256
                   and event != before['event'] and event != prepared_verification['nonce'])
    else:
        changed = before['value'] is _MISSING or before['value'] != actual
    nonce_ok = spec['source'] != 'public_json' or after['nonce'] == prepared_verification['nonce']
    if not same or (spec.get('require_change') and not changed) or not nonce_ok:
        raise ValueError('verification_mismatch')
    return {'status': 'verified', 'source': 'bash_independent_readback',
            'target': {'kind': spec['source'], 'id': prepared_verification['fingerprint']},
            'resource': prepared_verification['resource'], 'observed': actual, 'changed': changed, 'nonce_checked': spec['source'] == 'public_json',
            'observed_at': datetime.now(timezone.utc).isoformat(),
            **({'event_id_hash': hashlib.sha256(event.encode()).hexdigest()} if spec['source'] == 'public_json' else {}),
            'actor_user_id': str(context.task.user), 'task_id': str(context.task.task_id)}
