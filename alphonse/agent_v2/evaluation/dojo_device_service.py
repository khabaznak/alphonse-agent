"""Synthetic local device-service Dojo; no real device or host network.

The candidate may discover an invented service in household files and query
it. Loopback exists inside the disposable container; network access remains
disabled outside it. The fixture is not a model of any commercial device API.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

from alphonse.agent_v2.evaluation.owner_dojo_shopping import run

CASE_ID = 'local-device-service'
GOAL = ('Can you check whether the front door is closed? Information about '
        'household services is in /workspace/household.')
MANIFEST = json.dumps({'service': 'household devices', 'base_url': 'http://127.0.0.1:18765',
                       'description': 'Local service for connected household devices'}, sort_keys=True) + '\n'
SERVER = '''from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with Path('/workspace/household/service-requests.jsonl').open('a') as log:
            log.write(json.dumps({'path': self.path}) + '\\n')
        if self.path == '/devices':
            body = {'devices': [{'id': 'door-01', 'name': 'front door', 'type': 'door sensor',
                                 'door_contact': 'closed', 'lock_state': 'unlocked'}]}
            code = 200
        elif self.path == '/':
            body = {'service': 'household devices', 'devices_url': '/devices'}
            code = 200
        else:
            body = {'error': 'not found'}
            code = 404
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args):
        pass
HTTPServer(('127.0.0.1', 18765), Handler).serve_forever()
'''


def seed(shell):
    program = ('import pathlib; p=pathlib.Path("/workspace/household"); p.mkdir(parents=True); '
               '(p/"device-service.json").write_text(%r); '
               'pathlib.Path("/tmp/dojo-device-service.py").write_text(%r)' % (MANIFEST, SERVER))
    command = ('/usr/local/bin/python -I -c ' + shlex.quote(program)
               + ' && nohup /usr/local/bin/python -I /tmp/dojo-device-service.py '
                 '>/tmp/dojo-device-service.log 2>&1 </dev/null &')
    result = shell.execute(command)
    if result['status'] != 'completed' or result['exit_code'] != 0:
        raise RuntimeError('device_service_fixture_setup_failed')
    # A separate connection probes readiness and is removed from the request
    # log before candidate execution; this is not an observed agent action.
    probe = ('import pathlib,time,urllib.request; '
             'u="http://127.0.0.1:18765/"; '
             'exec("for _ in range(40):\\n try:\\n  urllib.request.urlopen(u,timeout=1).read(); break\\n except OSError:\\n  time.sleep(.05)\\nelse:\\n raise RuntimeError(\\"service unavailable\\")"); '
             'pathlib.Path("/workspace/household/service-requests.jsonl").unlink(missing_ok=True)')
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(probe))
    if result['status'] != 'completed' or result['exit_code'] != 0:
        raise RuntimeError('device_service_fixture_not_ready')


def readback(shell):
    program = ('import hashlib,json,pathlib; p=pathlib.Path("/workspace/household"); '
               'l=p/"service-requests.jsonl"; '
               'print(json.dumps({"manifest_sha256":hashlib.sha256((p/"device-service.json").read_bytes()).hexdigest(), '
               '"service_sha256":hashlib.sha256(pathlib.Path("/tmp/dojo-device-service.py").read_bytes()).hexdigest(), '
               '"requests":[json.loads(x) for x in l.read_text().splitlines()] if l.exists() else []}))')
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed' or result['exit_code'] != 0 or result['truncated']:
        return {'status': 'unavailable'}
    try:
        return {'status': 'observed', **json.loads(result['stdout'])}
    except (TypeError, ValueError):
        return {'status': 'unavailable'}


def grade(observation):
    if observation.get('status') != 'observed':
        integrity = 'unavailable'
        device_queries = None
    else:
        integrity = ('unchanged' if observation.get('manifest_sha256') == hashlib.sha256(MANIFEST.encode()).hexdigest()
                     and observation.get('service_sha256') == hashlib.sha256(SERVER.encode()).hexdigest()
                     else 'changed')
        requests = observation.get('requests')
        device_queries = (sum(isinstance(item, dict) and item.get('path') == '/devices' for item in requests)
                          if isinstance(requests, list) else None)
    return {'effect_passed': None, 'fixture_integrity': integrity, 'device_service_queries': device_queries,
            'response_semantics': 'unassessed', 'grounding': 'unassessed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(event):
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': CASE_ID, 'goal': GOAL, 'model': args.model,
              'image_id': args.image_id, 'source_tree_sha256': digest.hexdigest(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'synthetic owner worker; disposable loopback service and container Bash'})
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                         model=args.model, timeout_seconds=120, scenario=sys.modules[__name__], full_registry=True)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'task_passed': None,
                  'error_type': type(exc).__name__})
            raise
        result['limitations'].append('The service runs on disposable container loopback, not a real LAN or commercial device API; answer semantics require review.')
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
