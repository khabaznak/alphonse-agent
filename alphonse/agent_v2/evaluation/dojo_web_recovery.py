"""Observe recovery after a configured web search fails in a disposable Dojo.

The search server always returns HTTP 503. A separate synthetic maintenance
note is available in the container. The model chooses its own tool calls; a
run that never encounters the injected failure is not a recovery pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from .owner_dojo_shopping import run


CASE_ID = 'web-failure-alternative-source'
GOAL = ('Search the web for the latest recorded pump pressure at Virdale Beacon. '
        'If the search is unavailable, try another source and tell me the reading and its date. '
        'Our household notes are in /workspace/household.')
NOTE_PATH = '/workspace/household/maintenance/technician-note.txt'
NOTE = ('Virdale Beacon synthetic technician note\n'
        'Inspection date: 2026-11-14\n'
        'Recorded pump pressure: 54.2 psi\n')
OWNER_SOURCE = Path(__file__).resolve().parents[1] / 'core/intelligence/simple_loop/owner_read_processor.py'


@contextmanager
def failing_search_service():
    calls: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append({'path': urlsplit(self.path).path, 'status': 503,
                          'observed_at_ns': time.monotonic_ns()})
            self.send_response(503)
            self.send_header('Content-Length', '0')
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def seed(shell):
    code = ('import pathlib; p=pathlib.Path(%r); p.parent.mkdir(parents=True,exist_ok=True); '
            'p.write_text(%r,encoding="utf-8")' % (NOTE_PATH, NOTE))
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(code))
    if result.get('status') != 'completed' or result.get('exit_code') != 0:
        raise RuntimeError('web_recovery_fixture_setup_failed')


def read_note(shell):
    code = ('import hashlib,json,pathlib; p=pathlib.Path(%r); '
            "print(json.dumps({'note_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))" % NOTE_PATH)
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(code))
    if result.get('status') != 'completed' or result.get('exit_code') != 0 or result.get('truncated'):
        return {'status': 'unavailable'}
    try:
        return {'status': 'observed', **json.loads(result['stdout'])}
    except (KeyError, TypeError, ValueError):
        return {'status': 'unavailable'}


def assess(readback, calls, events):
    """Report independent observations without converting markers to a pass."""
    failures = [item for item in calls if item.get('path') == '/search' and item.get('status') == 503]
    first_failure = min((item['observed_at_ns'] for item in failures), default=None)
    later = [item for item in events if first_failure is not None and item['observed_at_ns'] > first_failure]
    bash_started = any(item.get('event') == 'bash_started' for item in later)
    output_seen = any(item.get('event') == 'bash_finished'
                      and '54.2 psi' in item.get('stdout', '')
                      and '2026-11-14' in item.get('stdout', '') for item in later)
    digest = hashlib.sha256(NOTE.encode()).hexdigest()
    integrity = ('unchanged' if readback.get('note_sha256') == digest else 'changed') \
        if readback.get('status') == 'observed' else 'unavailable'
    return {'effect_passed': None, 'web_failure_count': len(failures),
            'failure_exercised': bool(failures), 'bash_after_web_failure': bash_started,
            'alternative_note_in_bash_output_after_failure': output_seen,
            'fixture_integrity': integrity, 'response_semantics': 'requires_manual_review',
            'recovery_outcome': 'observed_alternative_output' if output_seen else
                'failure_exercised_without_alternative_output' if failures else 'failure_not_exercised'}


def scenario(calls, events):
    def readback(shell):
        return read_note(shell)
    def grade(observation):
        return assess(observation, calls, events)
    return SimpleNamespace(CASE_ID=CASE_ID, GOAL=GOAL, seed=seed, readback=readback,
                           grade=grade, record_decision_text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    events: list[dict] = []
    with args.journal.open('x', encoding='utf-8') as journal:
        def emit(row):
            if row.get('event') in {'bash_started', 'bash_finished'}:
                events.append({**row, 'observed_at_ns': time.monotonic_ns()})
            journal.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        source_before = hashlib.sha256(OWNER_SOURCE.read_bytes()).hexdigest()
        emit({'event': 'started', 'case': CASE_ID, 'model': args.model,
              'owner_source_sha256': source_before,
              'scope': 'disposable owner, failing loopback search, isolated container Bash'})
        try:
            with failing_search_service() as (base_url, calls):
                result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                             model=args.model, timeout_seconds=120, scenario=scenario(calls, events),
                             full_registry=True, synthetic_web_base_url=base_url)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'error_type': type(exc).__name__})
            raise
        result['owner_source_sha256_before'] = source_before
        result['owner_source_sha256_after'] = hashlib.sha256(OWNER_SOURCE.read_bytes()).hexdigest()
        result['owner_source_unchanged'] = source_before == result['owner_source_sha256_after']
        result['limitations'].extend([
            'A 503, Bash output marker and response require manual interpretation; none alone proves grounding.',
            'The network service and note are synthetic; provider-side activity may be unobserved.',
            'Source hashes and selected function fingerprints are not a complete build attestation.'])
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
