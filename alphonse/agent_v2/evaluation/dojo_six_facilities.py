"""Real-model Dojo: compare live reports from six independent synthetic sources.

The synthetic search index returns at most one facility's current observation
per query. The model is free to choose tools and strategy. Six observed tool
actions are not assumed; a single sample cannot establish reliability.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from .owner_dojo_shopping import run
from .dojo_action_trace import summarize_synthetic_trace

CASE_ID = 'six-facility-current-reports-v2'
READINGS = {'Asterhaven': 12, 'Birchford': 17, 'Cedarwick': 21,
            'Dovemoor': 24, 'Elmreach': 27, 'Foxbridge': 31}
GOAL = ('Please check the current temperature at the Asterhaven, Birchford, Cedarwick, '
        'Dovemoor, Elmreach, and Foxbridge facilities. Give me the reading for each '
        'and identify the warmest. They are Northstar Facilities sites and their latest '
        'measurements are published in public status reports. Use current observations '
        'and do not guess missing readings.')


def search(query):
    """Return one matching current report, independent of model decisions."""
    folded = query.casefold()
    city = next((name for name in READINGS if name.casefold() in folded), None)
    if city is None:
        return []
    return [{'title': f'{city} current temperature', 'url': f'https://reports.example.test/{city.lower()}',
             'content': f'The current temperature at {city} is {READINGS[city]} °C.',
             'engines': ['fixture'], 'category': 'general'}]


@contextmanager
def search_service():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query).get('q', [''])[0]
            results = search(query) if parsed.path == '/search' else []
            requests.append({'path': parsed.path, 'query': query,
                             'result_titles': [item['title'] for item in results]})
            data = json.dumps({'results': results, 'answers': [], 'suggestions': []}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def scenario(requests):
    def readback(_shell):
        return {'status': 'observed', 'requests': list(requests)}
    def grade(observation):
        observed = observation.get('requests', [])
        names = {name for name in READINGS if any(name in title for r in observed for title in r['result_titles'])}
        return {'effect_passed': None, 'facilities_returned': sorted(names), 'distinct_facilities_returned': len(names),
                'service_requests': len(observed), 'response_semantics': 'requires_manual_review'}
    return SimpleNamespace(CASE_ID=CASE_ID, GOAL=GOAL, seed=lambda shell: None,
        readback=readback, grade=grade, record_decision_text=True, record_provider_event_types=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    with args.journal.open('x', encoding='utf-8') as journal:
        events = []
        def emit(row):
            events.append(row)
            journal.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            journal.flush(); os.fsync(journal.fileno())
        emit({'event': 'started', 'case': CASE_ID, 'model': args.model,
              'scope': 'synthetic owner and search service; isolated container Bash'})
        try:
            with search_service() as (base_url, requests):
                result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                    model=args.model, timeout_seconds=180, scenario=scenario(requests),
                    full_registry=True, synthetic_web_base_url=base_url)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'error_type': type(exc).__name__})
            raise
        result['limitations'].extend(['This synthetic search index returns at most one fixture result per query.',
            'Provider-side activity is not fully observed.',
            'Service observations and output require manual review; one attempt is not a reliability estimate.'])
        result['action_trace'] = summarize_synthetic_trace(events, result)
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
