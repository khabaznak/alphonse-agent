"""Observe whether a configured general web search reaches an owner conversation.

The search service is a disposable loopback SearXNG-shaped fixture; its weather
result and place are invented. It is not a real weather or external web test.
The provider, registry, model-advertised tools and service calls are reported
separately; a model's answer does not prove a successful search.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from .owner_dojo_shopping import run

CITY = 'Virdale Beacon'
GOAL = f'What is the current temperature in {CITY}?'
_OWNER_SOURCE = Path(__file__).resolve().parents[1] / 'core/intelligence/simple_loop/owner_read_processor.py'


def _owner_source_hash():
    return hashlib.sha256(_OWNER_SOURCE.read_bytes()).hexdigest()


@contextmanager
def synthetic_search_service():
    observations: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query).get('q', [''])[0]
            matched = parsed.path == '/search' and 'virdale' in query.casefold()
            observations.append({'path': parsed.path, 'query_matches_city': matched})
            items = [{'title': f'{CITY} weather observation',
                      'url': 'https://weather.example.test/virdale-beacon',
                      'content': f'Current synthetic temperature in {CITY}: 28 °C.',
                      'engines': ['fixture'], 'category': 'general'}] if matched else []
            data = json.dumps({'results': items, 'answers': [], 'suggestions': []}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', observations
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def scenario(observations):
    def readback(_shell):
        return {'status': 'observed', 'search_requests': list(observations)}

    def grade(observation):
        requests = observation.get('search_requests') if isinstance(observation, dict) else None
        return {'effect_passed': None, 'web_search_requests': len(requests) if isinstance(requests, list) else None,
                'matching_search_requests': sum(r.get('query_matches_city') is True for r in requests)
                if isinstance(requests, list) else None, 'response_semantics': 'requires_manual_review'}

    return SimpleNamespace(CASE_ID='configured-web-weather-discovery', GOAL=GOAL,
                           seed=lambda _shell: None, readback=readback, grade=grade,
                           record_decision_text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    with args.journal.open('x', encoding='utf-8') as journal:
        def emit(row):
            journal.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        source_before = _owner_source_hash()
        emit({'event': 'started', 'case': 'configured-web-weather-discovery', 'model': args.model,
              'owner_source_sha256': source_before,
              'scope': 'disposable owner, configured loopback search, isolated container Bash'})
        try:
            with synthetic_search_service() as (base_url, observations):
                result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                             model=args.model, timeout_seconds=120, scenario=scenario(observations),
                             full_registry=True, synthetic_web_base_url=base_url)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'error_type': type(exc).__name__})
            raise
        result['limitations'].append('The search backend is synthetic; this does not establish real web availability or live weather accuracy.')
        result['owner_source_sha256_before'] = source_before
        result['owner_source_sha256_after'] = _owner_source_hash()
        result['owner_source_unchanged'] = source_before == result['owner_source_sha256_after']
        result['limitations'].append('Source hashes and selected function fingerprints are not a complete build attestation.')
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
