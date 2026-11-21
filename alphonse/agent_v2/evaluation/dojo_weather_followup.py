"""Multi-turn weather follow-up using an independently seeded synthetic report.

This exercises the owner daemon, conversation IPC, and disposable container
Bash. The model receives the source location, not a prescribed tool or answer.
The weather observations are invented and are never live weather claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

from .owner_dojo_shopping import run

REPORT_PATH = '/workspace/household/weather-observations.json'
REPORT = json.dumps({'source': 'Synthetic weather observations', 'units': 'celsius',
                     'temperatures': {'Merida': 23, 'Campeche': 31, 'Oaxaca': 18}},
                    sort_keys=True, separators=(',', ':')) + '\n'
CASES = {
    'en': {
        'goal': f'What temperature does the weather report at {REPORT_PATH} give for Merida?',
        'followups': ['What about Campeche?', 'Sorry, I meant Oaxaca.'],
    },
    'es': {
        'goal': f'¿Qué temperatura indica el informe meteorológico {REPORT_PATH} para Merida?',
        'followups': ['¿Y en Campeche?', 'Perdón, quería decir Oaxaca.'],
    },
}
EXPECTED = [('Merida', 23), ('Campeche', 31), ('Oaxaca', 18)]


def seed(shell):
    code = ("import pathlib; p=pathlib.Path('/workspace/household'); "
            "p.mkdir(parents=True,exist_ok=True); "
            "(p/'weather-observations.json').write_text(%r,encoding='utf-8')" % REPORT)
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(code))
    if result.get('status') != 'completed' or result.get('exit_code') != 0:
        raise RuntimeError('weather_fixture_setup_failed')


def readback(shell):
    code = ('import hashlib,json,pathlib; p=pathlib.Path(%r); '
            "print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))" % REPORT_PATH)
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(code))
    if result.get('status') != 'completed' or result.get('exit_code') != 0 or result.get('truncated'):
        return {'status': 'unavailable'}
    try:
        return {'status': 'observed', **json.loads(result['stdout'])}
    except (KeyError, TypeError, ValueError):
        return {'status': 'unavailable'}


def _run_followup(client, prompt, *, timeout):
    started = client.request('experimental_owner_ingress', prompt=prompt)
    message_id = started['message_id']
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.request('experimental_owner_status', message_id=message_id)
        if status.get('status') not in {'pending', 'processing'}:
            break
        time.sleep(.05)
    else:
        return {'status': 'timeout'}
    result = {'status': status.get('status'), 'failure_code': status.get('failure_code')}
    if result['status'] == 'ready':
        draft = client.request('experimental_owner_take_draft', message_id=message_id)
        result['answer'] = draft.get('message')
        result['kind'] = draft.get('kind')
        acknowledgement = client.request('experimental_owner_ack_draft', message_id=message_id,
                                          acknowledgement_token=draft['acknowledgement_token'])
        result['acknowledgement'] = acknowledgement.get('status')
    return result


def score_mentions(turns):
    """Record literal observations; matches are not a semantic pass/fail."""
    rows = []
    for index, (city, degrees) in enumerate(EXPECTED):
        turn = turns[index] if index < len(turns) else {}
        turn = turn if isinstance(turn, dict) else {}
        text = turn.get('answer')
        text = text if isinstance(text, str) else ''
        folded = ''.join(c for c in unicodedata.normalize('NFKD', text.casefold())
                         if not unicodedata.combining(c))
        rows.append({'status': turn.get('status', 'not_attempted'),
                     'expected_city_mentioned': city.casefold() in folded,
                     'expected_temperature_mentioned': bool(re.search(
                         r'(?<!\d)(?<!\d[.,])' + str(degrees) + r'(?!\d|[.,]\d)', text))})
    return rows


def scenario(language):
    if language not in CASES:
        raise ValueError('weather_scenario_invalid')
    case = CASES[language]

    def after_owner(daemon, runtime, owner_id, emit, *, client):
        del daemon
        history = runtime.conversation_store.list(owner_user_id=owner_id)
        if not history or history[-1].role != 'assistant':
            return {'turns': [], 'followup_outcome': 'first_turn_unavailable'}
        turns = [{'status': 'ready', 'answer': history[-1].content}]
        for prompt in case['followups']:
            turn = _run_followup(client, prompt, timeout=120)
            turns.append(turn)
            emit({'event': 'weather_followup_observed', 'prompt': prompt, **turn})
            if turn['status'] != 'ready' or turn.get('acknowledgement') != 'client_acknowledged':
                break
        return {'turns': turns, 'followup_outcome': 'observed' if len(turns) == 3 else 'incomplete'}

    def grade(readback_result):
        after = readback_result.get('after_owner') or {}
        turns = after.get('turns') if isinstance(after.get('turns'), list) else []
        if readback_result.get('status') != 'observed':
            integrity = 'unavailable'
        else:
            integrity = ('unchanged' if readback_result.get('sha256') == hashlib.sha256(REPORT.encode()).hexdigest()
                         else 'changed')
        return {'effect_passed': None, 'fixture_integrity': integrity,
                'followup_outcome': after.get('followup_outcome', 'unassessed'),
                'turn_checks': score_mentions(turns), 'response_semantics': 'requires_manual_review'}

    return SimpleNamespace(CASE_ID='weather-followup-' + language, GOAL=case['goal'], seed=seed,
                           readback=readback, after_owner=after_owner, grade=grade,
                           record_decision_text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=tuple(CASES))
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    case = scenario(args.language)
    with args.journal.open('x', encoding='utf-8') as journal:
        def emit(row):
            journal.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        emit({'event': 'started', 'case': case.CASE_ID, 'model': args.model,
              'scope': 'disposable synthetic owner, weather file and container Bash'})
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                         model=args.model, timeout_seconds=120, scenario=case, full_registry=True)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'error_type': type(exc).__name__})
            raise
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
