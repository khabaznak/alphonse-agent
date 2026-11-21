"""Observe a three-turn conversation with a corrected household appointment.

The fixture has no real appointment and asks only about conversation context;
it does not schedule, update, or cancel anything. Output semantics require
review even when literal markers are present.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

from .dojo_weather_followup import _run_followup
from .owner_dojo_shopping import run

GOAL = ("The plumber said the visit is Tuesday at 3 p.m. "
        "Can you repeat the time they gave me?")
FOLLOWUPS = (
    "Correction: the plumber changed it to Thursday at 10 a.m. What is the new plan?",
    "And when did I originally say it was?",
)


def mention_checks(turns):
    expected = (('tuesday', r'(?<!\d)(?:3|15)(?!\d)'),
                ('thursday', r'(?<!\d)10(?!\d)'),
                ('tuesday', r'(?<!\d)(?:3|15)(?!\d)'))
    result = []
    for index, (weekday, pattern) in enumerate(expected):
        item = turns[index] if index < len(turns) and isinstance(turns[index], dict) else {}
        text = item.get('answer') if isinstance(item.get('answer'), str) else ''
        result.append({'status': item.get('status', 'not_attempted'),
                       'expected_weekday_mentioned': weekday in text.casefold(),
                       'expected_hour_mentioned': bool(re.search(pattern, text))})
    return result


def scenario():
    def after_owner(daemon, runtime, owner_id, emit, *, client):
        del daemon
        history = runtime.conversation_store.list(owner_user_id=owner_id)
        if not history or history[-1].role != 'assistant':
            return {'turns': [], 'followup_outcome': 'first_turn_unavailable'}
        turns = [{'status': 'ready', 'answer': history[-1].content}]
        for prompt in FOLLOWUPS:
            turn = _run_followup(client, prompt, timeout=120)
            turns.append(turn)
            emit({'event': 'plan_followup_observed', 'prompt': prompt, **turn})
            if turn['status'] != 'ready' or turn.get('acknowledgement') != 'client_acknowledged':
                break
        return {'turns': turns, 'followup_outcome': 'observed' if len(turns) == 3 else 'incomplete'}

    def grade(readback):
        after = readback.get('after_owner') or {}
        turns = after.get('turns') if isinstance(after.get('turns'), list) else []
        return {'effect_passed': None, 'followup_outcome': after.get('followup_outcome', 'unassessed'),
                'turn_checks': mention_checks(turns), 'response_semantics': 'requires_manual_review'}

    return SimpleNamespace(CASE_ID='appointment-plan-followup', GOAL=GOAL, seed=lambda _shell: None,
                           readback=lambda _shell: {'status': 'observed'}, after_owner=after_owner,
                           grade=grade, record_decision_text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    case = scenario()
    with args.journal.open('x', encoding='utf-8') as journal:
        def emit(row):
            journal.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        emit({'event': 'started', 'case': case.CASE_ID, 'model': args.model,
              'scope': 'disposable synthetic owner, no real appointments'})
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
