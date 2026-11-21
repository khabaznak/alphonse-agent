"""Real owner-worker clarification and reminder follow-up with disposable data.

The scorer checks the recorded effect and simulated Desktop offer. It does not
use model claims as proof, and explicitly reports when the first question or
second turn could not be completed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from .owner_dojo_shopping import run
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders


def scenario():
    goal = 'Please remind me to call the dentist.'
    reply = 'In 5 minutes.'

    def after_owner(daemon, runtime, owner_id, emit, *, client):
        store = OwnerDesktopReminders(runtime.queue.db_path)
        before = store.list_open(owner_id)
        questions = client.request('experimental_owner_questions').get('questions', [])
        observation = {'before_reminders': before, 'questions': questions,
                       'reply': reply, 'reply_status': 'not_attempted'}
        if len(questions) != 1:
            return observation
        submitted_at = datetime.now(timezone.utc)
        try:
            second_id = client.request('experimental_owner_ingress', prompt=reply,
                reply_to_message_id=questions[0]['message_id'])['message_id']
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                payload = client.request('experimental_owner_status', message_id=second_id)
                if payload['status'] not in {'pending', 'processing'}:
                    break
                time.sleep(.02)
            observation['reply_status'] = payload['status']
            observation['failure_code'] = payload.get('failure_code')
            if payload['status'] == 'ready':
                draft = client.request('experimental_owner_take_draft', message_id=second_id)
                observation['reply_draft'] = {'kind': draft.get('kind'), 'message': draft.get('message')}
                observation['reply_acknowledgement'] = client.request('experimental_owner_ack_draft',
                    message_id=second_id, acknowledgement_token=draft['acknowledgement_token']).get('status')
        except (ValueError, RuntimeError, OSError) as exc:
            observation['reply_status'] = 'error'
            observation['error_type'] = type(exc).__name__
        completed_at = datetime.now(timezone.utc)
        rows = store.list_open(owner_id)
        observation['after_reminders'] = rows
        observation['submitted_at'] = submitted_at.isoformat()
        observation['completed_at'] = completed_at.isoformat()
        if len(rows) == 1:
            due = datetime.fromisoformat(rows[0]['due_at'])
            observation['offered_before_due'] = store.take_due(owner_id, now=due - timedelta(microseconds=1))['reminder'] is not None
            offered = store.take_due(owner_id, now=due)['reminder']
            observation['offered_at_due'] = offered is not None
            observation['offered_message'] = offered['message'] if offered else None
        emit({'event': 'followup_observed', **observation})
        return observation

    def grade(readback):
        data = readback.get('after_owner') or {}
        rows = data.get('after_reminders')
        if data.get('reply_status') == 'not_attempted' or not isinstance(rows, list):
            return {'effect_passed': None, 'followup_outcome': 'unassessed', 'response_semantics': 'unassessed'}
        if data.get('failure_code') in {'model_unavailable', 'usage_unconfirmed', 'usage_limit_exceeded'}:
            return {'effect_passed': None, 'followup_outcome': 'provider_blocked', 'response_semantics': 'unassessed'}
        try:
            due = datetime.fromisoformat(rows[0]['due_at']) if len(rows) == 1 else None
            earliest = datetime.fromisoformat(data['submitted_at']) + timedelta(minutes=5, seconds=-30)
            latest = datetime.fromisoformat(data['completed_at']) + timedelta(minutes=5, seconds=30)
            within = due is not None and earliest <= due <= latest
        except (KeyError, TypeError, ValueError):
            within = False
        correct = (data.get('before_reminders') == [] and len(rows) == 1
                   and rows[0].get('message', '').strip().rstrip('.!?').casefold() == 'call the dentist'
                   and within and data.get('offered_before_due') is False
                   and data.get('offered_at_due') is True
                   and (data.get('offered_message') or '').strip().rstrip('.!?').casefold() == 'call the dentist')
        return {'effect_passed': correct, 'followup_outcome': 'observed',
                'correct_timing': within, 'response_semantics': 'unassessed'}

    return SimpleNamespace(CASE_ID='reminder-followup', GOAL=goal, record_decision_text=True,
        seed=lambda _shell: None, readback=lambda _shell: {'status': 'observed'},
        after_owner=after_owner, grade=grade)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--synthetic-clarification-review', action='store_true')
    args = parser.parse_args()
    case = scenario()
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(event):
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': case.CASE_ID, 'goal': case.GOAL,
              'model': args.model, 'synthetic_clarification_review': args.synthetic_clarification_review,
              'scope': 'synthetic owner and disposable container; simulated reminder offer'})
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket,
                emit=emit, model=args.model, timeout_seconds=120, scenario=case,
                full_registry=True, synthetic_clarification_review=args.synthetic_clarification_review)
        except Exception as exc:
            emit({'event': 'infrastructure_error', 'type': type(exc).__name__})
            raise
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
