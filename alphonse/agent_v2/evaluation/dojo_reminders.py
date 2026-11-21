"""Two synthetic timed requests through the owner worker and Dojo Bash.

The independent reminder poll uses a simulated time, not a running Desktop.
The delayed-work case does not claim execution from creation of a reminder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time
from types import SimpleNamespace

from alphonse.agent_v2.evaluation.owner_dojo_shopping import run
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders


def _pickup_scheduled_result(client, message_id, emit):
    """Use real client IPC; a client ack is not proof that a person read it."""
    if client is None or not message_id:
        return {'status': 'not_attempted'}
    pending = client.request('owner_scheduled_results_list').get('results', [])
    matches = [row for row in pending if row.get('id') == message_id]
    emit({'event': 'scheduled_results_listed', 'message_id': message_id,
          'matches': matches, 'pending_count': len(pending)})
    if len(matches) != 1:
        return {'status': 'not_available' if not matches else 'ambiguous', 'matches': matches}
    acknowledgement = client.request('owner_scheduled_result_ack', id=message_id)
    after = client.request('owner_scheduled_results_list').get('results', [])
    repeated = client.request('owner_scheduled_result_ack', id=message_id)
    result = {'status': acknowledgement.get('status'), 'result': matches[0],
              'present_after_ack': any(row.get('id') == message_id for row in after),
              'repeated_ack_status': repeated.get('status'), 'remaining_count': len(after)}
    emit({'event': 'scheduled_result_ack_attempted', **result})
    return result


def scenario(case):
    pantry_goal = ('In 5 minutes, check the pantry note at /workspace/household/pantry-note.txt '
                   'and tell me how many cans of beans it lists.')
    soap_goal = ('In 6 minutes, check the inventory note at /workspace/household/inventory-note.txt '
                 'and tell me how many bars of soap it lists.')
    cancel_goal = 'Please cancel the pantry check I just scheduled, and keep the soap inventory check.'
    goals = {
        'reminder': 'Remind me to drink water in 5 minutes',
        'delayed-task': pantry_goal,
        'cancel-task': pantry_goal,
        'implicit-timing': 'In 2 minutes, check the clock and tell me what day of the week it is. This is a disposable Dojo test.',
    }
    def seed(shell):
        program = ('import pathlib; p=pathlib.Path("/workspace/household"); '
                   'p.mkdir(parents=True); p.joinpath("pantry-note.txt").write_text("Beans: 4 cans\\n"); '
                   'p.joinpath("inventory-note.txt").write_text("Soap: 7 bars\\n")')
        result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
        if result['status'] != 'completed':
            raise RuntimeError('reminder_fixture_setup_failed')

    def readback(shell):
        result = shell.execute('cat /workspace/household/pantry-note.txt')
        note = result['stdout'] if result['status'] == 'completed' and not result['truncated'] else None
        match = re.fullmatch(r'Beans: ([0-9]+) cans\n', note) if note is not None else None
        soap_result = shell.execute('cat /workspace/household/inventory-note.txt')
        soap = soap_result['stdout'] if soap_result['status'] == 'completed' and not soap_result['truncated'] else None
        soap_match = re.fullmatch(r'Soap: ([0-9]+) bars\n', soap) if soap is not None else None
        return {'pantry_note_unchanged': note == 'Beans: 4 cans\n',
                'observed_pantry_count': int(match.group(1)) if match else None,
                'inventory_note_unchanged': soap == 'Soap: 7 bars\n',
                'observed_soap_count': int(soap_match.group(1)) if soap_match else None}

    def readback_runtime(path, owner_id, started_at, runtime):
        store = OwnerDesktopReminders(str(path))
        records = store.list_open(owner_id)
        schedules = (getattr(runtime, 'owner_schedule_store', None) or runtime.schedule_store).list_tasks(owner_user_id=owner_id)
        result = {'owner_id': owner_id, 'open_reminders': records,
                  'scheduled_tasks': [item.to_dict() for item in schedules],
                  'request_started_at': started_at.isoformat(),
                  'readback_at': datetime.now(timezone.utc).isoformat()}
        if case == 'reminder' and len(records) == 1:
            due = datetime.fromisoformat(records[0]['due_at'])
            before = store.take_due(owner_id, now=due - timedelta(microseconds=1))['reminder']
            at_due = store.take_due(owner_id, now=due)['reminder']
            result['offered_before_due'] = before is not None
            result['offered_at_due'] = at_due is not None
            result['offered_message'] = at_due['message'] if at_due else None
        return result

    def grade(readback_result):
        runtime = readback_result.get('runtime', {})
        rows = runtime.get('open_reminders', [])
        if case == 'cancel-task':
            after = readback_result.get('after_owner') or {}
            triggered = after.get('trigger') == 'simulated_owner_schedule_clock'
            def requested_timing(prompt, window, minutes):
                try:
                    matches = [item for item in runtime.get('scheduled_tasks', []) if item.get('prompt') == prompt]
                    if len(matches) != 1:
                        return 'unassessed'
                    due = datetime.fromisoformat(matches[0]['schedule']['run_at'])
                    submitted = datetime.fromisoformat(window['submitted_at'])
                    acknowledged = datetime.fromisoformat(window['acknowledged_at'])
                    if (any(item.tzinfo is None or item.utcoffset() is None for item in (due, submitted, acknowledged))
                            or acknowledged < submitted):
                        return 'unassessed'
                    return ('within_requested_window' if submitted + timedelta(minutes=minutes, seconds=-30) <= due <=
                            acknowledged + timedelta(minutes=minutes, seconds=30) else 'outside_requested_window')
                except (KeyError, TypeError, ValueError, OverflowError):
                    return 'unassessed'
            target_cancelled = (after.get('target_status_before_due') == 'cancelled' and
                                after.get('target_status_after_due') == 'cancelled') if triggered else None
            target_not_run = after.get('target_occurrence') is None if triggered else None
            other_was_active = after.get('unaffected_status_before_due') == 'active' if triggered else None
            other = after.get('unaffected_occurrence')
            other_completed = other.get('state') == 'completed' if isinstance(other, dict) else False if triggered else None
            owner = runtime.get('owner_id')
            count = readback_result.get('observed_soap_count')
            message_id = after.get('message_id')
            replies = [event['content'] for event in after.get('scheduled_conversation_events', [])
                       if owner and event.get('owner_user_id') == owner and event.get('role') == 'assistant'
                       and event.get('source') == 'scheduled'
                       and event.get('source_message_id') == 'outbound:scheduled:' + str(message_id or '')
                       and isinstance(event.get('content'), str)]
            reply_matches = (any(re.search(r'(?<!\d)' + re.escape(str(count)) + r'(?!\d)', text) for text in replies)
                             if replies and type(count) is int else None)
            delivery = after.get('client_delivery') or {}
            delivered_reply = delivery.get('result') or {}
            return {'effect_passed': None, 'target_cancelled': target_cancelled,
                    'target_no_occurrence_after_due': target_not_run,
                    'unaffected_active_before_due': other_was_active,
                    'unaffected_occurrence_completed': other_completed,
                    'unaffected_reply_contains_observed_count': reply_matches,
                    'target_requested_timing': requested_timing(pantry_goal, readback_result.get('submission_window') or {}, 5),
                    'unaffected_requested_timing': requested_timing(soap_goal, after.get('second_turn') or {}, 6),
                    'response_semantics': 'unassessed',
                    'delivery': delivery.get('status', 'unassessed'),
                    'delivered_reply_matches_recorded': (delivered_reply.get('message') in replies
                        if delivery.get('status') == 'client_acknowledged' else None)}
        if case == 'implicit-timing':
            schedules = runtime.get('scheduled_tasks', [])
            after = readback_result.get('after_owner') or {}
            timing = 'unassessed'
            try:
                if len(schedules) == 1 and schedules[0].get('prompt') == goals[case]:
                    run_at = datetime.fromisoformat(schedules[0]['schedule']['run_at'])
                    window = readback_result['submission_window']
                    submitted = datetime.fromisoformat(window['submitted_at'])
                    acknowledged = datetime.fromisoformat(window['acknowledged_at'])
                    if (any(x.tzinfo is None or x.utcoffset() is None for x in (run_at, submitted, acknowledged))
                            or acknowledged < submitted):
                        raise ValueError
                    timing = ('within_requested_window' if submitted + timedelta(minutes=2, seconds=-30)
                              <= run_at <= acknowledged + timedelta(minutes=2, seconds=30)
                              else 'outside_requested_window')
            except (ValueError, TypeError, KeyError, OverflowError):
                pass
            occurrence = after.get('occurrence')
            return {'effect_passed': None, 'schedule_count': len(schedules),
                    'requested_timing': timing,
                    'scheduled_occurrence': occurrence.get('state') if isinstance(occurrence, dict) else None,
                    'scheduled_conversation_events': after.get('scheduled_conversation_events', []),
                    'client_delivery': after.get('client_delivery', {'status': 'not_attempted'}),
                    'response_semantics': 'unassessed'}
        if case == 'delayed-task':
            after = readback_result.get('after_owner') or {}
            owner = runtime.get('owner_id')
            count = readback_result.get('observed_pantry_count')
            timing = 'unassessed'
            schedule_rows = runtime.get('scheduled_tasks', [])
            try:
                if len(schedule_rows) == 1:
                    run_at = datetime.fromisoformat(schedule_rows[0]['schedule']['run_at'])
                    window = readback_result['submission_window']
                    submitted = datetime.fromisoformat(window['submitted_at'])
                    acknowledged = datetime.fromisoformat(window['acknowledged_at'])
                    if any(value.tzinfo is None or value.utcoffset() is None for value in (run_at, submitted, acknowledged)):
                        raise ValueError
                    earliest = submitted + timedelta(minutes=5, seconds=-30)
                    latest = acknowledged + timedelta(minutes=5, seconds=30)
                    timing = 'within_requested_window' if earliest <= run_at <= latest else 'outside_requested_window'
            except (ValueError, TypeError, KeyError, OverflowError):
                pass
            replies = []
            for execution in after.get('executions', []):
                key = execution.get('occurrence_key')
                if not key:
                    continue
                for message in execution.get('outbound', []):
                    if (message.get('audience_user_id') == owner and owner and
                            (message.get('metadata') or {}).get('occurrence_key') == key and
                            isinstance(message.get('message'), str)):
                        replies.append({'text': message['message'], 'delivery_status': message.get('status')})
            for event in after.get('scheduled_conversation_events', []):
                if (owner and event.get('owner_user_id') == owner and event.get('role') == 'assistant'
                        and event.get('source') == 'scheduled' and
                        event.get('source_message_id') == 'outbound:scheduled:' + str(after.get('message_id') or '')
                        and isinstance(event.get('content'), str)):
                    replies.append({'text': event['content'], 'delivery_status': 'not_established'})
            numeric_match = (any(re.search(r'(?<!\d)' + re.escape(str(count)) + r'(?!\d)', reply['text'])
                                 for reply in replies) if replies and type(count) is int else None)
            trace = after.get('tool_activity_during_handoff') or {}
            calls = trace.get('bash_calls') or []
            note_output = (any(call.get('status') == 'completed' and not call.get('truncated') and
                               f'Beans: {count} cans' in call.get('stdout', '') for call in calls)
                           if type(count) is int else None)
            return {'effect_passed': None, 'scheduled_execution': 'unassessed',
                    'response_semantics': 'unassessed', 'occurrence_bound_replies': replies,
                    'reply_contains_observed_count': numeric_match,
                    'fixture_text_in_bash_output': note_output,
                    'fixture_access': 'unassessed', 'requested_timing': timing,
                    'client_delivery': after.get('client_delivery', {'status': 'not_attempted'})}
        if len(rows) != 1:
            return {'effect_passed': None, 'desktop_reminder_saved': False,
                    'response_semantics': 'unassessed'}
        due = datetime.fromisoformat(rows[0]['due_at'])
        start = datetime.fromisoformat(runtime['request_started_at'])
        end = datetime.fromisoformat(runtime['readback_at'])
        correct = (rows[0]['message'].casefold() == 'drink water' and
                   start + timedelta(minutes=5, seconds=-30) <= due <= end + timedelta(minutes=5, seconds=30) and
                   runtime.get('offered_before_due') is False and runtime.get('offered_at_due') is True and
                   runtime.get('offered_message', '').casefold() == 'drink water')
        return {'effect_passed': correct, 'desktop_reminder_saved': True,
                'response_semantics': 'unassessed'}

    def after_owner(daemon, runtime, owner_id, emit, *, wait_seconds=90, client=None):
        if case not in {'delayed-task', 'cancel-task', 'implicit-timing'}:
            return None
        if case == 'cancel-task':
            if client is None:
                return {'trigger': 'not_attempted', 'reason': 'owner_client_unavailable'}

            def owner_turn(prompt):
                emit({'event': 'owner_followup_started', 'prompt': prompt})
                submitted_at = datetime.now(timezone.utc).isoformat()
                message_id = client.request('experimental_owner_ingress', prompt=prompt)['message_id']
                deadline = time.monotonic() + wait_seconds
                payload = {'status': 'pending'}
                while time.monotonic() < deadline:
                    payload = client.request('experimental_owner_status', message_id=message_id)
                    if payload['status'] not in {'pending', 'processing'}:
                        break
                    time.sleep(.05)
                answer = None
                acknowledged = False
                acknowledged_at = None
                if payload['status'] == 'ready':
                    draft = client.request('experimental_owner_take_draft', message_id=message_id)
                    answer = draft.get('message')
                    client.request('experimental_owner_ack_draft', message_id=message_id,
                                   acknowledgement_token=draft['acknowledgement_token'])
                    acknowledged = client.request('experimental_owner_status', message_id=message_id)['status'] == 'client_acknowledged'
                    if acknowledged:
                        acknowledged_at = datetime.now(timezone.utc).isoformat()
                result = {'message_id': message_id, 'status': payload['status'], 'answer': answer,
                          'client_acknowledged': acknowledged, 'failure_code': payload.get('failure_code'),
                          'submitted_at': submitted_at, 'acknowledged_at': acknowledged_at}
                emit({'event': 'owner_followup_finished', **result})
                return result

            second = owner_turn(soap_goal)
            store = getattr(runtime, 'owner_schedule_store', None) or runtime.schedule_store
            schedules = store.list_tasks(owner_user_id=owner_id)
            target = next((task for task in schedules if task.prompt == pantry_goal), None)
            unaffected = next((task for task in schedules if task.prompt == soap_goal), None)
            if target is None or unaffected is None or len(schedules) != 2 or not target.next_run_at or not unaffected.next_run_at:
                return {'trigger': 'not_attempted', 'reason': 'two_schedules_not_created',
                        'second_turn': second, 'schedule_count': len(schedules)}
            cancellation = owner_turn(cancel_goal)
            after_cancel = store.get_task_for_owner(target.scheduled_task_id, owner_user_id=owner_id)
            other_after_cancel = store.get_task_for_owner(unaffected.scheduled_task_id, owner_user_id=owner_id)
            target_due = datetime.fromisoformat(target.next_run_at)
            other_due = datetime.fromisoformat(unaffected.next_run_at)
            simulated_now = max(target_due, other_due) + timedelta(seconds=1)
            pending_before_due = client.request('owner_scheduled_results_list').get('results', [])
            runtime.queue._owner_schedule_clock = lambda: simulated_now
            emit({'event': 'owner_schedule_clock_advanced', 'simulated_now': simulated_now.isoformat()})

            def occurrence(task_id):
                with runtime.queue._connect() as conn:
                    table = conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_owner_schedule_occurrences'").fetchone()
                    row = (conn.execute('SELECT * FROM v2_owner_schedule_occurrences WHERE schedule_id=? AND owner_id=?',
                                        (task_id, owner_id)).fetchone() if table else None)
                    return dict(row) if row else None

            deadline = time.monotonic() + max(0, wait_seconds)
            while time.monotonic() < deadline:
                other_occurrence = occurrence(unaffected.scheduled_task_id)
                if other_occurrence and other_occurrence['state'] in {'completed', 'failed', 'cancelled'}:
                    break
                time.sleep(.1)
            target_occurrence = occurrence(target.scheduled_task_id)
            other_occurrence = occurrence(unaffected.scheduled_task_id)
            target_after = store.get_task_for_owner(target.scheduled_task_id, owner_user_id=owner_id)
            other_after = store.get_task_for_owner(unaffected.scheduled_task_id, owner_user_id=owner_id)
            message_id = other_occurrence['message_id'] if other_occurrence else None
            events = [event.to_dict() for event in runtime.conversation_store.list_all(owner_user_id=owner_id)
                      if message_id and event.source == 'scheduled' and event.role == 'assistant'
                      and event.source_message_id == 'outbound:scheduled:' + message_id]
            delivery = _pickup_scheduled_result(client, message_id, emit)
            return {'trigger': 'simulated_owner_schedule_clock', 'second_turn': second,
                    'cancel_turn': cancellation, 'target_status_before_due': getattr(after_cancel, 'status', None),
                    'unaffected_status_before_due': getattr(other_after_cancel, 'status', None),
                    'target_status_after_due': getattr(target_after, 'status', None),
                    'unaffected_status_after_due': getattr(other_after, 'status', None),
                    'target_occurrence': target_occurrence, 'unaffected_occurrence': other_occurrence,
                    'message_id': message_id, 'scheduled_conversation_events': events,
                    'pending_delivery_before_due': pending_before_due, 'client_delivery': delivery}
        schedules = (getattr(runtime, 'owner_schedule_store', None) or runtime.schedule_store).list_tasks(owner_user_id=owner_id)
        if len(schedules) != 1 or not schedules[0].next_run_at:
            return {'trigger': 'not_attempted', 'reason': 'no_single_due_schedule'}
        task = schedules[0]
        due = datetime.fromisoformat(task.next_run_at)
        # Only the due-time clock is simulated. The actual owner worker stages,
        # claims and runs the occurrence; claim leases still use wall time.
        # Never insert a trusted owner message or manufacture an occurrence.
        queue = runtime.queue
        queue._owner_schedule_clock = lambda: due - timedelta(microseconds=1)
        def lookup():
            with queue._connect() as conn:
                table = conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_owner_schedule_occurrences'").fetchone()
                row = (conn.execute('SELECT * FROM v2_owner_schedule_occurrences WHERE schedule_id=? AND owner_id=?',
                                    (task.scheduled_task_id, owner_id)).fetchone() if table else None)
                return dict(row) if row else None
        time.sleep(max(.2, daemon.poll_interval_sec * 3))
        before = lookup()
        if before is not None:
            return {'trigger': 'not_attempted', 'reason': 'occurrence_already_exists', 'before_due': before}
        before_delivery = client.request('owner_scheduled_results_list').get('results', []) if client else None
        emit({'event': 'owner_schedule_clock_advanced', 'simulated_now': due.isoformat()})
        queue._owner_schedule_clock = lambda: due
        deadline = time.monotonic() + max(0, wait_seconds)
        occurrence = None
        while True:
            occurrence = lookup()
            if occurrence and occurrence['state'] in {'completed', 'failed', 'cancelled'}:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
        message_id = occurrence['message_id'] if occurrence else None
        events = [event.to_dict() for event in runtime.conversation_store.list_all(owner_user_id=owner_id)
                  if message_id and event.source == 'scheduled' and event.role == 'assistant'
                  and event.source_message_id == 'outbound:scheduled:' + message_id]
        delivery = _pickup_scheduled_result(client, message_id, emit)
        return {'trigger': 'simulated_owner_schedule_clock', 'before_due': before,
                'occurrence': occurrence, 'message_id': message_id,
                'queue_status': queue.status_for(message_id) if message_id else None,
                'scheduled_conversation_events': events, 'pending_delivery_before_due': before_delivery,
                'client_delivery': delivery}

    return SimpleNamespace(CASE_ID=case, GOAL=goals[case], seed=seed, readback=readback,
                           readback_runtime=readback_runtime, grade=grade, after_owner=after_owner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['reminder', 'delayed-task', 'cancel-task', 'implicit-timing'])
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    case = scenario(args.case)
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(row):
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': args.case, 'goal': case.GOAL,
              'model': args.model, 'inference': 'openai_codex', 'image_id': args.image_id,
              'source_tree_sha256': digest.hexdigest(), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'disposable owner daemon and Bash container; synthetic data'})
        fd = os.open(args.journal.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                         model=args.model, timeout_seconds=120, scenario=case, full_registry=True)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'task_passed': None,
                  'error_type': type(exc).__name__})
            raise
        result['limitations'].extend([
            'Reminder due-time poll uses an injected time after the worker stops; Desktop delivery is untested.',
            'Delayed execution, future follow-through, and delivery are not evaluated by the reminder-store readback.',
            'The owner worker uses a simulated due-time clock in this run; claim leases use the real clock.',
            'An occurrence or conversation event does not prove that the requested work happened or reached a person.',
        ])
        emit({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
