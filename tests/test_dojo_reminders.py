from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from alphonse.agent_v2.evaluation.dojo_reminders import scenario, _pickup_scheduled_result
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.users import V2UserStore


def test_reminder_observer_checks_time_and_due_poll(tmp_path):
    path = tmp_path / 'dojo.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    now = datetime.now(timezone.utc)
    store = OwnerDesktopReminders(str(path))
    store.create(owner.user_id, 'drink water', (now + timedelta(minutes=5)).isoformat(), 'a' * 32, now=now)
    case = scenario('reminder')
    runtime = SimpleNamespace(schedule_store=ScheduledTaskStore())
    observed = case.readback_runtime(path, owner.user_id, now, runtime)
    grade = case.grade({'runtime': observed})
    assert grade['effect_passed'] is True
    assert observed['offered_before_due'] is False
    assert observed['offered_at_due'] is True
    observed['open_reminders'][0]['due_at'] = (now + timedelta(hours=1)).isoformat()
    assert case.grade({'runtime': observed})['effect_passed'] is False


def test_delayed_task_never_confuses_a_reminder_with_execution():
    result = scenario('delayed-task').grade({'runtime': {'open_reminders': [{'message': 'check pantry'}]}})
    assert result['effect_passed'] is None
    assert result['scheduled_execution'] == 'unassessed'


def test_implicit_timing_reports_missing_and_wrong_schedule_separately():
    case = scenario('implicit-timing')
    now = datetime.now(timezone.utc)
    window = {'submitted_at': now.isoformat(), 'acknowledged_at': (now + timedelta(seconds=10)).isoformat()}
    observed = {'runtime': {'scheduled_tasks': []}, 'submission_window': window}
    assert case.grade(observed)['schedule_count'] == 0
    assert case.grade(observed)['requested_timing'] == 'unassessed'
    observed['runtime']['scheduled_tasks'] = [{'prompt': case.GOAL,
        'schedule': {'run_at': (now + timedelta(hours=1)).isoformat()}}]
    assert case.grade(observed)['requested_timing'] == 'outside_requested_window'
    observed['runtime']['scheduled_tasks'][0]['schedule']['run_at'] = (now + timedelta(minutes=2)).isoformat()
    assert case.grade(observed)['requested_timing'] == 'within_requested_window'


def test_cancellation_grade_requires_target_and_unaffected_evidence():
    observed = {'runtime': {'owner_id': 'owner'}, 'observed_soap_count': 7,
                'after_owner': {'trigger': 'simulated_owner_schedule_clock',
                    'target_status_before_due': 'cancelled', 'target_status_after_due': 'cancelled',
                    'unaffected_status_before_due': 'active', 'target_occurrence': None,
                    'unaffected_occurrence': {'state': 'completed'}, 'message_id': 'other-work',
                    'scheduled_conversation_events': [
                        {'owner_user_id': 'owner', 'role': 'assistant', 'source': 'scheduled',
                         'source_message_id': 'outbound:scheduled:other-work', 'content': 'There are 7 bars.'}]}}
    result = scenario('cancel-task').grade(observed)
    assert result['target_cancelled'] is True
    assert result['target_no_occurrence_after_due'] is True
    assert result['unaffected_active_before_due'] is True
    assert result['unaffected_occurrence_completed'] is True
    assert result['unaffected_reply_contains_observed_count'] is True
    assert result['delivery'] == 'unassessed'
    observed['after_owner']['target_occurrence'] = {'state': 'completed'}
    assert scenario('cancel-task').grade(observed)['target_no_occurrence_after_due'] is False
    observed['after_owner']['trigger'] = 'not_attempted'
    assert scenario('cancel-task').grade(observed)['target_cancelled'] is None


def test_cancellation_grades_each_requested_timing_independently():
    now = datetime.now(timezone.utc)
    case = scenario('cancel-task')
    pantry = case.GOAL
    soap = ('In 6 minutes, check the inventory note at /workspace/household/inventory-note.txt '
            'and tell me how many bars of soap it lists.')
    window = {'submitted_at': now.isoformat(), 'acknowledged_at': (now + timedelta(seconds=10)).isoformat()}
    observed = {'runtime': {'scheduled_tasks': [
        {'prompt': pantry, 'schedule': {'run_at': (now + timedelta(minutes=5)).isoformat()}},
        {'prompt': soap, 'schedule': {'run_at': (now + timedelta(minutes=6)).isoformat()}}]},
        'submission_window': window, 'after_owner': {'second_turn': dict(window)}}
    grade = case.grade(observed)
    assert grade['target_requested_timing'] == grade['unaffected_requested_timing'] == 'within_requested_window'
    observed['runtime']['scheduled_tasks'][1]['schedule']['run_at'] = (now + timedelta(hours=6)).isoformat()
    assert case.grade(observed)['unaffected_requested_timing'] == 'outside_requested_window'
    observed['after_owner']['second_turn']['acknowledged_at'] = None
    assert case.grade(observed)['unaffected_requested_timing'] == 'unassessed'


def test_scheduled_delivery_lists_and_acknowledges_only_the_matching_message():
    class Client:
        pending = [{'id': 'wanted', 'message': '7 bars'}, {'id': 'other', 'message': 'unrelated'}]
        calls = []
        def request(self, method, **kwargs):
            self.calls.append((method, kwargs))
            if method == 'owner_scheduled_results_list':
                return {'results': list(self.pending)}
            assert method == 'owner_scheduled_result_ack' and kwargs == {'id': 'wanted'}
            self.pending = [row for row in self.pending if row['id'] != 'wanted']
            return {'status': 'client_acknowledged'}
    client = Client()
    result = _pickup_scheduled_result(client, 'wanted', lambda _: None)
    assert result['result']['message'] == '7 bars'
    assert result['status'] == result['repeated_ack_status'] == 'client_acknowledged'
    assert result['present_after_ack'] is False and result['remaining_count'] == 1
    assert [row['id'] for row in client.pending] == ['other']


def test_delayed_observer_correlates_reply_and_fixture_without_inferring_delivery():
    observed = {
        'observed_pantry_count': 4,
        'runtime': {'owner_id': 'owner'},
        'after_owner': {
            'executions': [{'occurrence_key': 'scheduled:1', 'outbound': [
                {'audience_user_id': 'owner', 'message': 'There are 4 cans of beans.',
                 'status': 'pending', 'metadata': {'occurrence_key': 'scheduled:1'}},
                {'audience_user_id': 'other', 'message': '4', 'status': 'delivered',
                 'metadata': {'occurrence_key': 'scheduled:1'}}]}],
            'tool_activity_during_handoff': {'bash_calls': [
                {'status': 'completed', 'truncated': False, 'stdout': 'Beans: 4 cans\n'}]}}}
    result = scenario('delayed-task').grade(observed)
    assert result['reply_contains_observed_count'] is True
    assert result['fixture_text_in_bash_output'] is True
    assert len(result['occurrence_bound_replies']) == 1
    assert result['occurrence_bound_replies'][0]['delivery_status'] == 'pending'
    assert result['effect_passed'] is None
    assert result['fixture_access'] == result['response_semantics'] == 'unassessed'
    observed['after_owner']['message_id'] = 'occurrence-message'
    observed['after_owner']['scheduled_conversation_events'] = [
        {'owner_user_id': 'owner', 'role': 'assistant', 'source': 'scheduled',
         'source_message_id': 'outbound:scheduled:occurrence-message', 'content': '4 cans.'},
        {'owner_user_id': 'owner', 'role': 'assistant', 'source': 'scheduled',
         'source_message_id': 'outbound:scheduled:unrelated', 'content': '4'}]
    result = scenario('delayed-task').grade(observed)
    assert len(result['occurrence_bound_replies']) == 2
    assert result['occurrence_bound_replies'][1]['delivery_status'] == 'not_established'
    observed['observed_pantry_count'] = 14
    result = scenario('delayed-task').grade(observed)
    assert result['reply_contains_observed_count'] is False
    assert result['fixture_text_in_bash_output'] is False


def test_delayed_observer_checks_five_minutes_against_submission_window():
    now = datetime.now(timezone.utc)
    observed = {'runtime': {'scheduled_tasks': [{'schedule': {'run_at': (now + timedelta(minutes=5)).isoformat()}}]},
                'submission_window': {'submitted_at': now.isoformat(),
                                      'acknowledged_at': (now + timedelta(seconds=12)).isoformat()}}
    case = scenario('delayed-task')
    assert case.grade(observed)['requested_timing'] == 'within_requested_window'
    observed['runtime']['scheduled_tasks'][0]['schedule']['run_at'] = (now + timedelta(hours=5)).isoformat()
    assert case.grade(observed)['requested_timing'] == 'outside_requested_window'
    observed['submission_window']['acknowledged_at'] = None
    assert case.grade(observed)['requested_timing'] == 'unassessed'


def test_due_observer_does_not_manufacture_execution_without_worker(tmp_path):
    now = datetime.now(timezone.utc)
    schedules = ScheduledTaskStore()
    task = schedules.create_task(owner_user_id='synthetic-owner', name='check pantry', prompt='check pantry',
        schedule_kind='once', run_at=(now + timedelta(minutes=5)).isoformat(), now=now)
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite', experimental_claim_fencing=True)
    runtime = SimpleNamespace(schedule_store=schedules, queue=queue,
                              conversation_store=SimpleNamespace(list_all=lambda **_: []))
    observed = scenario('delayed-task').after_owner(SimpleNamespace(poll_interval_sec=.01), runtime,
                                                    'synthetic-owner', lambda _: None, wait_seconds=0)
    assert observed['before_due'] is None
    assert observed['trigger'] == 'simulated_owner_schedule_clock'
    assert observed['occurrence'] is None
    assert observed['scheduled_conversation_events'] == []
    assert queue._owner_schedule_clock() == datetime.fromisoformat(task.next_run_at)
    with queue._connect() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_trusted_owner_ingress'").fetchone() is None
