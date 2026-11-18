"""Stateful synthetic obstacle contracts, not real scheduler qualification."""
from pathlib import Path

from alphonse.agent_v2.evaluation.owner_obstacle_gym import FixtureInterrupted
from alphonse.agent_v2.evaluation.owner_temporal_obstacle_gym import TemporalFixtureWorld, _local_time, run_temporal_corpus

CORPUS = Path(__file__).parent / 'fixtures/owner_obstacles/temporal_lifecycle_v1.json'


class ScriptedLifecycleAdapter:
    """Memorized fixture actions validate the oracle, not model understanding."""
    offline = True
    def run(self, scenario, world):
        assert set(scenario).isdisjoint({'oracle', 'expected_effects', 'expected_state'})
        actor, goal = scenario['actor'], scenario['goal']
        if goal.startswith('Move reminder'):
            args = {'request_id': 'change-1', 'reminder_id': 'r1', 'expected_version': 1, 'due_at': '2026-03-01T11:00:00+00:00'}
            assert world.act(actor, 'change_reminder', args)['status'] == 'saved'
        elif goal == 'Cancel reminder r1.':
            args = {'request_id': 'cancel-1', 'reminder_id': 'r1', 'expected_version': 1}
            assert world.act(actor, 'cancel_reminder', args)['status'] == 'saved'
        elif goal.startswith('Change stale reminder'):
            assert world.act(actor, 'change_reminder', {'request_id': 'stale', 'reminder_id': 'r1',
                'expected_version': 2, 'due_at': '2026-03-01T11:00:00+00:00'})['status'] == 'stale'
            return {'status': 'blocked'}
        elif 'wall time' in goal:
            local = '2026-03-08T02:30:00' if 'nonexistent' in goal else '2026-11-01T01:30:00'
            assert world.act(actor, 'change_schedule', {'request_id': 'ask', 'schedule_id': 's1',
                'expected_version': 2, 'local_time': local, 'timezone': 'America/New_York', 'fold': None})['status'] == 'clarification'
            return {'status': 'clarification'}
        elif goal.startswith('Move recurring schedule'):
            local = {'local_time': '2026-11-01T01:30:00', 'timezone': 'America/New_York'}
            resolution = world.act(actor, 'resolve_local_time', local)
            assert resolution == {'status': 'ambiguous', 'choices_utc': {0: '2026-11-01T05:30:00+00:00', 1: '2026-11-01T06:30:00+00:00'}}
            assert world.act(actor, 'change_schedule', {'request_id': 'fold-1', 'schedule_id': 's1', 'expected_version': 2, **local, 'fold': 1})['status'] == 'saved'
        elif goal.startswith('Process duplicate delivery of event'):
            event = {'subscription_id': 'sub1', 'event_id': 'evt1', 'source': 'door', 'event': 'open'}
            assert world.act(actor, 'receive_event', event)['status'] == 'saved'
            assert world.act(actor, 'receive_event', event)['status'] == 'duplicate'
        elif goal.startswith('Cancel subscription'):
            assert world.act(actor, 'cancel_subscription', {'request_id': 'cancel-sub', 'subscription_id': 'sub1', 'expected_version': 1})['status'] == 'saved'
            assert world.act(actor, 'receive_event', {'subscription_id': 'sub1', 'event_id': 'evt1', 'source': 'door', 'event': 'open'})['status'] == 'suppressed'
        elif goal.startswith('Cancel weekly schedule'):
            assert world.act(actor, 'cancel_schedule', {'request_id': 'cancel-schedule', 'schedule_id': 's1', 'expected_version': 2})['status'] == 'saved'
            assert world.act(actor, 'fire_occurrence', {'schedule_id': 's1', 'occurrence_id': 'occ1'})['status'] == 'suppressed'
        elif goal.startswith('Process duplicate delivery of occurrence'):
            occurrence = {'schedule_id': 's1', 'occurrence_id': 'occ1'}
            assert world.act(actor, 'fire_occurrence', occurrence)['status'] == 'saved'
            assert world.act(actor, 'fire_occurrence', occurrence)['status'] == 'duplicate'
        elif goal.startswith('Change subscription'):
            assert world.act(actor, 'change_subscription', {'request_id': 'change-sub', 'subscription_id': 'sub1',
                'expected_version': 1, 'source': 'door', 'event': 'close', 'action': 'check lights'})['status'] == 'saved'
            assert world.act(actor, 'receive_event', {'subscription_id': 'sub1', 'event_id': 'evt1', 'source': 'door', 'event': 'open'})['status'] == 'unmatched'
            assert world.act(actor, 'receive_event', {'subscription_id': 'sub1', 'event_id': 'evt2', 'source': 'door', 'event': 'close'})['status'] == 'saved'
        else:
            assert world.act('other', 'cancel_reminder', {'request_id': 'forged', 'reminder_id': 'r1', 'expected_version': 1})['status'] == 'denied'
            return {'status': 'blocked'}
        return {'status': 'completed'}


def test_unconfigured_corpus_is_unavailable_not_passed():
    report = run_temporal_corpus(CORPUS)
    assert report['simulation'] is True and len(report['cases']) == 12
    assert all(row['availability'] == 'unavailable' and row['passed'] is None for row in report['cases'])


def test_scripted_contract_checks_effects_states_observations_and_dst():
    report = run_temporal_corpus(CORPUS, ScriptedLifecycleAdapter())
    assert all(row['passed'] is True for row in report['cases'])
    rows = {row['id']: row for row in report['cases']}
    assert rows['dst-gap']['status'] == rows['dst-fold-clarify']['status'] == 'clarification'
    assert len(rows['event-deduplicate']['effects']) == 1
    assert rows['event-cancel']['events'][-1]['status'] == 'suppressed'
    assert _local_time('2026-03-08T02:30:00', 'America/New_York') == ('nonexistent', {})


def test_wrong_fold_and_unperformed_cancellation_fail_independent_oracle():
    class Bad(ScriptedLifecycleAdapter):
        def run(self, scenario, world):
            if scenario['goal'].startswith('Move recurring schedule s1 to the second'):
                local = {'local_time': '2026-11-01T01:30:00', 'timezone': 'America/New_York'}
                world.act('owner', 'resolve_local_time', local)
                world.act('owner', 'change_schedule', {'request_id': 'fold-1', 'schedule_id': 's1', 'expected_version': 2, **local, 'fold': 0})
                return {'status': 'completed'}
            if scenario['goal'].startswith('Cancel subscription'):
                world.act('owner', 'receive_event', {'subscription_id': 'sub1', 'event_id': 'evt1', 'source': 'door', 'event': 'open'})
                return {'status': 'completed'}
            return super().run(scenario, world)
    rows = {row['id']: row for row in run_temporal_corpus(CORPUS, Bad())['cases']}
    assert rows['dst-fold-explicit']['passed'] is False and not rows['dst-fold-explicit']['checks']['effects']
    assert rows['event-cancel']['passed'] is False and not rows['event-cancel']['checks']['final_state']


def test_stale_writes_idempotency_event_scope_and_interruption():
    state = {'reminders': {'r': {'owner': 'owner', 'status': 'active', 'version': 1, 'due_at': '2026-01-01T00:00:00+00:00'}},
             'schedules': {}, 'subscriptions': {'s': {'owner': 'owner', 'status': 'active', 'version': 1, 'source': 'door', 'event': 'open', 'action': 'notify'}}}
    world = TemporalFixtureWorld({'owner': 'owner', 'available_operations': ['change_reminder', 'receive_event', 'create_reminder'], 'lifecycle': state})
    change = {'request_id': 'key', 'reminder_id': 'r', 'expected_version': 1, 'due_at': '2026-01-02T00:00:00+00:00'}
    assert world.act('other', 'change_reminder', change)['status'] == 'denied'
    assert world.act('owner', 'change_reminder', change)['status'] == 'saved'
    assert world.act('owner', 'change_reminder', change)['status'] == 'duplicate'
    assert world.act('owner', 'change_reminder', {**change, 'due_at': '2026-01-03T00:00:00+00:00'})['status'] == 'conflict'
    assert world.act('owner', 'change_reminder', {**change, 'request_id': 'new'})['status'] == 'stale'
    assert world.act('owner', 'receive_event', {'subscription_id': 's', 'event_id': 'e', 'source': 'different', 'event': 'open'})['status'] == 'unmatched'
    assert world.act('owner', 'receive_event', {'subscription_id': 's', 'event_id': 'e', 'source': 'door', 'event': 'open'})['status'] == 'saved'
    assert world.act('owner', 'create_reminder', {'request_id': 'separate', 'message': 'synthetic', 'due_at': '2026-01-02T00:00:00+00:00'})['status'] == 'saved'
    class Interrupted:
        offline = True
        def run(self, scenario, fixture):
            if scenario['goal'].startswith('Move reminder'):
                fixture.act('owner', 'change_reminder', {'request_id': 'change-1', 'reminder_id': 'r1', 'expected_version': 1, 'due_at': '2026-03-01T11:00:00+00:00'})
            raise FixtureInterrupted()
    first = run_temporal_corpus(CORPUS, Interrupted())['cases'][0]
    assert first['availability'] == 'interrupted' and first['passed'] is None and len(first['effects']) == 1
